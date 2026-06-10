"""S1 Step 1: inject LowRankNoisyBottleneck into model.py.

What this patch does (per ADR-008 effect-only summary):
- Adds a small module `LowRankNoisyBottleneck` at the Generator's encoder/decoder seam.
- When all three new knobs default off (rank=0, dropout=0, noise_type='dropout'),
  the Generator falls back to a bare `nn.Identity()` at that seam, which consumes no
  RNG during weight-init and produces byte-identical forward outputs to the pre-S1 state.
- Extends `ALOCC.__init__` with 3 new kwargs that are forwarded to `Generator`.
- `ALOCC_LOSS` / `ALOCC_LOSS_CLS` inherit `__init__` unchanged.

Idempotent. Creates `.s1_bot.bak`. Sentinel token: `[S1-BOT]`.
"""
from __future__ import annotations
import shutil
from pathlib import Path

MODEL = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\model.py")
BAK   = Path(str(MODEL) + ".s1_bot.bak")
SENTINEL = "[S1-BOT]"

# --- injected module (added just before `class Generator(nn.Module):`) ---
BOTTLENECK_MODULE = '''\
class LowRankNoisyBottleneck(nn.Module):  # [S1-BOT] low-rank 1x1 sandwich + optional noise
    def __init__(self, channels: int, rank: int = 0, dropout: float = 0.0, noise_type: str = "dropout"):
        super().__init__()
        self.rank = int(rank)
        self.dropout_p = float(dropout)
        self.noise_type = str(noise_type)
        if self.rank > 0:
            self.down = nn.Conv2d(channels, self.rank, kernel_size=1, stride=1, padding=0, bias=False)
            self.up   = nn.Conv2d(self.rank, channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.down = None
            self.up = None
        if self.dropout_p > 0.0 and self.noise_type == "dropout":
            self.drop = nn.Dropout2d(p=self.dropout_p)
        else:
            self.drop = None

    def forward(self, x):
        if self.down is not None:
            x = self.up(self.down(x))
        if self.drop is not None:
            x = self.drop(x)
        elif self.dropout_p > 0.0 and self.noise_type == "gaussian" and self.training:
            x = x + torch.randn_like(x) * self.dropout_p
        return x


'''

# --- Generator.__init__ signature extension (drop in 3 kwargs) ---
GEN_INIT_OLD = "    def __init__(self, c_dim, gf_dim, df_dim, in_h, in_w=None, out_h=None, out_w=None, classify=False):\n"
GEN_INIT_NEW = ("    def __init__(self, c_dim, gf_dim, df_dim, in_h, in_w=None, out_h=None, out_w=None, classify=False,\n"
                "                 bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):  # [S1-BOT]\n")

# --- Generator: register bottleneck right BEFORE self.apply(_weights_init_normal) ---
GEN_APPLY_OLD = "        self.apply(_weights_init_normal)\n"
GEN_APPLY_NEW = (
    "        # [S1-BOT] bottleneck at encoder/decoder seam; Identity when all knobs off -> bitwise parity with baseline\n"
    "        if int(bottleneck_rank) > 0 or float(bottleneck_dropout) > 0.0:\n"
    "            self.bottleneck = LowRankNoisyBottleneck(df_dim * 8, rank=bottleneck_rank,\n"
    "                                                    dropout=bottleneck_dropout, noise_type=bottleneck_noise_type)\n"
    "        else:\n"
    "            self.bottleneck = nn.Identity()\n"
    "        self.apply(_weights_init_normal)\n"
)

# --- Generator.forward: pass through bottleneck between encoder and decoder ---
GEN_FORWARD_OLD = (
    "    def forward(self, x, classify=False):\n"
    "        enc_output = self.encoder(x)\n"
    "        if classify:\n"
    "            self.classifier_output = self.classifier(enc_output)\n"
    "        return self.decoder(enc_output)\n"
)
GEN_FORWARD_NEW = (
    "    def forward(self, x, classify=False):\n"
    "        enc_output = self.encoder(x)\n"
    "        enc_output = self.bottleneck(enc_output)  # [S1-BOT]\n"
    "        if classify:\n"
    "            self.classifier_output = self.classifier(enc_output)\n"
    "        return self.decoder(enc_output)\n"
)

# --- ALOCC.__init__ kwargs + forward to Generator ---
ALOCC_INIT_OLD = "        label_smoothing: float = 0.0,\n    ):\n"
ALOCC_INIT_NEW = (
    "        label_smoothing: float = 0.0,\n"
    "        bottleneck_rank: int = 0,  # [S1-BOT]\n"
    "        bottleneck_dropout: float = 0.0,  # [S1-BOT]\n"
    "        bottleneck_noise_type: str = \"dropout\",  # [S1-BOT]\n"
    "    ):\n"
)
ALOCC_G_OLD = "        self.G = Generator(c_dim, gf_dim, df_dim, in_h, classify=classify).to(DEVICE, non_blocking=True)\n"
ALOCC_G_NEW = (
    "        self.G = Generator(c_dim, gf_dim, df_dim, in_h, classify=classify,  # [S1-BOT]\n"
    "                           bottleneck_rank=bottleneck_rank,\n"
    "                           bottleneck_dropout=bottleneck_dropout,\n"
    "                           bottleneck_noise_type=bottleneck_noise_type).to(DEVICE, non_blocking=True)\n"
)


def apply_patch() -> None:
    src = MODEL.read_text(encoding="utf-8")
    if SENTINEL in src:
        print(f"[skip]  {MODEL.name} already patched")
        return
    if not BAK.exists():
        shutil.copyfile(MODEL, BAK)
        print(f"[bak]   {BAK.name}")
    # 1. inject module definition just before `class Generator(nn.Module):`
    anchor = "class Generator(nn.Module):\n"
    assert anchor in src, "anchor 'class Generator(...)' not found"
    src = src.replace(anchor, BOTTLENECK_MODULE + anchor, 1)
    # 2. extend Generator.__init__ signature
    assert GEN_INIT_OLD in src, "Generator.__init__ signature not matched"
    src = src.replace(GEN_INIT_OLD, GEN_INIT_NEW, 1)
    # 3. register self.bottleneck before apply(init)
    assert GEN_APPLY_OLD in src, "self.apply(_weights_init_normal) anchor not matched"
    src = src.replace(GEN_APPLY_OLD, GEN_APPLY_NEW, 1)
    # 4. route forward through bottleneck
    assert GEN_FORWARD_OLD in src, "Generator.forward body not matched"
    src = src.replace(GEN_FORWARD_OLD, GEN_FORWARD_NEW, 1)
    # 5. ALOCC.__init__ kwargs
    assert ALOCC_INIT_OLD in src, "ALOCC.__init__ tail not matched"
    src = src.replace(ALOCC_INIT_OLD, ALOCC_INIT_NEW, 1)
    # 6. ALOCC.__init__ Generator call
    assert ALOCC_G_OLD in src, "ALOCC Generator construction not matched"
    src = src.replace(ALOCC_G_OLD, ALOCC_G_NEW, 1)
    MODEL.write_text(src, encoding="utf-8")
    print(f"[patch] {MODEL.name} ({SENTINEL} injected in 6 places)")


def main() -> None:
    apply_patch()
    import ast
    ast.parse(MODEL.read_text(encoding="utf-8"))
    print("[ok]    model.py AST parse")


if __name__ == "__main__":
    main()
