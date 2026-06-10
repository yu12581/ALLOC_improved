"""T2 Step 1: inject spectral normalization on Discriminator.

Effect (ADR-008):
- Discriminator wraps its 4 Conv2d + 1 Linear in `spectral_norm` when
  `spectral_norm_d=True`; when False, modules are used bare and no RNG is
  consumed at construction time -> bitwise parity with pre-T2 state.
- `_weights_init_normal` is extended to initialise the original weight under
  parametrization (so SN-wrapped modules still receive N(0, 0.02) init).
- `ALOCC.__init__` gains 1 kwarg `spectral_norm_d` forwarded to D.
- `ALOCC_LOSS`/`ALOCC_LOSS_CLS` inherit unchanged.

Idempotent. Writes `.t2_sn.bak`. Sentinel token: `[T2-SN]`.
"""
from __future__ import annotations
import shutil
from pathlib import Path

MODEL = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\model.py")
BAK   = Path(str(MODEL) + ".t2_sn.bak")
SENTINEL = "[T2-SN]"

# 1. import spectral_norm at top (inserted after `from torch.utils.data import DataLoader`)
IMPORT_OLD = "from torch.utils.data import DataLoader\n"
IMPORT_NEW = ("from torch.utils.data import DataLoader\n"
              "from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm  # [T2-SN]\n")

# 2. extend _weights_init_normal to respect parametrized weights
WI_OLD = ("def _weights_init_normal(m):\n"
          "    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):\n"
          "        if getattr(m, 'weight', None) is not None:\n"
          "            nn.init.normal_(m.weight, 0.0, 0.02)\n"
          "        if getattr(m, 'bias', None) is not None and m.bias is not None:\n"
          "            nn.init.constant_(m.bias, 0.0)\n")
WI_NEW = ("def _weights_init_normal(m):\n"
          "    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):\n"
          "        # [T2-SN] if weight is re-parametrized (spectral_norm), init the original tensor\n"
          "        _parametrized = hasattr(m, 'parametrizations') and 'weight' in getattr(m, 'parametrizations', {})\n"
          "        if _parametrized:\n"
          "            nn.init.normal_(m.parametrizations.weight.original, 0.0, 0.02)\n"
          "        elif getattr(m, 'weight', None) is not None:\n"
          "            nn.init.normal_(m.weight, 0.0, 0.02)\n"
          "        if getattr(m, 'bias', None) is not None and m.bias is not None:\n"
          "            nn.init.constant_(m.bias, 0.0)\n")

# 3. Discriminator.__init__ signature
D_INIT_OLD = "    def __init__(self, c_dim, df_dim, in_h, in_w=None):\n"
D_INIT_NEW = "    def __init__(self, c_dim, df_dim, in_h, in_w=None, spectral_norm_d: bool = False):  # [T2-SN]\n"

# 4. Discriminator Sequential body: wrap 4 Conv + 1 Linear in optional SN
D_SEQ_OLD = (
    "        self.logits = nn.Sequential(\n"
    "            nn.Conv2d(c_dim, df_dim, 5, 2, 2, bias=False),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            nn.Conv2d(df_dim, df_dim*2, 5, 2, 2, bias=False),\n"
    "            nn.BatchNorm2d(df_dim*2),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            nn.Conv2d(df_dim*2, df_dim*4, 5, 2, 2, bias=False),\n"
    "            nn.BatchNorm2d(df_dim*4),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            nn.Conv2d(df_dim*4, df_dim*8, 5, 2, 2, bias=False),\n"
    "            nn.BatchNorm2d(df_dim*8),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            nn.Flatten(),\n"
    "            nn.Linear(df_dim * 8 * h_after_conv * w_after_conv, 1, bias=True)\n"
    "        )\n"
)
D_SEQ_NEW = (
    "        _sn = _spectral_norm if spectral_norm_d else (lambda m: m)  # [T2-SN] identity when OFF -> zero RNG consumption\n"
    "        self.logits = nn.Sequential(\n"
    "            _sn(nn.Conv2d(c_dim, df_dim, 5, 2, 2, bias=False)),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            _sn(nn.Conv2d(df_dim, df_dim*2, 5, 2, 2, bias=False)),\n"
    "            nn.BatchNorm2d(df_dim*2),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            _sn(nn.Conv2d(df_dim*2, df_dim*4, 5, 2, 2, bias=False)),\n"
    "            nn.BatchNorm2d(df_dim*4),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            _sn(nn.Conv2d(df_dim*4, df_dim*8, 5, 2, 2, bias=False)),\n"
    "            nn.BatchNorm2d(df_dim*8),\n"
    "            nn.LeakyReLU(0.2, inplace=True),\n"
    "            nn.Flatten(),\n"
    "            _sn(nn.Linear(df_dim * 8 * h_after_conv * w_after_conv, 1, bias=True))\n"
    "        )\n"
)

# 5. ALOCC.__init__ tail kwargs + D construction
A_INIT_OLD = ("        bottleneck_noise_type: str = \"dropout\",  # [S1-BOT]\n"
              "    ):\n")
A_INIT_NEW = ("        bottleneck_noise_type: str = \"dropout\",  # [S1-BOT]\n"
              "        spectral_norm_d: bool = False,  # [T2-SN]\n"
              "    ):\n")
A_D_OLD = "        self.D = Discriminator(c_dim, df_dim, in_h=in_h).to(DEVICE, non_blocking=True)\n"
A_D_NEW = "        self.D = Discriminator(c_dim, df_dim, in_h=in_h, spectral_norm_d=spectral_norm_d).to(DEVICE, non_blocking=True)  # [T2-SN]\n"


def apply_patch() -> None:
    src = MODEL.read_text(encoding="utf-8")
    if SENTINEL in src:
        print(f"[skip]  {MODEL.name} already patched")
        return
    if not BAK.exists():
        shutil.copyfile(MODEL, BAK)
        print(f"[bak]   {BAK.name}")
    for i, (old, new) in enumerate([
        (IMPORT_OLD, IMPORT_NEW),
        (WI_OLD, WI_NEW),
        (D_INIT_OLD, D_INIT_NEW),
        (D_SEQ_OLD, D_SEQ_NEW),
        (A_INIT_OLD, A_INIT_NEW),
        (A_D_OLD, A_D_NEW),
    ], 1):
        assert old in src, f"anchor #{i} not matched; head={old[:60]!r}"
        src = src.replace(old, new, 1)
    MODEL.write_text(src, encoding="utf-8")
    print(f"[patch] {MODEL.name} (6 hunks, {SENTINEL})")


def main() -> None:
    apply_patch()
    import ast
    ast.parse(MODEL.read_text(encoding="utf-8"))
    print("[ok]    model.py AST parse")


if __name__ == "__main__":
    main()
