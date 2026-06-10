"""T2 Step 2: thread --spectral-norm-d through runner / export / fig6_7.

Effect (ADR-008):
- Adds `--spectral-norm-d` (store_true) to the three runners.
- Extends each `build_model(...)` to accept `spectral_norm_d` and forward it to
  ALOCC / ALOCC_LOSS / ALOCC_LOSS_CLS.
- Runner echoes the switch in `summary.json["switches"]`.
- Default False -> bitwise parity preserved.

Idempotent. Writes `.t2_sn.bak` per file. Sentinel: `[T2-SN]`.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
RUNNER = ROOT / "mnist_experiment_runner.py"
EXPORT = ROOT / "export_mnist_triplets.py"
FIG    = ROOT / "run_paper_mnist_figure6_7.py"
SENTINEL = "[T2-SN]"


def _patch(path: Path, edits: list[tuple[str, str]]) -> None:
    src = path.read_text(encoding="utf-8")
    if SENTINEL in src:
        print(f"[skip]  {path.name} already patched")
        return
    bak = Path(str(path) + ".t2_sn.bak")
    if not bak.exists():
        shutil.copyfile(path, bak)
        print(f"[bak]   {bak.name}")
    for i, (old, new) in enumerate(edits, 1):
        assert old in src, f"[fail] anchor #{i} not found in {path.name}: {old[:80]!r}"
        src = src.replace(old, new, 1)
    path.write_text(src, encoding="utf-8")
    print(f"[patch] {path.name} ({len(edits)} hunks)")


# --- runner ---
# S1 already built `_bk = dict(...)` on the line *after* the signature.
# We extend `_bk` in-place to include spectral_norm_d, then reuse `**_bk`
# in all 4 ALOCC variant calls (already threaded by S1).
RUNNER_EDITS = [
    ("def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0,  # [S1-BOT]\n"
     "                bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):\n"
     "    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type)\n",
     "def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0,  # [S1-BOT][T2-SN]\n"
     "                bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\",\n"
     "                spectral_norm_d: bool = False):\n"
     "    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type,\n"
     "               spectral_norm_d=spectral_norm_d)\n"),
    # call site (append arg)
    ("        bottleneck_noise_type=str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n"
     "    )\n\n    if args.variant in (\"alocc\", \"alocc_tiny\"):\n",
     "        bottleneck_noise_type=str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n"
     "        spectral_norm_d=bool(getattr(args, \"spectral_norm_d\", False)),  # [T2-SN]\n"
     "    )\n\n    if args.variant in (\"alocc\", \"alocc_tiny\"):\n"),
    # argparse (after bottleneck-noise-type)
    ("    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT] noise family for bottleneck\")\n",
     "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT] noise family for bottleneck\")\n"
     "    parser.add_argument(\"--spectral-norm-d\", action=\"store_true\", help=\"[T2-SN] apply spectral_norm to D's 4 Conv + Linear\")\n"),
    # summary switches echo
    ("            \"bottleneck_noise_type\": str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n        },\n",
     "            \"bottleneck_noise_type\": str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n"
     "            \"spectral_norm_d\": bool(getattr(args, \"spectral_norm_d\", False)),  # [T2-SN]\n"
     "        },\n"),
]

# --- export ---
EXPORT_EDITS = [
    ("def build_model(variant: str, bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):  # [S1-BOT]\n"
     "    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type)\n",
     "def build_model(variant: str, bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\",  # [S1-BOT][T2-SN]\n"
     "                spectral_norm_d: bool = False):\n"
     "    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type,\n"
     "               spectral_norm_d=spectral_norm_d)\n"),
    ("    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n",
     "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"
     "    parser.add_argument(\"--spectral-norm-d\", action=\"store_true\", help=\"[T2-SN]\")\n"),
    ("                        bottleneck_noise_type=str(getattr(args, \"bottleneck_noise_type\", \"dropout\")))\n"
     "    model._load_checkpoint(args.checkpoint)\n",
     "                        bottleneck_noise_type=str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n"
     "                        spectral_norm_d=bool(getattr(args, \"spectral_norm_d\", False)))  # [T2-SN]\n"
     "    model._load_checkpoint(args.checkpoint)\n"),
]

# --- fig6_7 ---
FIG_EDITS = [
    ("    bottleneck_noise_type: str = \"dropout\"\n\n\n"
     "def _figure7_scores(",
     "    bottleneck_noise_type: str = \"dropout\"\n"
     "    spectral_norm_d: bool = False  # [T2-SN]\n\n\n"
     "def _figure7_scores("),
    ("def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int,\n"
     "                    bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):  # [S1-BOT]\n"
     "    model = export_mnist_triplets.build_model(variant,\n"
     "                                              bottleneck_rank=bottleneck_rank,\n"
     "                                              bottleneck_dropout=bottleneck_dropout,\n"
     "                                              bottleneck_noise_type=bottleneck_noise_type)\n",
     "def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int,\n"
     "                    bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\",  # [S1-BOT][T2-SN]\n"
     "                    spectral_norm_d: bool = False):\n"
     "    model = export_mnist_triplets.build_model(variant,\n"
     "                                              bottleneck_rank=bottleneck_rank,\n"
     "                                              bottleneck_dropout=bottleneck_dropout,\n"
     "                                              bottleneck_noise_type=bottleneck_noise_type,\n"
     "                                              spectral_norm_d=spectral_norm_d)\n"),
    ("    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n",
     "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"
     "    parser.add_argument(\"--spectral-norm-d\", action=\"store_true\", help=\"[T2-SN]\")\n"),
    ("        bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "    )\n\n    mnist_experiment_runner.run_experiment(exp_args)",
     "        bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "        spectral_norm_d=args.spectral_norm_d,  # [T2-SN]\n"
     "    )\n\n    mnist_experiment_runner.run_experiment(exp_args)"),
    ("                                              bottleneck_noise_type=args.bottleneck_noise_type)\n"
     "    model._load_checkpoint(str(best_ckpt))\n",
     "                                              bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "                                              spectral_norm_d=args.spectral_norm_d)  # [T2-SN]\n"
     "    model._load_checkpoint(str(best_ckpt))\n"),
    ("        bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "    )\n    (out_dir / \"figure7_scores.json\")",
     "        bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "        spectral_norm_d=args.spectral_norm_d,  # [T2-SN]\n"
     "    )\n    (out_dir / \"figure7_scores.json\")"),
]


def main() -> None:
    _patch(RUNNER, RUNNER_EDITS)
    _patch(EXPORT, EXPORT_EDITS)
    _patch(FIG, FIG_EDITS)
    import ast
    for p in (RUNNER, EXPORT, FIG):
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"[ok]    {p.name} AST parse")


if __name__ == "__main__":
    main()
