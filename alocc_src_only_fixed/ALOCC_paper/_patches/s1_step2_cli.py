"""S1 Step 2: thread bottleneck CLI switches through runner / export / fig6_7.

Effect (ADR-008 summary):
- Adds 3 CLI flags `--bottleneck-rank`, `--bottleneck-dropout`, `--bottleneck-noise-type`
  to `mnist_experiment_runner.py`, `export_mnist_triplets.py`, `run_paper_mnist_figure6_7.py`.
- Extends each `build_model(...)` to forward the 3 kwargs to ALOCC/ALOCC_LOSS/ALOCC_LOSS_CLS.
- Runner echoes them in `summary.json["switches"]` for traceability.
- Fig6/7 propagates them to the second `build_model` (for triplet export) and to `_figure7_scores`.
- Defaults preserve bitwise parity (rank=0, dropout=0.0, noise_type='dropout').

Idempotent. Writes `.s1_bot.bak` per file. Sentinel: `[S1-BOT]`.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
RUNNER = ROOT / "mnist_experiment_runner.py"
EXPORT = ROOT / "export_mnist_triplets.py"
FIG    = ROOT / "run_paper_mnist_figure6_7.py"
SENTINEL = "[S1-BOT]"


def _patch(path: Path, edits: list[tuple[str, str]], must_all_apply: bool = True) -> None:
    src = path.read_text(encoding="utf-8")
    if SENTINEL in src:
        print(f"[skip]  {path.name} already patched")
        return
    bak = Path(str(path) + ".s1_bot.bak")
    if not bak.exists():
        shutil.copyfile(path, bak)
        print(f"[bak]   {bak.name}")
    for i, (old, new) in enumerate(edits, 1):
        if old not in src:
            if must_all_apply:
                raise SystemExit(f"[fail] anchor #{i} not found in {path.name}: {old[:80]!r}")
            continue
        src = src.replace(old, new, 1)
    path.write_text(src, encoding="utf-8")
    print(f"[patch] {path.name} ({len(edits)} hunks)")


# --- mnist_experiment_runner.py ---
RUNNER_EDITS = [
    # build_model signature
    ("def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0):\n",
     "def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0,  # [S1-BOT]\n"
     "                bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):\n"
     "    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type)\n"),
    # forward kwargs to all 4 variants
    ("        return ALOCC(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing)\n",
     "        return ALOCC(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n"),
    ("        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing)\n",
     "        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n"),
    ("        return ALOCC_LOSS(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing)\n",
     "        return ALOCC_LOSS(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n"),
    ("        return ALOCC_LOSS_CLS(in_h=28, out_h=28, lr=lr, classify=True, weight_decay=weight_decay, label_smoothing=label_smoothing)\n",
     "        return ALOCC_LOSS_CLS(in_h=28, out_h=28, lr=lr, classify=True, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n"),
    # call site
    ("        label_smoothing=getattr(args, \"label_smoothing\", 0.0) or 0.0,\n    )\n\n    if args.variant in (\"alocc\", \"alocc_tiny\"):\n",
     "        label_smoothing=getattr(args, \"label_smoothing\", 0.0) or 0.0,\n"
     "        bottleneck_rank=int(getattr(args, \"bottleneck_rank\", 0) or 0),  # [S1-BOT]\n"
     "        bottleneck_dropout=float(getattr(args, \"bottleneck_dropout\", 0.0) or 0.0),\n"
     "        bottleneck_noise_type=str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n"
     "    )\n\n    if args.variant in (\"alocc\", \"alocc_tiny\"):\n"),
    # argparse additions (after --label-smoothing)
    ("    parser.add_argument(\"--label-smoothing\", type=float, default=0.0)\n",
     "    parser.add_argument(\"--label-smoothing\", type=float, default=0.0)\n"
     "    parser.add_argument(\"--bottleneck-rank\", type=int, default=0, help=\"[S1-BOT] low-rank 1x1 bottleneck; 0=disabled\")\n"
     "    parser.add_argument(\"--bottleneck-dropout\", type=float, default=0.0, help=\"[S1-BOT] dropout/gaussian std applied after low-rank; 0=disabled\")\n"
     "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT] noise family for bottleneck\")\n"),
    # summary switches echo
    ("            \"distortion_beta\": float(getattr(args, \"distortion_beta\", 1.0)),\n        },\n",
     "            \"distortion_beta\": float(getattr(args, \"distortion_beta\", 1.0)),\n"
     "            \"bottleneck_rank\": int(getattr(args, \"bottleneck_rank\", 0) or 0),  # [S1-BOT]\n"
     "            \"bottleneck_dropout\": float(getattr(args, \"bottleneck_dropout\", 0.0) or 0.0),\n"
     "            \"bottleneck_noise_type\": str(getattr(args, \"bottleneck_noise_type\", \"dropout\")),\n"
     "        },\n"),
]

# --- export_mnist_triplets.py ---
EXPORT_EDITS = [
    ("def build_model(variant: str):\n    if variant == \"alocc\":\n        return ALOCC(in_h=28, out_h=28)\n    if variant == \"alocc_tiny\":\n        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8)\n    if variant == \"alocc_loss\":\n        return ALOCC_LOSS(in_h=28, out_h=28)\n    if variant == \"alocc_loss_cls\":\n        return ALOCC_LOSS_CLS(in_h=28, out_h=28, classify=True)\n",
     "def build_model(variant: str, bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):  # [S1-BOT]\n"
     "    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type)\n"
     "    if variant == \"alocc\":\n        return ALOCC(in_h=28, out_h=28, **_bk)\n"
     "    if variant == \"alocc_tiny\":\n        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, **_bk)\n"
     "    if variant == \"alocc_loss\":\n        return ALOCC_LOSS(in_h=28, out_h=28, **_bk)\n"
     "    if variant == \"alocc_loss_cls\":\n        return ALOCC_LOSS_CLS(in_h=28, out_h=28, classify=True, **_bk)\n"),
    ("    parser.add_argument(\"--abnormal-labels\", type=int, nargs=\"*\", default=None)\n",
     "    parser.add_argument(\"--abnormal-labels\", type=int, nargs=\"*\", default=None)\n"
     "    parser.add_argument(\"--bottleneck-rank\", type=int, default=0, help=\"[S1-BOT]\")\n"
     "    parser.add_argument(\"--bottleneck-dropout\", type=float, default=0.0, help=\"[S1-BOT]\")\n"
     "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"),
    ("    model = build_model(args.variant)\n    model._load_checkpoint(args.checkpoint)\n",
     "    model = build_model(args.variant,  # [S1-BOT]\n"
     "                        bottleneck_rank=int(getattr(args, \"bottleneck_rank\", 0) or 0),\n"
     "                        bottleneck_dropout=float(getattr(args, \"bottleneck_dropout\", 0.0) or 0.0),\n"
     "                        bottleneck_noise_type=str(getattr(args, \"bottleneck_noise_type\", \"dropout\")))\n"
     "    model._load_checkpoint(args.checkpoint)\n"),
]

# --- run_paper_mnist_figure6_7.py ---
FIG_EDITS = [
    # Args dataclass extension
    ("    paper_score_normalization: str = \"relative\"\n\n\ndef _figure7_scores(",
     "    paper_score_normalization: str = \"relative\"\n"
     "    bottleneck_rank: int = 0  # [S1-BOT]\n"
     "    bottleneck_dropout: float = 0.0\n"
     "    bottleneck_noise_type: str = \"dropout\"\n\n\n"
     "def _figure7_scores("),
    # _figure7_scores signature + build_model call
    ("def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int):\n    model = export_mnist_triplets.build_model(variant)\n",
     "def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int,\n"
     "                    bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):  # [S1-BOT]\n"
     "    model = export_mnist_triplets.build_model(variant,\n"
     "                                              bottleneck_rank=bottleneck_rank,\n"
     "                                              bottleneck_dropout=bottleneck_dropout,\n"
     "                                              bottleneck_noise_type=bottleneck_noise_type)\n"),
    # argparse (after --figure7-sample-count)
    ("    parser.add_argument(\"--figure7-sample-count\", type=int, default=40)\n",
     "    parser.add_argument(\"--figure7-sample-count\", type=int, default=40)\n"
     "    parser.add_argument(\"--bottleneck-rank\", type=int, default=0, help=\"[S1-BOT]\")\n"
     "    parser.add_argument(\"--bottleneck-dropout\", type=float, default=0.0, help=\"[S1-BOT]\")\n"
     "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"),
    # exp_args construction (append 3 fields before closing paren)
    ("        paper_score_normalization=args.paper_score_normalization,\n    )\n\n    mnist_experiment_runner.run_experiment(exp_args)",
     "        paper_score_normalization=args.paper_score_normalization,\n"
     "        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]\n"
     "        bottleneck_dropout=args.bottleneck_dropout,\n"
     "        bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "    )\n\n    mnist_experiment_runner.run_experiment(exp_args)"),
    # second build_model call (for triplets export)
    ("    model = export_mnist_triplets.build_model(args.variant)\n    model._load_checkpoint(str(best_ckpt))\n",
     "    model = export_mnist_triplets.build_model(args.variant,  # [S1-BOT]\n"
     "                                              bottleneck_rank=args.bottleneck_rank,\n"
     "                                              bottleneck_dropout=args.bottleneck_dropout,\n"
     "                                              bottleneck_noise_type=args.bottleneck_noise_type)\n"
     "    model._load_checkpoint(str(best_ckpt))\n"),
    # _figure7_scores call
    ("        sample_count=args.figure7_sample_count,\n    )\n    (out_dir / \"figure7_scores.json\")",
     "        sample_count=args.figure7_sample_count,\n"
     "        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]\n"
     "        bottleneck_dropout=args.bottleneck_dropout,\n"
     "        bottleneck_noise_type=args.bottleneck_noise_type,\n"
     "    )\n    (out_dir / \"figure7_scores.json\")"),
]


def main() -> None:
    _patch(RUNNER, RUNNER_EDITS)
    _patch(EXPORT, EXPORT_EDITS)
    _patch(FIG,    FIG_EDITS)
    import ast
    for p in (RUNNER, EXPORT, FIG):
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"[ok]    {p.name} AST parse")


if __name__ == "__main__":
    main()
