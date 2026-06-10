"""[A4-SEED] Apply --seed CLI patch to ALOCC-master source.

ADR-007 contract:
- CLI flag `--seed` added to 3 entry points (runner / pipeline / triplet export).
- Default None = use module-level _CURRENT_SEED (42) = bitwise identical to prior.
- Int value updates _CURRENT_SEED so downstream no-arg calls (MNIST.py) inherit.
- `switches.seed` emitted in summary.json (None when flag absent).

This script:
1) Backs up originals to ALOCC_paper/_patches/_backups/a4_seed/<name>.orig
2) Verifies each old_str is unique before replacing
3) Writes the new content
4) Prints a diff summary

Idempotent: re-running after success detects "already patched" and bails.
"""
from __future__ import annotations
import pathlib
import shutil
import sys

BASE = pathlib.Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
BACKUP = pathlib.Path(r"D:\Trae_coding\ALOCC_paper\_patches\_backups\a4_seed")
BACKUP.mkdir(parents=True, exist_ok=True)

EDITS: list[tuple[str, str, str]] = []  # (filename, old, new)

# ---------- utils.py ----------
EDITS.append((
    "utils.py",
    "def set_random_seed():\n"
    "    random.seed(42)\n"
    "    numpy.random.seed(42)\n"
    "    torch.manual_seed(42)\n"
    "    torch.set_default_dtype(torch.float32)\n"
    "    if torch.cuda.is_available():\n"
    "        torch.cuda.manual_seed_all(42)\n"
    "        torch.backends.cudnn.deterministic = True\n"
    "        torch.backends.cudnn.benchmark = False\n"
    "    elif torch.xpu.is_available():\n"
    "        torch.xpu.manual_seed_all(42)\n"
    "        \n"
    "set_random_seed()",
    "_CURRENT_SEED = 42  # [A4-SEED]\n"
    "\n"
    "def set_random_seed(seed=None):\n"
    "    \"\"\"[A4-SEED] If seed is None reuse the module-level _CURRENT_SEED (default 42);\n"
    "    if an int is given, update _CURRENT_SEED then apply. Downstream no-arg calls\n"
    "    (e.g. MNIST.py) thus inherit the user-chosen seed. Behavior with seed=None or\n"
    "    set_random_seed() is bitwise identical to the historical fixed-42 implementation.\"\"\"\n"
    "    global _CURRENT_SEED\n"
    "    if seed is not None:\n"
    "        _CURRENT_SEED = int(seed)\n"
    "    s = _CURRENT_SEED\n"
    "    random.seed(s)\n"
    "    numpy.random.seed(s)\n"
    "    torch.manual_seed(s)\n"
    "    torch.set_default_dtype(torch.float32)\n"
    "    if torch.cuda.is_available():\n"
    "        torch.cuda.manual_seed_all(s)\n"
    "        torch.backends.cudnn.deterministic = True\n"
    "        torch.backends.cudnn.benchmark = False\n"
    "    elif torch.xpu.is_available():\n"
    "        torch.xpu.manual_seed_all(s)\n"
    "\n"
    "set_random_seed()",
))


# ---------- mnist_experiment_runner.py ----------
EDITS.append((
    "mnist_experiment_runner.py",
    "def run_experiment(args):\n"
    "    set_random_seed()\n",
    "def run_experiment(args):\n"
    "    set_random_seed(getattr(args, \"seed\", None))  # [A4-SEED]\n",
))
EDITS.append((
    "mnist_experiment_runner.py",
    "            \"redline_raw_auc_min\": float(getattr(args, \"redline_raw_auc_min\", 0.60)),  # [A1-SEL]\n"
    "        },\n",
    "            \"redline_raw_auc_min\": float(getattr(args, \"redline_raw_auc_min\", 0.60)),  # [A1-SEL]\n"
    "            \"seed\": (int(args.seed) if getattr(args, \"seed\", None) is not None else None),  # [A4-SEED]\n"
    "        },\n",
))
EDITS.append((
    "mnist_experiment_runner.py",
    "    parser.add_argument(\"--paper-score-normalization\", choices=[\"relative\", \"absolute\"], default=\"relative\",\n"
    "                        help=\"[RM-3b] paper_score normalization mode; 'absolute' uses approved anchors\")\n"
    "    return parser.parse_args()\n",
    "    parser.add_argument(\"--paper-score-normalization\", choices=[\"relative\", \"absolute\"], default=\"relative\",\n"
    "                        help=\"[RM-3b] paper_score normalization mode; 'absolute' uses approved anchors\")\n"
    "    parser.add_argument(\"--seed\", type=int, default=None,\n"
    "                        help=\"[A4-SEED] RNG seed override; None=42 (historical default, bitwise identical)\")\n"
    "    return parser.parse_args()\n",
))

# ---------- run_paper_mnist_figure6_7.py ----------
EDITS.append((
    "run_paper_mnist_figure6_7.py",
    "    redline_ssim_oc_max: float = 0.15  # [A1-SEL]\n"
    "    redline_raw_auc_min: float = 0.60  # [A1-SEL]\n",
    "    redline_ssim_oc_max: float = 0.15  # [A1-SEL]\n"
    "    redline_raw_auc_min: float = 0.60  # [A1-SEL]\n"
    "    seed: int | None = None  # [A4-SEED]\n",
))
EDITS.append((
    "run_paper_mnist_figure6_7.py",
    "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    set_random_seed()\n",
    "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"
    "    parser.add_argument(\"--seed\", type=int, default=None, help=\"[A4-SEED] RNG seed; None=42 (historical default)\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    set_random_seed(args.seed)  # [A4-SEED]\n",
))
EDITS.append((
    "run_paper_mnist_figure6_7.py",
    "        paper_score_normalization=args.paper_score_normalization,\n"
    "        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]\n"
    "        bottleneck_dropout=args.bottleneck_dropout,\n"
    "        bottleneck_noise_type=args.bottleneck_noise_type,\n"
    "    )\n",
    "        paper_score_normalization=args.paper_score_normalization,\n"
    "        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]\n"
    "        bottleneck_dropout=args.bottleneck_dropout,\n"
    "        bottleneck_noise_type=args.bottleneck_noise_type,\n"
    "        seed=args.seed,  # [A4-SEED]\n"
    "    )\n",
))
EDITS.append((
    "run_paper_mnist_figure6_7.py",
    "        \"paper_score_normalization\": args.paper_score_normalization,\n"
    "    }\n"
    "    (out_dir / \"pipeline_summary.json\").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding=\"utf-8\")\n",
    "        \"paper_score_normalization\": args.paper_score_normalization,\n"
    "        \"seed\": args.seed,  # [A4-SEED]\n"
    "    }\n"
    "    (out_dir / \"pipeline_summary.json\").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding=\"utf-8\")\n",
))

# ---------- export_mnist_triplets.py ----------
EDITS.append((
    "export_mnist_triplets.py",
    "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    set_random_seed()\n",
    "    parser.add_argument(\"--bottleneck-noise-type\", choices=[\"dropout\", \"gaussian\"], default=\"dropout\", help=\"[S1-BOT]\")\n"
    "    parser.add_argument(\"--seed\", type=int, default=None, help=\"[A4-SEED] RNG seed; None=42 (historical default)\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    set_random_seed(args.seed)  # [A4-SEED]\n",
))


def apply_edit(fname: str, old: str, new: str) -> str:
    src = BASE / fname
    text = src.read_text(encoding="utf-8")
    # Idempotency: if new block is already present anywhere, skip (handles both
    # "old_str replaced" and "old_str was a prefix of new_str" cases).
    if new in text:
        return "ALREADY-PATCHED"
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"{fname}: old_str not found")
    if count > 1:
        raise RuntimeError(f"{fname}: old_str not unique (found {count})")
    # Backup once (before first edit on this file)
    bk = BACKUP / (fname + ".orig")
    if not bk.exists():
        shutil.copy2(src, bk)
    new_text = text.replace(old, new, 1)
    src.write_text(new_text, encoding="utf-8")
    return f"OK (+{len(new.splitlines()) - len(old.splitlines())} lines)"


def main() -> int:
    for fname, old, new in EDITS:
        try:
            status = apply_edit(fname, old, new)
            print(f"[{fname}] {status}")
        except Exception as e:
            print(f"[{fname}] FAIL: {e}")
            return 1
    print(f"\nBackups at: {BACKUP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
