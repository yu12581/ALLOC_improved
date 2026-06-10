"""Restore RM-3 CLI flags on mnist_experiment_runner.py (idempotent patch).

Context: `run_paper_mnist_figure6_7.py` already exposes the RM-3 flags correctly,
but the runner's own argparse block (used if anyone runs the runner standalone
via `python mnist_experiment_runner.py`) regressed at some point — it still has
`choices=["acc_auc", "paper"]` and is missing --distortion-alpha /
--distortion-beta / --paper-score-normalization. Internal plumbing
(_select_records, _attach_distortion_score, _normalize_metric_absolute) is
intact and reads the values via getattr(args, ..., default), so the figure6_7
path works fine — this patch only closes the standalone-CLI hole so PROJECT_LOG
§5 claim ("RM-3 CLI flags landed") matches reality.

Target: D:\\Trae_coding\\ALLOC\\ALOCC-master\\mnist_experiment_runner.py
Edits (idempotent):
  1. Replace `choices=["acc_auc", "paper"]` with `choices=["acc_auc", "paper", "distortion"]`
  2. Insert three new add_argument calls just before `return parser.parse_args()`.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\mnist_experiment_runner.py")
BACKUP = TARGET.with_suffix(".py.rm3_cli_restore.bak")

OLD_CHOICES = 'parser.add_argument("--selection-strategy", choices=["acc_auc", "paper"], default="acc_auc")'
NEW_CHOICES = 'parser.add_argument("--selection-strategy", choices=["acc_auc", "paper", "distortion"], default="acc_auc")'

NEW_FLAGS_BLOCK = '''    parser.add_argument("--distortion-alpha", type=float, default=1.0,
                        help="[RM-3a] exponent for ssim_gap in distortion strategy (default 1.0)")
    parser.add_argument("--distortion-beta", type=float, default=1.0,
                        help="[RM-3a] exponent for refined_auc in distortion strategy (default 1.0)")
    parser.add_argument("--paper-score-normalization", choices=["relative", "absolute"], default="relative",
                        help="[RM-3b] paper_score normalization mode; 'absolute' uses approved anchors")
'''
ANCHOR_LINE = "    return parser.parse_args()"


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: target not found: {TARGET}", file=sys.stderr)
        return 1

    src = TARGET.read_text(encoding="utf-8")
    changed = False

    # Edit 1: widen --selection-strategy choices
    if "distortion" not in src.split(OLD_CHOICES)[0].split("\n")[-1] if OLD_CHOICES in src else True:
        if OLD_CHOICES in src:
            src = src.replace(OLD_CHOICES, NEW_CHOICES, 1)
            changed = True
            print("[1/2] widened --selection-strategy choices with 'distortion'")
        else:
            # already has distortion? verify
            if 'choices=["acc_auc", "paper", "distortion"]' in src:
                print("[1/2] choices already contain 'distortion' (idempotent skip)")
            else:
                print(
                    "[1/2] WARNING: neither old nor new choices line found; "
                    "CLI layout may have diverged further — manual review needed"
                )

    # Edit 2: insert new flags just before `return parser.parse_args()`
    if "--distortion-alpha" in src:
        print("[2/2] --distortion-alpha already present (idempotent skip)")
    else:
        if ANCHOR_LINE not in src:
            print(
                "[2/2] ERROR: could not find anchor line `    return parser.parse_args()`",
                file=sys.stderr,
            )
            return 2
        src = src.replace(ANCHOR_LINE, NEW_FLAGS_BLOCK + ANCHOR_LINE, 1)
        changed = True
        print("[2/2] inserted --distortion-alpha / --distortion-beta / --paper-score-normalization")

    if not changed:
        print("No changes needed (patch already applied).")
        return 0

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"Backup saved to {BACKUP.name}")
    TARGET.write_text(src, encoding="utf-8")
    print(f"Patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
