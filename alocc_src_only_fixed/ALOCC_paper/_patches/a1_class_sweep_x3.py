"""A1 class sweep under X3 selection configuration (RM-3 R2 all-open).

Identical to a1_class_sweep.py but passes the X3 selection flags:
- --selection-strategy distortion
- --paper-score-normalization absolute
- --selection-min-auc 0.0          (no AUC floor)
- --selection-epoch-start 1        (widen from 2)
- --selection-epoch-end 10         (widen from 6)
- --distortion-alpha 1.0
- --distortion-beta 1.0

All training anchors unchanged (ADR-006: epochs=10, train=4096, batch=64,
noise=0.31, r_alpha=0.2, lr=0.002). 10 classes sequential, ~4 min on CUDA.
Outputs: ALOCC_paper/a1_diagnostic_x3/class_{0..9}/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
PY = REPO / ".venv" / "Scripts" / "python.exe"
SCRIPT = REPO / "run_paper_mnist_figure6_7.py"
OUT_ROOT = Path(r"d:\codeVS\ALOCC_paper\a1_diagnostic_x3")


def run_one(inner_class: int) -> int:
    outliers = [c for c in range(10) if c != inner_class]
    out_dir = OUT_ROOT / f"class_{inner_class}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        str(PY),
        str(SCRIPT),
        "--output-dir", str(out_dir),
        "--variant", "alocc_loss",
        "--specific", str(inner_class),
        "--outlier-labels", *[str(c) for c in outliers],
        "--epochs", "10",
        "--train-count", "4096",
        "--test-inlier-count", "400",
        "--out-per-class-count", "100",
        "--batch-size", "64",
        "--noise-std", "0.31",
        "--r-alpha", "0.2",
        "--lr", "0.002",
        # --- X3 selection block ---
        "--selection-strategy", "distortion",
        "--paper-score-normalization", "absolute",
        "--selection-min-auc", "0.0",
        "--selection-epoch-start", "1",
        "--selection-epoch-end", "10",
        "--distortion-alpha", "1.0",
        "--distortion-beta", "1.0",
        # keep fallback logging on for consistency with A1 baseline
        "--selection-log-fallback",
    ]

    print(f"\n[class {inner_class}] launching...")
    result = subprocess.run(cmd, cwd=str(REPO))
    print(f"[class {inner_class}] exit={result.returncode}")
    return result.returncode


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    failed: list[int] = []
    for k in range(10):
        rc = run_one(k)
        if rc != 0:
            failed.append(k)
    print("\n===== sweep done =====")
    print(f"classes completed: {10 - len(failed)} / 10")
    if failed:
        print(f"failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
