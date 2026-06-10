"""A1 first round: sweep Baseline A across all 10 MNIST classes as inlier.

Paper convention: inlier = class K, outliers = all other 9 classes.
ADR-006 anchor config: epochs=10, train=4096, batch=64, noise=0.31, r_alpha=0.2, lr=0.002.
Selection strategy = paper (window [2,6], min_auc=0.95) — matches Baseline A anchor.

Each run writes to ALOCC_paper/a1_diagnostic/class_K/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
PY = SRC / ".venv" / "Scripts" / "python.exe"
OUT_ROOT = REPO / "ALOCC_paper" / "a1_diagnostic"
LOG_ROOT = OUT_ROOT / "_logs"


def build_cmd(inlier: int, out_dir: Path) -> list[str]:
    outliers = [str(k) for k in range(10) if k != inlier]
    return [
        str(PY),
        "run_paper_mnist_figure6_7.py",
        "--output-dir", str(out_dir),
        "--variant", "alocc",
        "--specific", str(inlier),
        "--outlier-labels", *outliers,
        "--epochs", "10",
        "--train-count", "4096",
        "--batch-size", "64",
        "--eval-batch-size", "128",
        "--noise-std", "0.31",
        "--r-alpha", "0.2",
        "--lr", "0.002",
        "--selection-strategy", "paper",
        "--selection-epoch-start", "2",
        "--selection-epoch-end", "6",
        "--selection-min-auc", "0.95",
        "--triplet-count", "12",
        "--figure7-sample-count", "40",
    ]


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, object]] = {}
    total_t0 = time.time()

    for k in range(10):
        out_dir = OUT_ROOT / f"class_{k}"
        log_path = LOG_ROOT / f"class_{k}.log"
        print(f"[a1-sweep] === inlier={k} === -> {out_dir}", flush=True)
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as fh:
            rc = subprocess.call(
                build_cmd(k, out_dir),
                cwd=str(SRC),
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - t0
        results[k] = {"rc": rc, "elapsed_s": round(elapsed, 1)}
        print(f"[a1-sweep] inlier={k} rc={rc} elapsed={elapsed:.1f}s", flush=True)
        if rc != 0:
            print(f"[a1-sweep][WARN] inlier={k} failed, see {log_path}", flush=True)

    total = time.time() - total_t0
    print(f"[a1-sweep] total elapsed: {total:.1f}s", flush=True)
    (OUT_ROOT / "_sweep_manifest.json").write_text(
        json.dumps({"total_elapsed_s": round(total, 1), "runs": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
