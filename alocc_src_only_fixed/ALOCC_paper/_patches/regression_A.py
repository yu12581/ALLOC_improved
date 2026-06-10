"""PR-A/PR-B regression gate (ADR-007 \u00a79.6).

Re-runs Baseline A with all new switches at their DEFAULT values (i.e. "OFF":
--stop-recon-threshold unset, --selection-min-auc-hard absent, new outclass
scales already 0.0 for the alocc variant) and checks the five North-Star
metrics against the CUDA anchor within a tight tolerance.

Anchor: ALOCC_paper/baselines_cuda/A/experiment/summary.json (best_epoch=2).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
PY = SRC / ".venv" / "Scripts" / "python.exe"
OUT = REPO / "ALOCC_paper" / "_patches" / "_regression_A"
ANCHOR = REPO / "ALOCC_paper" / "baselines_cuda" / "A" / "experiment" / "summary.json"

TOL = 1e-6
FIELDS = (
    "refined_auc",
    "auc_gain",
    "ssim_oc",
    "ssim_gap",
    "score_gap_gain",
)


def run() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY),
        "run_paper_mnist_figure6_7.py",
        "--output-dir", str(OUT),
        "--variant", "alocc",
        "--specific", "1",
        "--outlier-labels", "6", "7",
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
    print("[regression-A] launching baseline A (flags OFF by default)...")
    rc = subprocess.call(cmd, cwd=str(SRC))
    if rc != 0:
        sys.exit(f"[regression-A] baseline run failed rc={rc}")


def compare() -> int:
    new = json.loads((OUT / "experiment" / "summary.json").read_text(encoding="utf-8"))
    old = json.loads(ANCHOR.read_text(encoding="utf-8"))

    new_best = new["best_metrics"]
    old_best = old["best_metrics"]

    ok = True
    print(f"\n[regression-A] best_epoch anchor={old['best_epoch']} new={new['best_epoch']}")
    if new["best_epoch"] != old["best_epoch"]:
        ok = False
        print("  -> MISMATCH (best_epoch drift)")

    print(f"\n{'field':>16} | {'anchor':>14} {'new':>14} {'delta':>12}  status")
    print("-" * 70)
    for f in FIELDS:
        a = float(old_best[f])
        b = float(new_best[f])
        d = b - a
        status = "OK" if abs(d) <= TOL else "DRIFT"
        if abs(d) > TOL:
            ok = False
        print(f"{f:>16} | {a:>14.10f} {b:>14.10f} {d:>+12.3e}  {status}")

    sw = new.get("switches", {})
    print(f"\n[regression-A] switches: {json.dumps(sw, ensure_ascii=False)}")
    si = new.get("selection_info", {})
    print(f"[regression-A] selection_info.fallback_triggered={si.get('fallback_triggered')}")
    print(f"[regression-A] selection_info.fallback_reason={si.get('fallback_reason')}")

    return 0 if ok else 1


if __name__ == "__main__":
    run()
    sys.exit(compare())
