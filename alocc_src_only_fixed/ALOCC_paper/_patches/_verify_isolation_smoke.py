"""[BASELINE-A][ADR-007] Bit-for-bit isolation verification.

Phase 1 (--snapshot): captures current d1_s42 summary.json under _isolation_check/before/.
Phase 2 (--compare):  reads the freshly-rerun d1_s42 summary.json and diffs best_metrics
                      vs the snapshot. Also asserts switches.tf_verbatim_rmsprop == True.

Usage:
    python _verify_isolation_smoke.py --snapshot   # before patch validation
    # ... (run smoke via run_baseline_a_50.ps1 -Smoke) ...
    python _verify_isolation_smoke.py --compare
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27")
RUN  = ROOT / "d1_s42" / "summary.json"
CHECK_DIR = Path(r"d:\codeVS\ALOCC_paper\_patches\_isolation_check")
SNAP = CHECK_DIR / "d1_s42_before.json"

KEYS = [
    "acc", "auc", "raw_auc", "auc_gain",
    "ssim_ic", "ssim_oc", "ssim_gap",
    "raw_score_in_mean", "raw_score_out_mean", "raw_score_gap",
    "score_in_mean", "score_out_mean", "score_gap", "score_gap_gain",
    "paper_score",
]


def snapshot() -> int:
    if not RUN.exists():
        print(f"[ERR] no source summary at {RUN}")
        return 1
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN, SNAP)
    j = json.loads(SNAP.read_text(encoding="utf-8"))
    bm = j.get("best_metrics") or {}
    print(f"[OK] snapshot -> {SNAP}")
    print(f"     best_epoch = {j.get('best_epoch')}")
    print(f"     auc        = {bm.get('auc')}")
    print(f"     paper_score= {bm.get('paper_score')}")
    print(f"     switches.tf_verbatim_rmsprop = "
          f"{j.get('switches', {}).get('tf_verbatim_rmsprop', '<absent>')}")
    return 0


def compare() -> int:
    if not SNAP.exists():
        print(f"[ERR] no snapshot at {SNAP}; run --snapshot first")
        return 2
    if not RUN.exists():
        print(f"[ERR] no fresh run at {RUN}")
        return 2
    before = json.loads(SNAP.read_text(encoding="utf-8"))
    after  = json.loads(RUN.read_text(encoding="utf-8"))

    bm_b = before.get("best_metrics") or {}
    bm_a = after.get("best_metrics")  or {}
    sw_a = after.get("switches", {})

    print("=" * 70)
    print("BEFORE (legacy 50-run)        |  AFTER (smoke w/ --tf-verbatim-rmsprop)")
    print("=" * 70)

    diffs: list[tuple[str, float, float, float]] = []
    fmt = "{:<22s} {:>16}    |  {:>16}    diff={:+.3e}"
    for k in KEYS:
        vb = bm_b.get(k)
        va = bm_a.get(k)
        if vb is None or va is None:
            print(f"{k:<22s} {str(vb):>16s}    |  {str(va):>16s}    [missing]")
            continue
        d = float(va) - float(vb)
        if abs(d) > 0:
            diffs.append((k, float(vb), float(va), d))
        print(fmt.format(k, f"{vb:.6f}", f"{va:.6f}", d))

    print("-" * 70)
    print(f"best_epoch: before={before.get('best_epoch')}  after={after.get('best_epoch')}")
    print(f"switches.tf_verbatim_rmsprop (after) = {sw_a.get('tf_verbatim_rmsprop', '<absent>')}")
    print(f"switches.seed (after)                = {sw_a.get('seed')}")
    print("-" * 70)

    # Assertions
    if sw_a.get("tf_verbatim_rmsprop") is not True:
        print("[FAIL] switches.tf_verbatim_rmsprop is not True; flag not landed in summary.")
        return 3
    if before.get("best_epoch") != after.get("best_epoch"):
        print("[FAIL] best_epoch drift detected (D4-C broken).")
        return 4
    if not diffs:
        print("[PASS] bit-for-bit identical on all 15 best_metrics keys.")
        return 0

    # Floating-point tolerance: anything beyond 1e-9 indicates a real drift.
    max_abs = max(abs(d) for _, _, _, d in diffs)
    if max_abs < 1e-9:
        print(f"[PASS-FP] {len(diffs)} keys differ but max |Δ| = {max_abs:.2e} (FP noise, acceptable).")
        return 0

    print(f"[FAIL] {len(diffs)} key(s) drifted; max |Δ| = {max_abs:.6e}")
    for k, vb, va, d in diffs:
        print(f"   {k}: {vb} -> {va} (Δ={d:+e})")
    return 5


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", action="store_true")
    g.add_argument("--compare", action="store_true")
    a = p.parse_args()
    if a.snapshot:
        return snapshot()
    return compare()


if __name__ == "__main__":
    sys.exit(main())
