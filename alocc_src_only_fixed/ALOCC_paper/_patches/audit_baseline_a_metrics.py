"""Audit Baseline A v2026-04-27 50-run matrix:
- per-cell raw_auc / auc / ssim_ic / ssim_oc / score_in / score_out
- detect bimodality (polarity flip) in raw_auc per class
- detect whether ssim values are computed in the right range
- emit a compact CSV-like dump for downstream analysis
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27")

KEYS = [
    "raw_auc", "auc", "auc_gain",
    "ssim_ic", "ssim_oc", "ssim_gap",
    "raw_score_in_mean", "raw_score_out_mean",
    "score_in_mean", "score_out_mean",
    "paper_score",
]

print(f"# Audit root: {ROOT}")
print(f"# Exists: {ROOT.exists()}\n")

if not ROOT.exists():
    raise SystemExit("Baseline A root not found")

rows = []
for d in range(10):
    for s in range(42, 47):
        f = ROOT / f"d{d}_s{s}" / "summary.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        bm = j.get("best_metrics") or {}
        row = {"d": d, "s": s}
        for k in KEYS:
            row[k] = bm.get(k)
        # polarity check: how far from 0.5
        ra = row["raw_auc"]
        if ra is not None:
            row["polarity"] = "POS" if ra >= 0.5 else "NEG"
            row["dist_from_05"] = abs(ra - 0.5)
        rows.append(row)

print(f"# Loaded {len(rows)} cells\n")

# header
hdr = ["d", "s", "raw_auc", "auc", "auc_gain", "ssim_ic", "ssim_oc",
       "ssim_gap", "raw_in", "raw_out", "ref_in", "ref_out", "polarity"]
print("|".join(hdr))
for r in rows:
    print("|".join([
        f"{r['d']}", f"{r['s']}",
        f"{r['raw_auc']:.4f}" if r['raw_auc'] is not None else "—",
        f"{r['auc']:.4f}" if r['auc'] is not None else "—",
        f"{r['auc_gain']:+.4f}" if r['auc_gain'] is not None else "—",
        f"{r['ssim_ic']:.4f}" if r['ssim_ic'] is not None else "—",
        f"{r['ssim_oc']:.4f}" if r['ssim_oc'] is not None else "—",
        f"{r['ssim_gap']:+.4f}" if r['ssim_gap'] is not None else "—",
        f"{r['raw_score_in_mean']:.4f}" if r['raw_score_in_mean'] is not None else "—",
        f"{r['raw_score_out_mean']:.4f}" if r['raw_score_out_mean'] is not None else "—",
        f"{r['score_in_mean']:.4f}" if r['score_in_mean'] is not None else "—",
        f"{r['score_out_mean']:.4f}" if r['score_out_mean'] is not None else "—",
        r.get("polarity", "?"),
    ]))

print("\n## Per-class polarity / bimodality\n")
print("class | n | raw_auc(mean±std) | min  | max  | POS/NEG split | flipped seeds")
for d in range(10):
    cell = [r for r in rows if r["d"] == d and r["raw_auc"] is not None]
    if not cell:
        continue
    vals = [r["raw_auc"] for r in cell]
    pos = sum(1 for v in vals if v >= 0.5)
    neg = len(vals) - pos
    flipped = [r["s"] for r in cell if r["raw_auc"] < 0.5]
    print(f"  {d}   | {len(cell)} | {mean(vals):.4f}±{pstdev(vals):.4f} | "
          f"{min(vals):.4f} | {max(vals):.4f} | {pos}/{neg}    | {flipped}")

# bimodality check: |auc - 0.5| distribution
print("\n## Bimodality check (|raw_auc - 0.5|)\n")
all_dists = [abs(r["raw_auc"] - 0.5) for r in rows if r["raw_auc"] is not None]
print(f"  mean |raw_auc-0.5|     = {mean(all_dists):.4f}")
print(f"  >=0.10 (clearly polar) = {sum(1 for d in all_dists if d>=0.10)}/{len(all_dists)}")
print(f"  >=0.25 (strongly polar)= {sum(1 for d in all_dists if d>=0.25)}/{len(all_dists)}")
print(f"  <0.05 (truly random)   = {sum(1 for d in all_dists if d<0.05)}/{len(all_dists)}")

# folded AUC: if we auto-correct polarity, what would mean be?
print("\n## Folded (polarity-corrected) raw_auc\n")
folded = [max(r["raw_auc"], 1 - r["raw_auc"]) for r in rows if r["raw_auc"] is not None]
print(f"  mean folded raw_auc = {mean(folded):.4f} ± {pstdev(folded):.4f}")
print(f"  (this is the AUC the model would yield if a polarity-aware sign was used)")

# SSIM sanity
print("\n## SSIM sanity (Baseline A)\n")
ic = [r["ssim_ic"] for r in rows if r["ssim_ic"] is not None]
oc = [r["ssim_oc"] for r in rows if r["ssim_oc"] is not None]
gap = [r["ssim_gap"] for r in rows if r["ssim_gap"] is not None]
print(f"  ssim_ic   = {mean(ic):.4f} ± {pstdev(ic):.4f}  (expected ~0.94 if R≈identity)")
print(f"  ssim_oc   = {mean(oc):.4f} ± {pstdev(oc):.4f}  (expected ~0.92 if R is generic copier)")
print(f"  ssim_gap  = {mean(gap):+.4f} ± {pstdev(gap):.4f}")
