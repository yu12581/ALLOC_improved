"""[BASELINE-A v2026-04-27] Aggregate the 5 seeds x 10 digits matrix.

Inputs : <root>/d{0..9}_s{42..46}/summary.json
Outputs: <root>/_aggregate.json + <root>/_aggregate.md (printed too).
"""
from __future__ import annotations
import json
import math
import os
import statistics as st
import sys

ROOT = r"D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27"
SEEDS = [42, 43, 44, 45, 46]
DIGITS = list(range(10))
KEYS = [
    "acc", "auc", "raw_auc", "auc_gain",
    "ssim_ic", "ssim_oc", "ssim_gap",
    "score_gap", "raw_score_gap", "score_gap_gain",
    "paper_score",
]


def load_one(digit: int, seed: int) -> dict:
    p = os.path.join(ROOT, f"d{digit}_s{seed}", "summary.json")
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    bm = s.get("best_metrics", {})
    out = {"best_epoch": s.get("best_epoch")}
    for k in KEYS:
        v = bm.get(k, s.get(k))
        out[k] = float(v) if v is not None else float("nan")
    return out


def mstd(vals: list[float]) -> tuple[float, float]:
    vs = [v for v in vals if not math.isnan(v)]
    if not vs:
        return float("nan"), float("nan")
    m = st.fmean(vs)
    s = st.stdev(vs) if len(vs) > 1 else 0.0
    return m, s


def main() -> int:
    grid: dict[int, dict[int, dict]] = {d: {} for d in DIGITS}
    for d in DIGITS:
        for sd in SEEDS:
            grid[d][sd] = load_one(d, sd)

    best_epoch_all = [grid[d][sd]["best_epoch"] for d in DIGITS for sd in SEEDS]
    d4c_ok = all(be == 10 for be in best_epoch_all)

    per_class = {}
    for d in DIGITS:
        per_class[d] = {}
        for k in KEYS:
            m, s = mstd([grid[d][sd][k] for sd in SEEDS])
            per_class[d][k] = {"mean": m, "std": s}

    global_stats = {}
    for k in KEYS:
        m, s = mstd([grid[d][sd][k] for d in DIGITS for sd in SEEDS])
        global_stats[k] = {"mean": m, "std": s}

    out = {
        "protocol": "Baseline A v2026-04-27 (TF1.15 verbatim D1-B/D2-B/D3-B/D4-C)",
        "matrix": {"seeds": SEEDS, "digits": DIGITS, "n_runs": 50},
        "d4c_last_epoch_check": {
            "all_best_epoch_eq_10": d4c_ok,
            "distinct_best_epochs": sorted(set(best_epoch_all)),
        },
        "per_class": per_class,
        "global": global_stats,
    }
    with open(os.path.join(ROOT, "_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Markdown report
    lines = []
    lines.append("# Baseline A v2026-04-27 · 50-run aggregate\n")
    lines.append(f"- Matrix: 5 seeds × 10 digits = 50 runs")
    lines.append(f"- Protocol: TF1.15 verbatim (D1-B BCE-on-noisy / D2-B RMSprop α=0.9 ε=1e-10 / D3-B src-level / D4-C last-epoch)")
    lines.append(f"- D4-C check: all best_epoch == 10 → **{'PASS' if d4c_ok else 'FAIL'}**  (distinct={sorted(set(best_epoch_all))})\n")

    lines.append("## Per-class (mean ± std over 5 seeds)\n")
    cols = ["acc", "auc", "raw_auc", "auc_gain", "ssim_oc", "ssim_gap", "score_gap_gain", "paper_score"]
    lines.append("| digit | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for d in DIGITS:
        row = [f"{d}"]
        for k in cols:
            m, s = per_class[d][k]["mean"], per_class[d][k]["std"]
            row.append(f"{m:.4f} ± {s:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Global aggregate (50 runs)\n")
    lines.append("| metric | mean ± std |")
    lines.append("|---|---|")
    for k in KEYS:
        m, s = global_stats[k]["mean"], global_stats[k]["std"]
        lines.append(f"| {k} | {m:.4f} ± {s:.4f} |")

    # rank classes by paper_score and ssim_oc
    rank_ps = sorted(DIGITS, key=lambda d: per_class[d]["paper_score"]["mean"], reverse=True)
    rank_oc = sorted(DIGITS, key=lambda d: per_class[d]["ssim_oc"]["mean"])
    lines.append("\n## Rankings\n")
    lines.append(f"- By paper_score (best→worst): {rank_ps}")
    lines.append(f"- By ssim_oc (lowest=R distorts outliers most, best→worst): {rank_oc}")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "_aggregate.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0 if d4c_ok else 2


if __name__ == "__main__":
    sys.exit(main())
