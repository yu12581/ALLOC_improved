"""[A4-SEED] Build a markdown report from the 3-seed x 4-config robustness sweep.

For each run: read experiment/summary.json and selection_info.
Redline clean-pass = selection_info.redline_fallback_triggered == False.
Emit per-config mean/std table and per-run detail table.
Output: ALOCC_paper/s1_seed_robustness.md
"""
from __future__ import annotations
import json, math, pathlib, statistics

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")

CONFIGS = [
    (0, 8, 0.3),
    (3, 8, 0.3),
    (7, 4, 0.3),
    (7, 4, 0.5),
]
SEEDS = [42, 1337, 2026]


def pTag(p): return f"{int(p*10):02d}"


def load(c, r, p, s):
    d = ROOT / f"s1_c{c}_r{r}_p{pTag(p)}_seed{s}_redline" / "experiment" / "summary.json"
    if not d.exists():
        return None
    j = json.loads(d.read_text(encoding="utf-8"))
    bm = j.get("best_metrics", {})
    si = j.get("selection_info", {})
    rl_fb = bool(si.get("redline_fallback_triggered", False))
    return {
        "best_epoch": j.get("best_epoch"),
        "auc": float(bm.get("auc", float("nan"))),
        "raw_auc": float(bm.get("raw_auc", float("nan"))),
        "ssim_ic": float(bm.get("ssim_ic", float("nan"))),
        "ssim_oc": float(bm.get("ssim_oc", float("nan"))),
        "rl_fallback": rl_fb,
        "clean_pass": (not rl_fb) and (bm.get("ssim_oc", 1.0) <= 0.15) and (bm.get("raw_auc", 0.0) >= 0.60),
    }


rows = []
for c, r, p in CONFIGS:
    for s in SEEDS:
        rec = load(c, r, p, s)
        if rec is None:
            rows.append({"class": c, "rank": r, "dropout": p, "seed": s, **{k: None for k in ["best_epoch","auc","raw_auc","ssim_ic","ssim_oc","rl_fallback","clean_pass"]}})
            continue
        rec.update({"class": c, "rank": r, "dropout": p, "seed": s})
        rows.append(rec)


def fmt(v, prec=4):
    if v is None: return "-"
    if isinstance(v, bool): return "Y" if v else "N"
    if isinstance(v, float):
        if math.isnan(v): return "-"
        return f"{v:.{prec}f}"
    return str(v)


lines = []
lines.append("# [A4-SEED] 3-seed robustness sweep (4 configs x 3 seeds = 12 runs)\n")
lines.append("**Configs** = C-2 per-class winners; **seeds** = {42, 1337, 2026}; seed=42 is the historical anchor.\n")
lines.append("**Clean-pass** = redline fallback NOT triggered AND ssim_oc<=0.15 AND raw_auc>=0.60.\n")
lines.append("")
lines.append("## Per-run detail\n")
lines.append("| class | rank | dropout | seed | best_ep | auc | raw_auc | ssim_ic | ssim_oc | rl_fb | clean |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for row in rows:
    lines.append(
        f"| {row['class']} | {row['rank']} | {row['dropout']} | {row['seed']} | "
        f"{fmt(row['best_epoch'])} | {fmt(row['auc'])} | {fmt(row['raw_auc'])} | "
        f"{fmt(row['ssim_ic'])} | {fmt(row['ssim_oc'])} | {fmt(row['rl_fallback'])} | {fmt(row['clean_pass'])} |"
    )

lines.append("")
lines.append("## Per-config aggregate\n")
lines.append("| class | rank | dropout | N | clean/N | raw_auc mean±std | ssim_oc mean±std | auc mean±std |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for c, r, p in CONFIGS:
    grp = [row for row in rows if row["class"] == c and row["rank"] == r and row["dropout"] == p and row["auc"] is not None]
    n = len(grp)
    cp = sum(1 for g in grp if g["clean_pass"])
    def agg(key):
        xs = [g[key] for g in grp]
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if n >= 2 else 0.0
        return f"{m:.4f} ± {s:.4f}"
    lines.append(f"| {c} | {r} | {p} | {n} | {cp}/{n} | {agg('raw_auc')} | {agg('ssim_oc')} | {agg('auc')} |")

total = len(rows)
total_clean = sum(1 for row in rows if row["clean_pass"])
lines.append("")
lines.append(f"## Headline\n")
lines.append(f"- total runs: **{total}**")
lines.append(f"- clean-pass: **{total_clean}/{total}**")
lines.append("")
lines.append("## Interpretation\n")
lines.append("Baseline C-2 claim: at seed=42 implicit, the 4 winners all pass redline (3 classes recovered).")
lines.append("Seed-42 column below should reproduce that (modulo code-drift).")
lines.append("Other seeds show **true variance** of the per-class rank choice: if clean-pass << 1 on them,")
lines.append("the C-2 'per-class rank tuning => 10/10' claim is a single-seed artefact and must be downgraded.")

out = ROOT / "s1_seed_robustness.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote: {out}")
print()
print("\n".join(lines))
