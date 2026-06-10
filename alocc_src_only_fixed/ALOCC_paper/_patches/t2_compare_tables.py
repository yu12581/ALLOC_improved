"""Collate S1-only vs T2 (SN+S1) best_metrics + epoch-1 diagnostic for {2, 6, 1}."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALOCC_paper")

VARIANTS = {
    "S1-only":  lambda c: ROOT / f"s1_c{c}_r16_p03" / "experiment" / "summary.json",
    "T2 SN+S1": lambda c: ROOT / f"t2_c{c}_sn_r16_p03" / "experiment" / "summary.json",
    "OFF (bl)": lambda c: ROOT / f"s1_c{c}_off" / "experiment" / "summary.json",
}
CLASSES = [2, 6, 1]

KEYS_BEST = ["auc", "raw_auc", "auc_gain", "ssim_ic", "ssim_oc", "ssim_gap",
             "score_gap", "raw_score_gap", "score_gap_gain", "paper_score"]

rows = []
epoch1 = []
for c in CLASSES:
    for name, p in VARIANTS.items():
        path = p(c)
        if not path.exists():
            rows.append({"class": c, "variant": name, "missing": True})
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        m = j.get("best_metrics", {})
        row = {"class": c, "variant": name, "best_epoch": j.get("best_epoch")}
        row.update({k: m.get(k) for k in KEYS_BEST})
        rows.append(row)
        # epoch-1 diagnostic (pre-GAN-dynamics)
        e1 = None
        for r in j.get("records", []):
            if isinstance(r, dict) and r.get("epoch") == 1:
                e1 = r
                break
        if e1 is None:
            continue
        epoch1.append({
            "class": c, "variant": name,
            "e1_auc": e1.get("auc"),
            "e1_ssim_oc": e1.get("ssim_oc"),
            "e1_raw_auc": e1.get("raw_auc"),
            "e1_score_gap": e1.get("score_gap"),
            "e1_raw_score_gap": e1.get("raw_score_gap"),
        })


def fmt(v, nd=4):
    if v is None: return "   -  "
    if isinstance(v, (int, bool)): return str(v)
    return f"{v:+.{nd}f}" if abs(v) < 10 else f"{v:.{nd}f}"


print("== Best-epoch metrics ==")
cols = ["class", "variant", "best_epoch"] + KEYS_BEST
print(" | ".join(f"{c:>10}" for c in cols))
print("-" * (len(cols) * 13))
for r in rows:
    if r.get("missing"):
        print(f"{r['class']:>10} | {r['variant']:>10} |  MISSING")
        continue
    line = [f"{r['class']:>10}", f"{r['variant']:>10}", f"{r['best_epoch']:>10}"]
    for k in KEYS_BEST:
        line.append(f"{fmt(r[k]):>10}")
    print(" | ".join(line))

print()
print("== Epoch-1 diagnostic (pre-GAN-collapse signal) ==")
print(f"{'class':>6} | {'variant':>10} | {'e1_auc':>10} | {'e1_ssim_oc':>12} | {'e1_raw_auc':>12} | {'e1_score_gap':>14} | {'e1_raw_score_gap':>18}")
print("-" * 98)
for r in epoch1:
    print(f"{r['class']:>6} | {r['variant']:>10} | {fmt(r['e1_auc']):>10} | {fmt(r['e1_ssim_oc']):>12} | {fmt(r['e1_raw_auc']):>12} | {fmt(r['e1_score_gap']):>14} | {fmt(r['e1_raw_score_gap']):>18}")

# save combined JSON
out = ROOT / "t2_compare.json"
out.write_text(json.dumps({"best": rows, "epoch1": epoch1}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nwritten: {out}")
