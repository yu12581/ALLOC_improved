"""Aggregate A1 X3 sweep results and compare against the paper-selection baseline.

Reads from a1_diagnostic_x3/class_{0..9}/experiment/summary.json, joins with
a1_diagnostic/class_{0..9}/experiment/summary.json (original paper-selection
baseline), and emits:
- ALOCC_paper/a1_diagnostic_x3/_aggregate.json
- ALOCC_paper/a1_diagnostic_x3/_aggregate.md   (paper vs X3 Δ table)
"""
from __future__ import annotations

import json
from pathlib import Path

OLD_ROOT = Path(r"d:\codeVS\ALOCC_paper\a1_diagnostic")
NEW_ROOT = Path(r"d:\codeVS\ALOCC_paper\a1_diagnostic_x3")


def load(root: Path, k: int) -> dict:
    path = root / f"class_{k}" / "experiment" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def extract(best: dict) -> dict:
    # Field name is "refined_auc" in records; top-level "best_metrics" uses same name.
    auc = best.get("refined_auc", best.get("auc"))
    return {
        "epoch": best["epoch"],
        "auc": round(float(auc), 4),
        "ssim_ic": round(float(best["ssim_ic"]), 4),
        "ssim_oc": round(float(best["ssim_oc"]), 4),
        "ssim_gap": round(float(best["ssim_gap"]), 4),
    }


def main() -> None:
    rows = []
    for k in range(10):
        old = load(OLD_ROOT, k)
        new = load(NEW_ROOT, k)
        o = extract(old["best_metrics"])
        n = extract(new["best_metrics"])
        rows.append({
            "class": k,
            "paper": o,
            "x3": n,
            "delta_auc": round(n["auc"] - o["auc"], 4),
            "delta_ssim_oc": round(n["ssim_oc"] - o["ssim_oc"], 4),
            "delta_ssim_gap": round(n["ssim_gap"] - o["ssim_gap"], 4),
            "paper_fallback": old.get("selection_info", {}).get("fallback_triggered", False),
            "x3_fallback": new.get("selection_info", {}).get("fallback_triggered", False),
        })

    (NEW_ROOT / "_aggregate.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# A1 X3 Selection Sweep — paper vs X3 comparison\n")
    lines.append("X3 config: `--selection-strategy distortion --paper-score-normalization absolute "
                 "--selection-min-auc 0.0 --selection-epoch-start 1 --selection-epoch-end 10 "
                 "--distortion-alpha 1.0 --distortion-beta 1.0`.\n")
    lines.append("Training anchors identical to ADR-006 Baseline A.\n")

    lines.append("## Per-class comparison\n")
    lines.append("| K | paper_ep | paper_auc | paper_oc | paper_gap | x3_ep | x3_auc | x3_oc | x3_gap | ΔAUC | Δoc | Δgap | fb(p→x3) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        p, n = r["paper"], r["x3"]
        fb = f"{'Y' if r['paper_fallback'] else 'N'}→{'Y' if r['x3_fallback'] else 'N'}"
        lines.append(
            f"| {r['class']} | {p['epoch']} | {p['auc']} | {p['ssim_oc']} | {p['ssim_gap']} | "
            f"{n['epoch']} | {n['auc']} | {n['ssim_oc']} | {n['ssim_gap']} | "
            f"**{r['delta_auc']:+.4f}** | {r['delta_ssim_oc']:+.4f} | {r['delta_ssim_gap']:+.4f} | {fb} |"
        )

    mean_paper_auc = sum(r["paper"]["auc"] for r in rows) / 10
    mean_x3_auc = sum(r["x3"]["auc"] for r in rows) / 10
    mean_paper_oc = sum(r["paper"]["ssim_oc"] for r in rows) / 10
    mean_x3_oc = sum(r["x3"]["ssim_oc"] for r in rows) / 10
    mean_paper_gap = sum(r["paper"]["ssim_gap"] for r in rows) / 10
    mean_x3_gap = sum(r["x3"]["ssim_gap"] for r in rows) / 10

    lines.append("\n## Aggregate (mean over 10 classes)\n")
    lines.append(f"- **refined_auc** : paper={mean_paper_auc:.4f}  → x3={mean_x3_auc:.4f}  (Δ {mean_x3_auc-mean_paper_auc:+.4f})")
    lines.append(f"- **ssim_oc**     : paper={mean_paper_oc:.4f}  → x3={mean_x3_oc:.4f}  (Δ {mean_x3_oc-mean_paper_oc:+.4f})")
    lines.append(f"- **ssim_gap**    : paper={mean_paper_gap:.4f}  → x3={mean_x3_gap:.4f}  (Δ {mean_x3_gap-mean_paper_gap:+.4f})")

    fb_p = sum(1 for r in rows if r["paper_fallback"])
    fb_x3 = sum(1 for r in rows if r["x3_fallback"])
    lines.append(f"- fallback_triggered : paper={fb_p}/10  → x3={fb_x3}/10")

    mid = [r for r in rows if r["class"] in (2, 6, 9)]
    mean_mid_paper_auc = sum(r["paper"]["auc"] for r in mid) / 3
    mean_mid_x3_auc = sum(r["x3"]["auc"] for r in mid) / 3
    mean_mid_paper_oc = sum(r["paper"]["ssim_oc"] for r in mid) / 3
    mean_mid_x3_oc = sum(r["x3"]["ssim_oc"] for r in mid) / 3
    lines.append("\n## RM-1 target: mid classes {2, 6, 9}\n")
    lines.append(f"- mid refined_auc : paper={mean_mid_paper_auc:.4f} → x3={mean_mid_x3_auc:.4f}  (Δ {mean_mid_x3_auc-mean_mid_paper_auc:+.4f})")
    lines.append(f"- mid ssim_oc     : paper={mean_mid_paper_oc:.4f} → x3={mean_mid_x3_oc:.4f}  (Δ {mean_mid_x3_oc-mean_mid_paper_oc:+.4f})")

    out = "\n".join(lines)
    (NEW_ROOT / "_aggregate.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
