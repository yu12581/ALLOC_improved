"""Aggregate A1 class-sweep results into a per-class summary table.

Reads each class_K/experiment/summary.json and produces:
- ALOCC_paper/a1_diagnostic/_aggregate.json  (full data)
- ALOCC_paper/a1_diagnostic/_aggregate.md    (markdown table)

Reports for each K:
- best_epoch (paper-selected) and its refined_auc / ssim_ic / ssim_oc / ssim_gap / auc_gain
- best_epoch_by_auc (scan all 10 epochs) and its refined_auc / ssim_oc / ssim_gap
- best_epoch_by_distortion (ssim_gap * refined_auc) and its metrics
- fallback_triggered flag (paper strategy could not find auc>=0.95 in window)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "a1_diagnostic"


def scan_class(k: int) -> dict:
    summary_path = ROOT / f"class_{k}" / "experiment" / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    records = data["records"]
    best_paper = data["best_metrics"]
    sel = data.get("selection_info", {})

    by_auc = max(records, key=lambda r: r["refined_auc"])
    by_dist = max(records, key=lambda r: r["ssim_gap"] * r["refined_auc"])

    def extract(r: dict) -> dict:
        return {
            "epoch": r["epoch"],
            "refined_auc": round(r["refined_auc"], 4),
            "ssim_ic": round(r["ssim_ic"], 4),
            "ssim_oc": round(r["ssim_oc"], 4),
            "ssim_gap": round(r["ssim_gap"], 4),
            "auc_gain": round(r["auc_gain"], 4),
        }

    return {
        "class": k,
        "paper_best": extract(best_paper),
        "by_auc": extract(by_auc),
        "by_distortion": extract(by_dist),
        "fallback_triggered": sel.get("fallback_triggered", False),
        "fallback_reason": sel.get("fallback_reason", ""),
        "candidate_epochs": sel.get("candidate_epochs", []),
    }


def main() -> None:
    rows = [scan_class(k) for k in range(10)]
    (ROOT / "_aggregate.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# A1 Class Sweep Aggregate (Baseline A anchor)\n")
    lines.append("Anchor: epochs=10, train=4096, batch=64, noise=0.31, r_alpha=0.2, lr=0.002.\n")
    lines.append(
        "Paper selection window [2,6], min_auc=0.95 (fallback to in-window argmax on failure).\n"
    )

    header = (
        "| K | paper_ep | paper_auc | paper_ssim_oc | paper_ssim_gap | "
        "best_ep (by auc) | best_auc | best_ssim_oc | best_ssim_gap | "
        "dist_ep | dist_auc | dist_ssim_gap | fallback |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append("\n## Per-class summary\n")
    lines.append(header)
    lines.append(sep)
    for row in rows:
        p = row["paper_best"]
        a = row["by_auc"]
        d = row["by_distortion"]
        fb = "Y" if row["fallback_triggered"] else "N"
        lines.append(
            f"| {row['class']} | {p['epoch']} | {p['refined_auc']} | {p['ssim_oc']} | "
            f"{p['ssim_gap']} | {a['epoch']} | {a['refined_auc']} | {a['ssim_oc']} | "
            f"{a['ssim_gap']} | {d['epoch']} | {d['refined_auc']} | {d['ssim_gap']} | {fb} |"
        )

    by_auc_avg = sum(r["by_auc"]["refined_auc"] for r in rows) / 10
    by_paper_avg = sum(r["paper_best"]["refined_auc"] for r in rows) / 10
    ssim_oc_at_auc = sum(r["by_auc"]["ssim_oc"] for r in rows) / 10
    ssim_gap_at_auc = sum(r["by_auc"]["ssim_gap"] for r in rows) / 10

    lines.append("\n## Aggregate\n")
    lines.append(f"- Mean refined_auc (paper selection): **{by_paper_avg:.4f}**")
    lines.append(f"- Mean refined_auc (oracle by-auc):   **{by_auc_avg:.4f}**")
    lines.append(f"- Mean ssim_oc   @ oracle by-auc:     **{ssim_oc_at_auc:.4f}**")
    lines.append(f"- Mean ssim_gap  @ oracle by-auc:     **{ssim_gap_at_auc:.4f}**")
    fb_count = sum(1 for r in rows if r["fallback_triggered"])
    lines.append(f"- Fallback triggered: **{fb_count}/10** classes (paper window had no auc>=0.95)")

    (ROOT / "_aggregate.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
