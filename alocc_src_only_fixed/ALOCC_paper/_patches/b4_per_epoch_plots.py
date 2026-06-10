"""[B-4 / C-1] Per-epoch visualization for pass classes {2, 6, 1} (row 1,
selection-fork example) and fail classes {0, 3, 7} (row 2, S1 ceiling example).

Output:
- ALOCC_paper/figures/b4_per_epoch_c{c}.png (6 files: 2,6,1,0,3,7)
- ALOCC_paper/figures/b4_per_epoch_grid.png (2x3 pass-on-top / fail-on-bottom)
"""
from __future__ import annotations
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

TAU_OC = 0.15
TAU_RAW = 0.60

PASS_CLASSES = [2, 6, 1]
FAIL_CLASSES = [0, 3, 7]
CLASSES = PASS_CLASSES + FAIL_CLASSES


def load_records(c: int) -> list[dict]:
    p = ROOT / f"s1_c{c}_r16_p03_redline" / "experiment" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")).get("records", [])


def pick_distortion(records):
    return max(records, key=lambda r: r.get("distortion_score", -1))


def pick_redline(records):
    elig = [r for r in records
            if r["ssim_oc"] <= TAU_OC and r["raw_auc"] >= TAU_RAW]
    return min(elig, key=lambda r: r["epoch"]) if elig else None


def pick_acc_auc(records):
    return max(records, key=lambda r: r.get("refined_auc", r.get("auc", 0)))


def _vline(ax, x, color, label, ymax=1.0):
    ax.axvline(x, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(x, ymax * 0.96, f" {label}", color=color,
            fontsize=8, ha="left", va="top", rotation=0)


def plot_one(ax, c: int):
    records = load_records(c)
    eps = [r["epoch"] for r in records]
    raw = [r["raw_auc"] for r in records]
    oc = [r["ssim_oc"] for r in records]
    ic = [r["ssim_ic"] for r in records]

    p_d = pick_distortion(records)
    p_r = pick_redline(records)
    p_a = pick_acc_auc(records)

    # primary axis: raw_auc (left, blue) + refined_auc faded
    color_raw = "#1f77b4"
    color_oc = "#d62728"
    color_ic = "#2ca02c"

    l1 = ax.plot(eps, raw, color=color_raw, marker="o", linewidth=2, label="raw_auc")
    ax.axhline(TAU_RAW, color=color_raw, linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylabel("raw_auc", color=color_raw)
    ax.tick_params(axis="y", labelcolor=color_raw)
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_xticks(eps)

    ax2 = ax.twinx()
    l2 = ax2.plot(eps, oc, color=color_oc, marker="s", linewidth=2,
                  label="ssim_oc (outlier recon)")
    l3 = ax2.plot(eps, ic, color=color_ic, marker="^", linewidth=1.5,
                  linestyle="--", alpha=0.7, label="ssim_ic (inlier recon)")
    ax2.axhline(TAU_OC, color=color_oc, linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("SSIM", color=color_oc)
    ax2.tick_params(axis="y", labelcolor=color_oc)
    ax2.set_ylim(0, 1)

    # strategy pick lines
    _vline(ax, p_d["epoch"], "#9467bd",
           f"distortion ep{p_d['epoch']}", ymax=1.0)
    if p_r is not None:
        _vline(ax, p_r["epoch"], "#2ca02c",
               f"redline ep{p_r['epoch']}", ymax=0.88)
    _vline(ax, p_a["epoch"], "#ff7f0e",
           f"acc_auc ep{p_a['epoch']}", ymax=0.76)

    ax.set_title(
        f"class {c}  (S1 r=16 p=0.3)\n"
        f"redline={'ep '+str(p_r['epoch']) if p_r else 'FAIL'} / "
        f"distortion=ep {p_d['epoch']}",
        fontsize=10)

    lines = l1 + l2 + l3
    ax.legend(lines, [l.get_label() for l in lines],
              loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    # individual
    for c in CLASSES:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        plot_one(ax, c)
        fig.tight_layout()
        out = FIGS / f"b4_per_epoch_c{c}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")

    # 2x3 grid: row 0 = pass classes, row 1 = fail classes
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2))
    for col, c in enumerate(PASS_CLASSES):
        plot_one(axes[0][col], c)
    for col, c in enumerate(FAIL_CLASSES):
        plot_one(axes[1][col], c)
    # Row labels via text on the left margin
    axes[0][0].annotate("PASS (S1 ceiling OK)", xy=(-0.15, 0.5),
                        xycoords="axes fraction", rotation=90, fontsize=12,
                        color="#2ca02c", ha="center", va="center", weight="bold")
    axes[1][0].annotate("FAIL (S1 ceiling hit)", xy=(-0.15, 0.5),
                        xycoords="axes fraction", rotation=90, fontsize=12,
                        color="#d62728", ha="center", va="center", weight="bold")
    fig.suptitle(
        "S1 (rank=16 p=0.3): per-epoch trajectories; "
        "dashed lines = thresholds (raw_auc≥0.60 / ssim_oc≤0.15); "
        "vertical = selection by strategy",
        fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    out = FIGS / "b4_per_epoch_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
