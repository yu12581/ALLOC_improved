"""Compose S1 R1-ext.A triplet comparison sheets.

For each of classes {2, 6, 1}:
- Stitch OFF.normal | ON.normal     -> compare_c{c}_normal.png
- Stitch OFF.abnormal | ON.abnormal -> compare_c{c}_abnormal.png
Each column gets a large banner header: "OFF  auc=...  ssim_oc=..." etc.
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"D:\Trae_coding\ALOCC_paper")
OUT  = ROOT / "s1_compare_sheets"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [2, 6, 1]
MODES   = [("off", "OFF (baseline)"), ("r16_p03", "ON  r=16 p=0.3 dropout")]
BANNER_H = 48
GAP = 24


def summary_metrics(cls: int, mode: str) -> dict:
    p = ROOT / f"s1_c{cls}_{mode}" / "experiment" / "summary.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    bm = j["best_metrics"]
    return {
        "best_epoch": j["best_epoch"],
        "auc": bm["auc"],
        "ssim_oc": bm["ssim_oc"],
        "ssim_ic": bm["ssim_ic"],
        "score_gap": bm["score_gap"],
    }


def compose_for_class(cls: int, kind: str) -> Path:
    imgs: list[Image.Image] = []
    banners: list[str] = []
    for mode, label in MODES:
        p = ROOT / f"s1_c{cls}_{mode}" / "triplets" / f"{kind}_triplets.png"
        if not p.exists():
            raise FileNotFoundError(p)
        imgs.append(Image.open(p))
        m = summary_metrics(cls, mode)
        banners.append(
            f"{label}  |  best_ep={m['best_epoch']}  auc={m['auc']:.4f}  "
            f"ssim_oc={m['ssim_oc']:.4f}  ssim_ic={m['ssim_ic']:.4f}  score_gap={m['score_gap']:+.4f}"
        )

    heights = [im.size[1] for im in imgs]
    widths  = [im.size[0] for im in imgs]
    total_h = max(heights) + BANNER_H
    total_w = sum(widths) + GAP * (len(imgs) - 1)

    canvas = Image.new("L", (total_w, total_h), color=0)
    draw = ImageDraw.Draw(canvas)

    # Class title (single line at very top of each panel's banner)
    x = 0
    for im, banner in zip(imgs, banners):
        draw.text((x + 4, 4), f"class {cls} / {kind}", fill=255)
        draw.text((x + 4, 22), banner, fill=255)
        canvas.paste(im, (x, BANNER_H))
        x += im.size[0] + GAP

    out = OUT / f"compare_c{cls}_{kind}.png"
    canvas.save(out)
    return out


def main() -> None:
    paths = []
    for cls in CLASSES:
        for kind in ("normal", "abnormal"):
            paths.append(compose_for_class(cls, kind))
    # 3x2 grid master sheet: rows = classes, cols = (normal, abnormal)
    # each cell is a compose_for_class output pasted in.
    cells = [[Image.open(OUT / f"compare_c{cls}_{kind}.png") for kind in ("normal", "abnormal")] for cls in CLASSES]
    col_w = max(c.size[0] for row in cells for c in row)
    row_h = max(c.size[1] for row in cells for c in row)
    GAP_R = 32
    master_w = col_w * 2 + GAP_R
    master_h = row_h * 3 + GAP_R * 2
    master = Image.new("L", (master_w, master_h), color=0)
    for ri, row in enumerate(cells):
        for ci, cell in enumerate(row):
            master.paste(cell, (ci * (col_w + GAP_R), ri * (row_h + GAP_R)))
    master_path = OUT / "MASTER_3x2.png"
    master.save(master_path)
    paths.append(master_path)

    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
