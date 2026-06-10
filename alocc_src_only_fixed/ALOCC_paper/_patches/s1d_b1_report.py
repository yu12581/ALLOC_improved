"""[S1D-B1] Phase 2 hinge-activation report: scan margin x scale.

Grid: margin {0.2, 0.4, 0.6} x scale {0.1, 0.3} x 4 cfg x 3 seeds = 72 cells.
  - 12 cells at (m=0.2, s=0.1) reuse Phase 1 dirs: s1d_c{c}_r{r}_p{pTag}_seed{seed}_redline/
  - 60 new cells:                                  s1d_c{c}_r{r}_p{pTag}_seed{seed}_m{mTag}_s{sTag}_redline/

Reports three things:
  A. per-cell detail (raw_auc, ssim_ic, ssim_oc, clean, g_out_mean)
  B. per (margin x scale) aggregate (N=12 per stratum)
  C. hinge-activation test: does g_out_mean actually rise when margin rises?
     If yes, hinge is now doing work; if no, ALOCC hinge is structurally
     unsuited for MNIST-scale L1 and we must pivot (B2 early-stop or Round 2).

Output: ALOCC_paper/s1d_b1.md
"""
from __future__ import annotations
import json, math, pathlib, re, statistics

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")
CONFIGS = [(0, 8, 0.3), (3, 8, 0.3), (7, 4, 0.3), (7, 4, 0.5)]
SEEDS = [42, 1337, 2026]
MARGINS = [0.2, 0.4, 0.6]
SCALES = [0.1, 0.3]

GOUT_RE = re.compile(r"g_out=([0-9.eE+\-]+)")


def pTag(p): return f"{int(p*10):02d}"
def mTag(m): return f"{int(m*10):02d}"


def dir_name(c, r, p, s, margin, scale):
    if margin == 0.2 and scale == 0.1:
        return f"s1d_c{c}_r{r}_p{pTag(p)}_seed{s}_redline"
    return f"s1d_c{c}_r{r}_p{pTag(p)}_seed{s}_m{mTag(margin)}_s{mTag(scale)}_redline"


def log_path(c, r, p, s, margin, scale):
    return ROOT / (dir_name(c, r, p, s, margin, scale) + ".log")


def scan_gout(log_file: pathlib.Path):
    if not log_file.exists():
        return None, None
    try:
        raw = log_file.read_bytes()
    except Exception:
        return None, None
    # PowerShell's Tee-Object / Out-File default to UTF-16 LE w/ BOM on PS5.1.
    if raw.startswith(b"\xff\xfe"):
        txt = raw.decode("utf-16", errors="ignore")
    elif raw.startswith(b"\xfe\xff"):
        txt = raw.decode("utf-16-be", errors="ignore")
    elif raw.startswith(b"\xef\xbb\xbf"):
        txt = raw.decode("utf-8-sig", errors="ignore")
    else:
        txt = raw.decode("utf-8", errors="ignore")
    vals = []
    for m in GOUT_RE.finditer(txt):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue
    if not vals:
        return 0.0, 0.0
    return statistics.mean(vals), max(vals)


def load(c, r, p, s, margin, scale):
    d = ROOT / dir_name(c, r, p, s, margin, scale) / "experiment" / "summary.json"
    if not d.exists():
        return None
    j = json.loads(d.read_text(encoding="utf-8"))
    bm = j.get("best_metrics", {})
    si = j.get("selection_info", {})
    rl_fb = bool(si.get("redline_fallback_triggered", False))
    ssim_ic = float(bm.get("ssim_ic", float("nan")))
    ssim_oc = float(bm.get("ssim_oc", float("nan")))
    raw_auc = float(bm.get("raw_auc", 0.0))
    gmean, gmax = scan_gout(log_path(c, r, p, s, margin, scale))
    return {
        "best_epoch": j.get("best_epoch"),
        "auc": float(bm.get("auc", float("nan"))),
        "raw_auc": raw_auc,
        "ssim_ic": ssim_ic,
        "ssim_oc": ssim_oc,
        "ssim_gap": ssim_ic - ssim_oc,
        "rl_fallback": rl_fb,
        "clean_pass": (not rl_fb) and (ssim_oc <= 0.15) and (raw_auc >= 0.60),
        "g_out_mean": gmean,
        "g_out_max": gmax,
    }


rows = []
for c, r, p in CONFIGS:
    for s in SEEDS:
        for margin in MARGINS:
            for scale in SCALES:
                rec = load(c, r, p, s, margin, scale)
                if rec is None:
                    rows.append({"c": c, "r": r, "p": p, "seed": s, "margin": margin, "scale": scale, "missing": True})
                    continue
                rec.update({"c": c, "r": r, "p": p, "seed": s, "margin": margin, "scale": scale, "missing": False})
                rows.append(rec)


def f4(v, prec=4):
    if v is None: return "-"
    if isinstance(v, bool): return "Y" if v else "N"
    if isinstance(v, float):
        if math.isnan(v): return "-"
        return f"{v:.{prec}f}"
    return str(v)


def fsci(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "-"
    if v == 0: return "0.00e+00"
    return f"{v:.2e}"


lines = []
lines.append("# [S1D-B1] Phase 2: hinge activation by scanning margin x scale\n")
lines.append("Grid: margin {0.2, 0.4, 0.6} x scale {0.1, 0.3} x 4 cfg x 3 seeds = **72 cells** (12 reuse Phase 1 at m=0.2,s=0.1).\n")
lines.append("Clean = redline_fallback=N AND ssim_oc<=0.15 AND raw_auc>=0.60.  g_out = train-log distortion hinge value (ReLU(margin - L1)).\n")

# --- (C) Hinge activation test FIRST: this is the central question of B1. ---
lines.append("## (C) Hinge activation test\n")
lines.append("Aggregate over all 12 (c, seed) pairs per (margin, scale) cell.\n")
lines.append("| margin | scale | N | g_out mean (log avg) | g_out max | clean/N | raw_auc mean | ssim_oc mean | ssim_gap mean |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for margin in MARGINS:
    for scale in SCALES:
        grp = [x for x in rows if not x.get("missing") and x["margin"] == margin and x["scale"] == scale]
        n = len(grp)
        if n == 0:
            lines.append(f"| {margin} | {scale} | 0 | - | - | - | - | - | - |")
            continue
        cp = sum(1 for g in grp if g["clean_pass"])
        gmeans = [g["g_out_mean"] for g in grp if g["g_out_mean"] is not None]
        gmaxs = [g["g_out_max"] for g in grp if g["g_out_max"] is not None]
        lines.append(
            f"| {margin} | {scale} | {n} | {fsci(statistics.mean(gmeans) if gmeans else None)} | "
            f"{fsci(max(gmaxs) if gmaxs else None)} | {cp}/{n} | "
            f"{f4(statistics.mean([g['raw_auc'] for g in grp]))} | "
            f"{f4(statistics.mean([g['ssim_oc'] for g in grp]))} | "
            f"{f4(statistics.mean([g['ssim_gap'] for g in grp]))} |"
        )

# --- Hinge activation verdict ---
lines.append("\n### Verdict\n")
g02 = [x["g_out_mean"] for x in rows if not x.get("missing") and x["margin"] == 0.2 and x["g_out_mean"] is not None]
g06 = [x["g_out_mean"] for x in rows if not x.get("missing") and x["margin"] == 0.6 and x["g_out_mean"] is not None]
if g02 and g06:
    ratio = statistics.mean(g06) / max(statistics.mean(g02), 1e-9)
    activated = ratio >= 10.0
    lines.append(f"- mean g_out at margin=0.2 : **{statistics.mean(g02):.4e}**")
    lines.append(f"- mean g_out at margin=0.6 : **{statistics.mean(g06):.4e}**")
    lines.append(f"- activation ratio (0.6/0.2) : **{ratio:.2f}x**  ({'ACTIVATED' if activated else 'STILL DEAD'})")
else:
    lines.append("- insufficient data to compute activation ratio.")

# --- (B) per (margin, scale) clean/raw_auc ranking ---
lines.append("\n## (B) Per-stratum ranking (best cells first, by clean/N then raw_auc)\n")
lines.append("| margin | scale | clean/12 | raw_auc | ssim_ic | ssim_oc | gap |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
stratum_rows = []
for margin in MARGINS:
    for scale in SCALES:
        grp = [x for x in rows if not x.get("missing") and x["margin"] == margin and x["scale"] == scale]
        if not grp: continue
        cp = sum(1 for g in grp if g["clean_pass"])
        stratum_rows.append((
            cp, statistics.mean([g["raw_auc"] for g in grp]), margin, scale, grp,
        ))
stratum_rows.sort(key=lambda t: (-t[0], -t[1]))
for cp, ra, margin, scale, grp in stratum_rows:
    lines.append(
        f"| {margin} | {scale} | {cp}/12 | {f4(ra)} | "
        f"{f4(statistics.mean([g['ssim_ic'] for g in grp]))} | "
        f"{f4(statistics.mean([g['ssim_oc'] for g in grp]))} | "
        f"{f4(statistics.mean([g['ssim_gap'] for g in grp]))} |"
    )

# --- (A) per-cell detail (72 rows) ---
lines.append("\n## (A) Per-cell detail (72 cells)\n")
lines.append("| class | r | p | seed | m | s | ep | raw_auc | ssim_ic | ssim_oc | gap | g_out mean | rl_fb | clean |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for x in rows:
    if x.get("missing"):
        lines.append(f"| {x['c']} | {x['r']} | {x['p']} | {x['seed']} | {x['margin']} | {x['scale']} | - | - | - | - | - | - | - | - |")
        continue
    lines.append(
        f"| {x['c']} | {x['r']} | {x['p']} | {x['seed']} | {x['margin']} | {x['scale']} | "
        f"{f4(x['best_epoch'])} | {f4(x['raw_auc'])} | {f4(x['ssim_ic'])} | "
        f"{f4(x['ssim_oc'])} | {f4(x['ssim_gap'])} | {fsci(x['g_out_mean'])} | "
        f"{f4(x['rl_fallback'])} | {f4(x['clean_pass'])} |"
    )

out = ROOT / "s1d_b1.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote: {out}")
print()
# Print just the headline (C section) for quick glance
for ln in lines[:60]:
    print(ln)
