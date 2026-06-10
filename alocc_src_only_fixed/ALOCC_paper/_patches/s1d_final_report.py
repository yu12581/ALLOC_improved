"""[S1D-FINAL] Phase 3 final validation report: 10 classes x 5 seeds at (m=0.6, s=0.3).

Layout:
  - s1d_final_c{c}_seed{seed}_redline/experiment/summary.json
  - s1d_final_c{c}_seed{seed}_redline.log  (UTF-16 LE + BOM, Tee-Object default)

Reports:
  (A) per-class aggregate (5 seeds): clean/5, raw_auc mean+/-std, ssim_oc mean+/-std
  (B) per-cell detail (50 rows)
  (C) overall headline: total clean/50, per-class pass rate, seed stability

Output: ALOCC_paper/s1d_final.md
"""
from __future__ import annotations
import json, math, pathlib, re, statistics

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")
CLASSES = list(range(10))
SEEDS = [42, 1337, 2026, 7, 123]

GOUT_RE = re.compile(r"g_out=([0-9.eE+\-]+)")


def dir_name(c, seed):
    return f"s1d_final_c{c}_seed{seed}_redline"


def log_path(c, seed):
    return ROOT / (dir_name(c, seed) + ".log")


def scan_gout(log_file: pathlib.Path):
    if not log_file.exists():
        return None, None
    try:
        raw = log_file.read_bytes()
    except Exception:
        return None, None
    if raw.startswith(b"\xff\xfe"):
        txt = raw.decode("utf-16", errors="ignore")
    elif raw.startswith(b"\xfe\xff"):
        txt = raw.decode("utf-16-be", errors="ignore")
    elif raw.startswith(b"\xef\xbb\xbf"):
        txt = raw.decode("utf-8-sig", errors="ignore")
    else:
        txt = raw.decode("utf-8", errors="ignore")
    vals = [float(m.group(1)) for m in GOUT_RE.finditer(txt) if m.group(1)]
    if not vals:
        return 0.0, 0.0
    return statistics.mean(vals), max(vals)


def load(c, seed):
    sp = ROOT / dir_name(c, seed) / "experiment" / "summary.json"
    if not sp.exists():
        return None
    j = json.loads(sp.read_text(encoding="utf-8"))
    bm = j.get("best_metrics", {})
    si = j.get("selection_info", {})
    rl_fb = bool(si.get("redline_fallback_triggered", False))
    ssim_ic = float(bm.get("ssim_ic", float("nan")))
    ssim_oc = float(bm.get("ssim_oc", float("nan")))
    raw_auc = float(bm.get("raw_auc", 0.0))
    gmean, gmax = scan_gout(log_path(c, seed))
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
for c in CLASSES:
    for s in SEEDS:
        rec = load(c, s)
        if rec is None:
            rows.append({"c": c, "seed": s, "missing": True})
            continue
        rec.update({"c": c, "seed": s, "missing": False})
        rows.append(rec)


def f4(v, prec=4):
    if v is None: return "-"
    if isinstance(v, bool): return "Y" if v else "N"
    if isinstance(v, float):
        if math.isnan(v): return "-"
        return f"{v:.{prec}f}"
    return str(v)


def stat_pair(vs):
    if not vs: return "-", "-"
    return f"{statistics.mean(vs):.4f}", (f"{statistics.stdev(vs):.4f}" if len(vs) > 1 else "0.0000")


lines = []
lines.append("# [S1D-FINAL] Phase 3: 10 classes x 5 seeds at (margin=0.6, scale=0.3)\n")
lines.append("Config: c in {0,1,2,3,4,5,6,8,9} use r=8 p=0.3; c=7 uses r=4 p=0.5.  variant=alocc_loss, d_out_scale=0.1.\n")
lines.append("Clean = redline_fallback=N AND ssim_oc<=0.15 AND raw_auc>=0.60.\n")

# (C) overall headline
valid = [x for x in rows if not x.get("missing")]
total_clean = sum(1 for x in valid if x["clean_pass"])
all_raw = [x["raw_auc"] for x in valid]
all_ssim_oc = [x["ssim_oc"] for x in valid]
all_gap = [x["ssim_gap"] for x in valid]
all_gout = [x["g_out_mean"] for x in valid if x["g_out_mean"] is not None]
lines.append("## (C) Overall headline\n")
lines.append(f"- total clean-pass : **{total_clean}/{len(valid)}** ({total_clean/max(len(valid),1):.1%})")
ram, ras = stat_pair(all_raw); lines.append(f"- raw_auc (all 50) : mean={ram}, std={ras}")
som, sos = stat_pair(all_ssim_oc); lines.append(f"- ssim_oc (all 50) : mean={som}, std={sos}")
gm, gs = stat_pair(all_gap); lines.append(f"- ssim_gap (all 50): mean={gm}, std={gs}")
if all_gout:
    lines.append(f"- g_out mean (hinge, all 50) : {statistics.mean(all_gout):.4e}  (verify hinge stays active)\n")

# (A) per-class aggregate
lines.append("## (A) Per-class aggregate (5 seeds each)\n")
lines.append("| class | cfg (r,p) | clean/5 | raw_auc mean +/- std | ssim_ic mean | ssim_oc mean +/- std | gap mean | best_ep mode |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for c in CLASSES:
    grp = [x for x in valid if x["c"] == c]
    if not grp:
        lines.append(f"| {c} | - | 0/5 | - | - | - | - | - |"); continue
    cp = sum(1 for g in grp if g["clean_pass"])
    cfg = "(4, 0.5)" if c == 7 else "(8, 0.3)"
    ra_m, ra_s = stat_pair([g["raw_auc"] for g in grp])
    ic_m, _ = stat_pair([g["ssim_ic"] for g in grp])
    oc_m, oc_s = stat_pair([g["ssim_oc"] for g in grp])
    gap_m, _ = stat_pair([g["ssim_gap"] for g in grp])
    eps = [g["best_epoch"] for g in grp if g["best_epoch"] is not None]
    mode_ep = statistics.mode(eps) if eps else "-"
    lines.append(f"| {c} | {cfg} | {cp}/5 | {ra_m} +/- {ra_s} | {ic_m} | {oc_m} +/- {oc_s} | {gap_m} | {mode_ep} |")

# (B) per-cell detail
lines.append("\n## (B) Per-cell detail (50 cells)\n")
lines.append("| class | seed | ep | raw_auc | ssim_ic | ssim_oc | gap | g_out mean | rl_fb | clean |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for x in rows:
    if x.get("missing"):
        lines.append(f"| {x['c']} | {x['seed']} | - | - | - | - | - | - | - | - |"); continue
    gout = x["g_out_mean"]; gstr = f"{gout:.2e}" if gout else "0.00e+00"
    lines.append(
        f"| {x['c']} | {x['seed']} | {f4(x['best_epoch'])} | {f4(x['raw_auc'])} | "
        f"{f4(x['ssim_ic'])} | {f4(x['ssim_oc'])} | {f4(x['ssim_gap'])} | "
        f"{gstr} | {f4(x['rl_fallback'])} | {f4(x['clean_pass'])} |"
    )

out = ROOT / "s1d_final.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote: {out}")
print()
for ln in lines[:30]:
    print(ln)
