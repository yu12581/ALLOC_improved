"""[S1D-Phase1] Paired baseline(alocc) vs combo(alocc_loss) report, 4 configs x 3 seeds.

Baseline dirs: s1_c{c}_r{r}_p{pTag}_seed{seed}_redline      (reused from A4-SEED sweep)
Combo    dirs: s1d_c{c}_r{r}_p{pTag}_seed{seed}_redline     (from run_s1d_phase1.ps1)

Go/No-Go gates (Phase 1 of S1 + Distortion combo, see PROJECT_LOG §2.8.22+ proposal):
  (a) combo clean-pass >= baseline clean-pass + 1 on the {c0 r8 p0.3, c3 r8 p0.3} fragile pair
  (b) mean(combo.raw_auc) - mean(baseline.raw_auc) >= +0.05 across all 12 runs
  (c) mean(combo.ssim_gap) - mean(baseline.ssim_gap) >= +0.05
Pass any one => Go to Phase 2.

Output: ALOCC_paper/s1d_phase1.md
"""
from __future__ import annotations
import json, math, pathlib, statistics

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")
CONFIGS = [(0, 8, 0.3), (3, 8, 0.3), (7, 4, 0.3), (7, 4, 0.5)]
SEEDS = [42, 1337, 2026]


def pTag(p): return f"{int(p*10):02d}"


def load(prefix, c, r, p, s):
    d = ROOT / f"{prefix}_c{c}_r{r}_p{pTag(p)}_seed{s}_redline" / "experiment" / "summary.json"
    if not d.exists():
        return None
    j = json.loads(d.read_text(encoding="utf-8"))
    bm = j.get("best_metrics", {})
    si = j.get("selection_info", {})
    rl_fb = bool(si.get("redline_fallback_triggered", False))
    ssim_ic = float(bm.get("ssim_ic", float("nan")))
    ssim_oc = float(bm.get("ssim_oc", float("nan")))
    raw_auc = float(bm.get("raw_auc", 0.0))
    return {
        "best_epoch": j.get("best_epoch"),
        "auc": float(bm.get("auc", float("nan"))),
        "raw_auc": raw_auc,
        "ssim_ic": ssim_ic,
        "ssim_oc": ssim_oc,
        "ssim_gap": ssim_ic - ssim_oc,
        "rl_fallback": rl_fb,
        "clean_pass": (not rl_fb) and (ssim_oc <= 0.15) and (raw_auc >= 0.60),
    }


pairs = []  # list of dicts with baseline_* and combo_* keys
for c, r, p in CONFIGS:
    for s in SEEDS:
        b = load("s1", c, r, p, s)
        k = load("s1d", c, r, p, s)
        pairs.append({"c": c, "r": r, "p": p, "seed": s, "base": b, "combo": k})


def f4(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "-"
    if isinstance(v, bool): return "Y" if v else "N"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def delta(a, b, key):
    if a is None or b is None: return None
    return b[key] - a[key]


lines = []
lines.append("# [S1D-Phase1] S1 + Distortion Go/No-Go (4 cfg x 3 seed = 12 pairs)\n")
lines.append("Baseline = `--variant alocc` (S1 only; reused from A4-SEED sweep).\n")
lines.append("Combo    = `--variant alocc_loss` with `--d-outclass-loss-scale 0.1`, `--out-per-class-count 300`.\n")
lines.append("Clean-pass = redline_fallback==N AND ssim_oc<=0.15 AND raw_auc>=0.60.\n")
lines.append("")

lines.append("## Per-pair detail\n")
lines.append("| class | r | p | seed | who | ep | raw_auc | ssim_ic | ssim_oc | gap | rl_fb | clean |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for pr in pairs:
    for who, rec in (("base", pr["base"]), ("combo", pr["combo"])):
        if rec is None:
            lines.append(f"| {pr['c']} | {pr['r']} | {pr['p']} | {pr['seed']} | {who} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {pr['c']} | {pr['r']} | {pr['p']} | {pr['seed']} | {who} | "
            f"{f4(rec['best_epoch'])} | {f4(rec['raw_auc'])} | {f4(rec['ssim_ic'])} | "
            f"{f4(rec['ssim_oc'])} | {f4(rec['ssim_gap'])} | {f4(rec['rl_fallback'])} | {f4(rec['clean_pass'])} |"
        )

lines.append("\n## Per-config paired aggregate\n")
lines.append("| class | r | p | base clean/3 | combo clean/3 | d_raw_auc | d_ssim_ic | d_ssim_oc | d_gap |")
lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for c, r, p in CONFIGS:
    grp = [pr for pr in pairs if pr["c"] == c and pr["r"] == r and pr["p"] == p]
    base_cp = sum(1 for x in grp if x["base"] and x["base"]["clean_pass"])
    combo_cp = sum(1 for x in grp if x["combo"] and x["combo"]["clean_pass"])
    def dmean(k):
        ds = [delta(x["base"], x["combo"], k) for x in grp if x["base"] and x["combo"]]
        return statistics.mean(ds) if ds else float("nan")
    lines.append(f"| {c} | {r} | {p} | {base_cp}/3 | {combo_cp}/3 | "
                 f"{dmean('raw_auc'):+.4f} | {dmean('ssim_ic'):+.4f} | "
                 f"{dmean('ssim_oc'):+.4f} | {dmean('ssim_gap'):+.4f} |")

valid = [pr for pr in pairs if pr["base"] and pr["combo"]]
total_pairs = len(valid)
base_clean = sum(1 for x in valid if x["base"]["clean_pass"])
combo_clean = sum(1 for x in valid if x["combo"]["clean_pass"])
d_raw_auc = statistics.mean([delta(x["base"], x["combo"], "raw_auc") for x in valid]) if valid else float("nan")
d_ssim_ic = statistics.mean([delta(x["base"], x["combo"], "ssim_ic") for x in valid]) if valid else float("nan")
d_ssim_oc = statistics.mean([delta(x["base"], x["combo"], "ssim_oc") for x in valid]) if valid else float("nan")
d_gap = statistics.mean([delta(x["base"], x["combo"], "ssim_gap") for x in valid]) if valid else float("nan")

# Fragile pair (c0 r8 p0.3, c3 r8 p0.3) focused gate
fragile = [pr for pr in valid if (pr["c"], pr["r"], pr["p"]) in ((0, 8, 0.3), (3, 8, 0.3))]
f_base = sum(1 for x in fragile if x["base"]["clean_pass"])
f_combo = sum(1 for x in fragile if x["combo"]["clean_pass"])

gate_a = (f_combo - f_base) >= 1
gate_b = d_raw_auc >= 0.05
gate_c = d_gap >= 0.05

lines.append("\n## Headline\n")
lines.append(f"- paired runs: **{total_pairs}/12**")
lines.append(f"- clean-pass  base vs combo: **{base_clean}/{total_pairs} -> {combo_clean}/{total_pairs}**  (delta = {combo_clean - base_clean:+d})")
lines.append(f"- mean delta raw_auc  : **{d_raw_auc:+.4f}**")
lines.append(f"- mean delta ssim_ic  : **{d_ssim_ic:+.4f}**")
lines.append(f"- mean delta ssim_oc  : **{d_ssim_oc:+.4f}**  (negative = further suppressed)")
lines.append(f"- mean delta ssim_gap : **{d_gap:+.4f}**  (positive = gap widened)")
lines.append(f"- fragile-pair clean  : base {f_base}/6 -> combo {f_combo}/6")

lines.append("\n## Go/No-Go gates\n")
lines.append(f"- (a) fragile-pair clean rescue (+1 required) : **{'PASS' if gate_a else 'FAIL'}**  ({f_base}/6 -> {f_combo}/6)")
lines.append(f"- (b) mean d_raw_auc >= +0.05                : **{'PASS' if gate_b else 'FAIL'}**  ({d_raw_auc:+.4f})")
lines.append(f"- (c) mean d_ssim_gap >= +0.05               : **{'PASS' if gate_c else 'FAIL'}**  ({d_gap:+.4f})")
decision = "GO to Phase 2" if (gate_a or gate_b or gate_c) else "NO-GO (reconsider design)"
lines.append(f"\n## Decision: **{decision}**\n")

out = ROOT / "s1d_phase1.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote: {out}")
print()
print("\n".join(lines))
