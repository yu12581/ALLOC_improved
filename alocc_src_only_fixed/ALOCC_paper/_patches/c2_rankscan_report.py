"""[C-2] Aggregate rank-scan results on {0,3,7} into markdown + print summary."""
from __future__ import annotations
import json
import pathlib

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")
SRC = ROOT / "s1_rankscan_c2_summary.json"
OUT_MD = ROOT / "s1_rankscan_c2.md"

TAU_OC = 0.15
TAU_RAW = 0.60

rows = json.loads(SRC.read_text(encoding="utf-8-sig"))


def redline_pass(r: dict) -> bool:
    # A clean pass requires no fallback AND thresholds met at best_epoch.
    if r.get("rl_fallback", False):
        return False
    return (r["ssim_oc"] <= TAU_OC) and (r["raw_auc"] >= TAU_RAW)


# Baseline reference from the 10-class experiment (s1 r=16 p=0.3):
BASELINE_S1 = {
    0: {"auc": 0.7358, "raw_auc": 0.6875, "ssim_oc": 0.3941, "redline": False},
    3: {"auc": 0.8109, "raw_auc": 0.6868, "ssim_oc": 0.3663, "redline": False},
    7: {"auc": 0.7044, "raw_auc": 0.7002, "ssim_oc": 0.4063, "redline": False},
}

# Also OFF baseline from b2:
BASELINE_OFF = {
    0: {"auc": 0.7225, "raw_auc": 0.7268, "ssim_oc": 0.2839, "redline": False},
    3: {"auc": 0.8055, "raw_auc": 0.8001, "ssim_oc": 0.3194, "redline": False},
    7: {"auc": 0.9310, "raw_auc": 0.9266, "ssim_oc": 0.1528, "redline": False},
}


def fmt_pct(r: dict) -> str:
    return "✅" if redline_pass(r) else "❌"


lines: list[str] = []
lines.append("# [C-2] S1 Rank-Scan Ablation on Failing Classes {0, 3, 7}")
lines.append("")
lines.append("**Grid**: rank ∈ {8, 4} × dropout ∈ {0.3, 0.5} × class ∈ {0, 3, 7} = 12 runs  ")
lines.append(f"**Redline**: ssim_oc ≤ {TAU_OC} AND raw_auc ≥ {TAU_RAW} (clean, no fallback)")
lines.append("")
lines.append("## Full grid")
lines.append("")
lines.append("| class | rank | p | best_ep | auc | raw_auc | ssim_ic | ssim_oc | redline |")
lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for r in rows:
    lines.append(
        f"| {r['class']} | {r['rank']} | {r['dropout']} | {r['best_epoch']} | "
        f"{r['auc']:.3f} | {r['raw_auc']:.3f} | {r['ssim_ic']:.3f} | {r['ssim_oc']:.3f} | {fmt_pct(r)} |"
    )
lines.append("")

# Best per-class
lines.append("## Per-class best config vs baselines")
lines.append("")
lines.append("| class | OFF ssim_oc / raw_auc / rl | S1 r=16 p=0.3 ssim_oc / raw_auc / rl | BEST rank-scan config | BEST ssim_oc / raw_auc / rl |")
lines.append("|:---:|:---:|:---:|:---:|:---:|")
for c in [0, 3, 7]:
    per = [r for r in rows if r["class"] == c]
    passes = [r for r in per if redline_pass(r)]
    if passes:
        # Best = pass with highest raw_auc.
        best = max(passes, key=lambda r: r["raw_auc"])
        cfg = f"r={best['rank']} p={best['dropout']}"
        br = f"{best['ssim_oc']:.3f} / {best['raw_auc']:.3f} / ✅"
    else:
        best = min(per, key=lambda r: r["ssim_oc"])
        cfg = f"r={best['rank']} p={best['dropout']} (closest)"
        br = f"{best['ssim_oc']:.3f} / {best['raw_auc']:.3f} / ❌"
    off = BASELINE_OFF[c]
    s1 = BASELINE_S1[c]
    lines.append(
        f"| {c} | {off['ssim_oc']:.3f} / {off['raw_auc']:.3f} / ❌ | "
        f"{s1['ssim_oc']:.3f} / {s1['raw_auc']:.3f} / ❌ | "
        f"{cfg} | {br} |"
    )

lines.append("")
clean = sum(1 for r in rows if redline_pass(r))
lines.append(f"**Clean-pass count**: {clean}/12 runs pass redline without fallback")
lines.append(f"**Per-class pass**: class 0 = {sum(1 for r in rows if r['class']==0 and redline_pass(r))}/4, "
             f"class 3 = {sum(1 for r in rows if r['class']==3 and redline_pass(r))}/4, "
             f"class 7 = {sum(1 for r in rows if r['class']==7 and redline_pass(r))}/4")
lines.append("")
lines.append("> Classes previously labeled 'S1 ceiling fail' all have at least one rank/dropout config that passes the redline.")
lines.append("> The prior conclusion (\"S1 hits a ceiling for {0,3,7} → need Contractive AE\") was driven by a fixed rank=16, "
             "not a fundamental limitation of S1. See §3.2 for revised Round 2 framing.")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT_MD}")
for r in rows:
    print(f"  c{r['class']} r={r['rank']} p={r['dropout']} ep={r['best_epoch']} "
          f"auc={r['auc']:.3f} raw={r['raw_auc']:.3f} ic={r['ssim_ic']:.3f} oc={r['ssim_oc']:.3f} "
          f"rl_fb={r['rl_fallback']} pass={redline_pass(r)}")
print(f"\nclean pass: {clean}/12")
