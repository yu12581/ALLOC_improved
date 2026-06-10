"""RM-3 run analysis: tabulate per-epoch metrics + show selection rationale.

Covers two runs:
  R1 = `_rm3_run_C`            (window=[2,6], min_auc=0.60, distortion+absolute)
  R2 = `_rm3_run_C_nofilter`   (window=[1,10], no min_auc, distortion+absolute)
"""
import json
import sys
from pathlib import Path

RUNS = [
    ("R1 (windowed 2-6, min_auc=0.60)", Path(r"d:\codeVS\ALOCC_paper\_patches\_rm3_run_C")),
    ("R2 (full window 1-10, no filter)", Path(r"d:\codeVS\ALOCC_paper\_patches\_rm3_run_C_nofilter")),
]

ANCHOR_C = Path(r"d:\codeVS\ALOCC_paper\baselines_cuda\C\experiment\summary.json")


def _tabulate(tag: str, path: Path):
    S = json.loads((path / "experiment" / "summary.json").read_text(encoding="utf-8"))

    best = S["best_epoch"]
    sw = S["switches"]
    si = S["selection_info"]
    print("\n" + "=" * 100)
    print(f"  {tag}")
    print("=" * 100)
    print(f"  best_epoch={best}  strategy={sw['selection_strategy']}  "
          f"norm={sw['paper_score_normalization']}  "
          f"alpha/beta={sw['distortion_alpha']}/{sw['distortion_beta']}  "
          f"min_auc={sw['selection_min_auc']}")
    print(f"  candidate_epochs={si['candidate_epochs']}  eligible_epochs={si['eligible_epochs']}")
    print()
    print(f"  {'ep':>3}  {'win':>3}  {'elig':>4}  {'ssim_gap':>9}  {'refined_auc':>11}  "
          f"{'distortion':>10}  {'auc_gain':>9}  {'ssim_oc':>8}  {'acc':>6}  {'paper_abs':>10}")
    print("  " + "-" * 96)
    for r in S["records"]:
        ep = r["epoch"]
        in_win = "*" if ep in si["candidate_epochs"] else ""
        elig = "*" if ep in si["eligible_epochs"] else ""
        mark = "  <-BEST" if ep == best else ""
        print(
            f"  {ep:>3}  {in_win:>3}  {elig:>4}  "
            f"{r['ssim_gap']:>+9.4f}  {r['refined_auc']:>11.4f}  "
            f"{r['distortion_score']:>10.5f}  {r['auc_gain']:>+9.4f}  "
            f"{r['ssim_oc']:>8.4f}  {r['acc']:>6.4f}  "
            f"{r['paper_score']:>10.4f}{mark}"
        )
    return S


anchor = json.loads(ANCHOR_C.read_text(encoding="utf-8")) if ANCHOR_C.exists() else None

summaries = [(tag, path, _tabulate(tag, path)) for tag, path in RUNS]

print("\n\n" + "=" * 100)
print("  CROSS-RUN COMPARISON vs. Baseline C anchor (paper strategy, relative, min_auc=0.95)")
print("=" * 100)
if anchor:
    print(f"  anchor:            ep{anchor['best_epoch']}  "
          f"ssim_gap={anchor['best_metrics']['ssim_gap']:+.4f}  "
          f"refined_auc={anchor['best_metrics']['refined_auc']:.4f}  "
          f"auc_gain={anchor['best_metrics']['auc_gain']:+.4f}  "
          f"ssim_oc={anchor['best_metrics']['ssim_oc']:.4f}")
for tag, path, S in summaries:
    bm = S["best_metrics"]
    print(f"  {tag[:35]:<35}:  ep{S['best_epoch']}  "
          f"ssim_gap={bm['ssim_gap']:+.4f}  "
          f"refined_auc={bm['refined_auc']:.4f}  "
          f"auc_gain={bm['auc_gain']:+.4f}  "
          f"ssim_oc={bm['ssim_oc']:.4f}")
print()

print("  Triplet images (columns = [original | noisy input | reconstructed by R]):")
for tag, path, _ in summaries:
    print(f"\n    -- {tag} --")
    for name in ("normal_triplets.png", "abnormal_triplets.png"):
        p = path / "triplets" / name
        print(f"       [{name:>22}] {'OK' if p.exists() else 'MISSING':>8}  {p}")
