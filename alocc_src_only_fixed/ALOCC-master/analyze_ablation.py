import json
import numpy as np
from pathlib import Path

base = Path("runs/ablation_cbam_s1")
TAGS = ["A_base", "B_s1", "C_cbam", "D_s1_cbam"]
DIGITS = [0, 2, 8]
SEEDS = [42, 2026]

def load(tag, digit, seed):
    p = base / f"{tag}_d{digit}_s{seed}" / "summary.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    bm = j["best_metrics"]
    return {
        "ep":       bm["epoch"],
        "auc":      round(float(bm["auc"]), 4),
        "raw_auc":  round(float(bm["raw_auc"]), 4),
        "ssim_ic":  round(float(bm["ssim_ic"]), 4),
        "ssim_oc":  round(float(bm["ssim_oc"]), 4),
        "ssim_gap": round(float(bm["ssim_gap"]), 4),
        "records":  j.get("records", []),
    }

# ── 1. 全量明细表 ────────────────────────────────────────────────────────────
print("=" * 100)
print("消融实验结果：CBAM vs S1 vs CBAM+S1")
print("=" * 100)
print(f"\n{'配置':<14} {'digit':>5} {'seed':>6} {'ep':>4} {'auc':>7} {'raw_auc':>8} {'ssim_ic':>8} {'ssim_oc':>8} {'ssim_gap':>9}")
print("-" * 80)
all_data = {}
for tag in TAGS:
    for digit in DIGITS:
        for seed in SEEDS:
            r = load(tag, digit, seed)
            all_data[(tag, digit, seed)] = r
            if r is None:
                print(f"{tag:<14} {digit:>5} {seed:>6}  -- 缺失")
            else:
                print(f"{tag:<14} {digit:>5} {seed:>6} {r['ep']:>4} {r['auc']:>7.4f} "
                      f"{r['raw_auc']:>8.4f} {r['ssim_ic']:>8.4f} {r['ssim_oc']:>8.4f} {r['ssim_gap']:>9.4f}")

# ── 2. 按 digit 聚合（2 seeds 均值）────────────────────────────────────────
print(f"\n{'配置':<14} {'digit':>5} {'auc':>10} {'ssim_oc':>10} {'ssim_gap':>10} {'ssim_ic':>10}")
print("-" * 60)
for digit in DIGITS:
    for tag in TAGS:
        rows = [all_data[(tag, digit, s)] for s in SEEDS if all_data.get((tag, digit, s))]
        if not rows:
            continue
        auc  = np.mean([r["auc"]      for r in rows])
        soc  = np.mean([r["ssim_oc"]  for r in rows])
        gap  = np.mean([r["ssim_gap"] for r in rows])
        sic  = np.mean([r["ssim_ic"]  for r in rows])
        print(f"{tag:<14} {digit:>5} {auc:>10.4f} {soc:>10.4f} {gap:>10.4f} {sic:>10.4f}")
    print()

# ── 3. 组合效果判定 ──────────────────────────────────────────────────────────
print("=" * 60)
print("组合效果判定（D vs B/C，按 digit）")
print("=" * 60)
for digit in DIGITS:
    print(f"\ndigit {digit}:")
    results = {}
    for tag in TAGS:
        rows = [all_data[(tag, digit, s)] for s in SEEDS if all_data.get((tag, digit, s))]
        if rows:
            results[tag] = {
                "auc":     np.mean([r["auc"]     for r in rows]),
                "ssim_oc": np.mean([r["ssim_oc"] for r in rows]),
                "ssim_ic": np.mean([r["ssim_ic"] for r in rows]),
            }
    if len(results) < 4:
        print("  数据不完整，跳过")
        continue

    A = results["A_base"]
    B = results["B_s1"]
    C = results["C_cbam"]
    D = results["D_s1_cbam"]

    print(f"  ssim_oc: A={A['ssim_oc']:.4f}  B={B['ssim_oc']:.4f}  C={C['ssim_oc']:.4f}  D={D['ssim_oc']:.4f}")
    print(f"  auc:     A={A['auc']:.4f}  B={B['auc']:.4f}  C={C['auc']:.4f}  D={D['auc']:.4f}")

    # 判定
    d_better_soc = D["ssim_oc"] < min(B["ssim_oc"], C["ssim_oc"])
    d_no_auc_collapse = D["auc"] >= A["auc"] - 0.05
    d_ic_ok = D["ssim_ic"] >= 0.15

    if d_better_soc and d_no_auc_collapse and d_ic_ok:
        verdict = "✅ 正向叠加：D 的 ssim_oc 优于 B/C，且 AUC 未塌"
    elif d_better_soc and not d_no_auc_collapse:
        verdict = "⚠️  ssim_oc 改善但 AUC 塌（过度约束），建议降低 rank 或 dropout"
    elif not d_better_soc and d_no_auc_collapse:
        verdict = "➡️  部分叠加：AUC 维持但 ssim_oc 未进一步改善"
    else:
        verdict = "❌ 无叠加效果或退化"
    print(f"  判定：{verdict}")

# ── 4. per-epoch 趋势对比（digit 8，seed 42）────────────────────────────────
print(f"\n{'='*60}")
print("per-epoch 趋势对比（digit=8, seed=42）")
print(f"{'='*60}")
for tag in TAGS:
    r = all_data.get((tag, 8, 42))
    if r is None or not r["records"]:
        continue
    recs = sorted(r["records"], key=lambda x: x["epoch"])
    print(f"\n  {tag}:")
    print(f"  {'ep':>4} {'auc':>7} {'raw':>7} {'ssim_ic':>8} {'ssim_oc':>8}")
    for rec in recs[::5]:  # 每5个epoch打印一次
        print(f"  {rec['epoch']:>4} {rec.get('auc',0):>7.4f} {rec.get('raw_auc',0):>7.4f} "
              f"{rec.get('ssim_ic',0):>8.4f} {rec.get('ssim_oc',0):>8.4f}")
