import json
import numpy as np
from pathlib import Path

base = Path("runs/full_40ep_bestauc")
results = []
for d in sorted(base.iterdir()):
    s = d / "summary.json"
    if not s.exists():
        continue
    j = json.loads(s.read_text(encoding="utf-8"))
    bm = j["best_metrics"]
    rec = j.get("records", [])
    results.append({
        "name": d.name,
        "ep": bm["epoch"],
        "auc": round(float(bm["auc"]), 4),
        "raw_auc": round(float(bm["raw_auc"]), 4),
        "ssim_ic": round(float(bm["ssim_ic"]), 4),
        "ssim_oc": round(float(bm["ssim_oc"]), 4),
        "ssim_gap": round(float(bm["ssim_gap"]), 4),
        "score_gap": round(float(bm["score_gap"]), 4),
        "auc_gain": round(float(bm["auc_gain"]), 4),
        "records": rec,
    })

print(f"鍏?{len(results)} 缁勭粨鏋淺n")

# 鍏ㄩ噺鏄庣粏琛? print(f"{'name':<14} {'ep':>4} {'auc':>7} {'raw_auc':>8} {'ssim_ic':>8} {'ssim_oc':>8} {'ssim_gap':>9} {'score_gap':>10} {'auc_gain':>9}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<14} {r['ep']:>4} {r['auc']:>7.4f} {r['raw_auc']:>8.4f} "
          f"{r['ssim_ic']:>8.4f} {r['ssim_oc']:>8.4f} {r['ssim_gap']:>9.4f} "
          f"{r['score_gap']:>10.4f} {r['auc_gain']:>9.4f}")

# 鎸?digit 鑱氬悎
print("\n--- 鎸?digit 鑱氬悎锛? seeds 鍧囧€?-std锛?--")
print(f"{'digit':<6} {'auc':>12} {'raw_auc':>12} {'ssim_oc':>12} {'ssim_gap':>12} {'auc_gain':>12}")
print("-" * 68)
all_aucs = []
for digit in range(10):
    rows = [r for r in results if r["name"].startswith(f"d{digit}_")]
    if not rows:
        continue
    def ms(key):
        v = [r[key] for r in rows]
        return f"{np.mean(v):.4f}+-{np.std(v):.4f}"
    all_aucs.extend([r["auc"] for r in rows])
    print(f"d{digit}     {ms('auc'):>12} {ms('raw_auc'):>12} {ms('ssim_oc'):>12} {ms('ssim_gap'):>12} {ms('auc_gain'):>12}")
print(f"\n鍏ㄧ被骞冲潎 AUC: {np.mean(all_aucs):.4f}")

# ssim_oc 鍒嗗竷
print("\n--- ssim_oc 鍒嗗竷锛坕dentity shortcut 璇婃柇锛?--")
soc_vals = [r["ssim_oc"] for r in results]
print(f"  mean={np.mean(soc_vals):.4f}  std={np.std(soc_vals):.4f}  min={np.min(soc_vals):.4f}  max={np.max(soc_vals):.4f}")
print(f"  <=0.15 (绾㈢嚎閫氳繃): {sum(v<=0.15 for v in soc_vals)}/{len(soc_vals)}")
print(f"  <=0.30           : {sum(v<=0.30 for v in soc_vals)}/{len(soc_vals)}")
print(f"  > 0.80 (涓ラ噸shortcut): {sum(v>0.80 for v in soc_vals)}/{len(soc_vals)}")

# best_epoch 鍒嗗竷
print("\n--- best_epoch 鍒嗗竷 ---")
eps = [r["ep"] for r in results]
print(f"  mean={np.mean(eps):.1f}  median={np.median(eps):.0f}  min={np.min(eps)}  max={np.max(eps)}")
print(f"  鏃╁仠(ep<=5): {sum(e<=5 for e in eps)}/{len(eps)}   鏅氭敹鏁?ep>=30): {sum(e>=30 for e in eps)}/{len(eps)}")

# per-epoch 瓒嬪娍锛坉2/d8 seed=42锛? print("\n--- per-epoch 瓒嬪娍锛坉2_s42 / d8_s42锛?--")
for name in ["d2_s42", "d8_s42"]:
    row = next((r for r in results if r["name"] == name), None)
    if row is None or not row["records"]:
        print(f"  {name}: 鏃?records")
        continue
    recs = sorted(row["records"], key=lambda x: x["epoch"])
    print(f"  {name}:")
    for rec in recs:
        ep   = rec["epoch"]
        auc  = rec.get("auc", float("nan"))
        rauc = rec.get("raw_auc", float("nan"))
        soc  = rec.get("ssim_oc", float("nan"))
        sic  = rec.get("ssim_ic", float("nan"))
        print(f"    ep{ep:>2}: auc={auc:.4f} raw={rauc:.4f} ssim_ic={sic:.4f} ssim_oc={soc:.4f}")
