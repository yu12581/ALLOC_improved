"""Numerical audit of Baseline A v2026-04-27 verbatim 50-run.

Purpose
-------
Re-check whether AUC ~= 0.50 reflects a *broken* baseline, or a polarity-flipped
distribution whose folded mean recovers paper-grade performance.

For each of the 50 runs (5 seeds x 10 digits), we load summary.json from the
verbatim training directory and inspect:
  - last-epoch (ep10) raw_auc / auc      <- what D4-C reports as 'best'
  - per-epoch raw_auc trajectory         <- to see if the model ever worked
  - polarity (raw_auc < 0.5)             <- which seeds learned 'inverted' D
  - folded raw_auc = max(rauc, 1-rauc)   <- true discriminative power

We also look at paper-window-style selection: for each run, take the BEST
raw_auc across epochs 2..6 (paper ALOCC stop range). This emulates what the
paper actually selects, vs forced last-epoch.

No writes - print only.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:/Trae_coding/ALLOC/ALOCC-master/baseline_a_v2026_04_27")
SEEDS = [42, 43, 44, 45, 46]
DIGITS = list(range(10))


def load_run(d: int, s: int) -> dict | None:
    fp = ROOT / f"d{d}_s{s}" / "summary.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []
    missing = []
    for d in DIGITS:
        for s in SEEDS:
            rec = load_run(d, s)
            if rec is None:
                missing.append((d, s))
                continue
            best = rec.get("best_metrics", {})
            records = rec.get("records", [])
            recs_by_ep = {r["epoch"]: r for r in records}
            ep10 = recs_by_ep.get(10, {})
            # paper-window: best raw_auc in epochs 2..6
            window = [recs_by_ep[e] for e in (2, 3, 4, 5, 6) if e in recs_by_ep]
            if window:
                best_pw = max(window, key=lambda r: r.get("raw_auc", 0.0))
            else:
                best_pw = {}
            rows.append({
                "digit": d, "seed": s,
                "best_epoch": rec.get("best_epoch"),
                "ep10_raw_auc": ep10.get("raw_auc", float("nan")),
                "ep10_auc": ep10.get("auc", float("nan")),
                "ep10_ssim_ic": ep10.get("ssim_ic", float("nan")),
                "ep10_ssim_oc": ep10.get("ssim_oc", float("nan")),
                "best_raw_auc_any": max(
                    (r.get("raw_auc", 0.0) for r in records), default=float("nan")),
                "pw_best_ep": best_pw.get("epoch"),
                "pw_best_raw_auc": best_pw.get("raw_auc", float("nan")),
                "pw_best_auc": best_pw.get("auc", float("nan")),
                "pw_best_ssim_ic": best_pw.get("ssim_ic", float("nan")),
                "pw_best_ssim_oc": best_pw.get("ssim_oc", float("nan")),
            })

    if missing:
        print(f"!! missing {len(missing)} run(s): {missing[:5]} ...")

    n = len(rows)
    print(f"\n=== loaded {n} runs from {ROOT} ===")

    # --- 1) ep10 raw_auc histogram (polarity diagnosis) ---
    print("\n--- 1) ep10 raw_auc distribution (last_epoch = D4-C verbatim) ---")
    rauc10 = [r["ep10_raw_auc"] for r in rows]
    auc10 = [r["ep10_auc"] for r in rows]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    counts = [0] * (len(bins) - 1)
    for v in rauc10:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    for i, c in enumerate(counts):
        bar = "#" * c
        print(f"  raw_auc [{bins[i]:.1f}, {bins[i+1]:.1f}) : {c:2d}  {bar}")

    folded = [max(v, 1 - v) for v in rauc10]
    inverted = sum(1 for v in rauc10 if v < 0.5)
    print(f"\n  ep10 raw_auc:    mean={st.mean(rauc10):.4f}  median={st.median(rauc10):.4f}  std={st.pstdev(rauc10):.4f}")
    print(f"  ep10 raw_auc INVERTED runs (rauc<0.5): {inverted}/{n}")
    print(f"  ep10 |raw_auc-0.5|>=0.10:  {sum(1 for v in rauc10 if abs(v-0.5)>=0.10)}/{n}  (decisive runs)")
    print(f"  ep10 |raw_auc-0.5|>=0.25:  {sum(1 for v in rauc10 if abs(v-0.5)>=0.25)}/{n}  (high-confidence)")
    print(f"  ep10 FOLDED raw_auc: mean={st.mean(folded):.4f}  median={st.median(folded):.4f}  std={st.pstdev(folded):.4f}")

    # --- 2) paper-window best raw_auc in ep2..6 ---
    print("\n--- 2) paper-window selection (best raw_auc in epochs 2..6) ---")
    pw = [r["pw_best_raw_auc"] for r in rows]
    pw_folded = [max(v, 1 - v) for v in pw]
    print(f"  paper-window best raw_auc: mean={st.mean(pw):.4f}  median={st.median(pw):.4f}  std={st.pstdev(pw):.4f}")
    print(f"  paper-window FOLDED raw_auc: mean={st.mean(pw_folded):.4f}  median={st.median(pw_folded):.4f}")
    print(f"  paper-window inverted (raw_auc<0.5): {sum(1 for v in pw if v<0.5)}/{n}")
    print(f"  paper-window >=0.90: {sum(1 for v in pw if v>=0.90)}/{n}")

    # --- 3) per-class breakdown ---
    print("\n--- 3) per-class folded raw_auc (5 seeds each) ---")
    print(f"  {'cls':>3} {'ep10_rauc_mean':>14} {'ep10_FOLDED':>11} {'pw_rauc_mean':>13} {'pw_FOLDED':>10}  ep10_polarities")
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["digit"]].append(r)
    for d in sorted(by_d):
        sub = by_d[d]
        rauc = [r["ep10_raw_auc"] for r in sub]
        f10 = [max(v, 1 - v) for v in rauc]
        pwsub = [r["pw_best_raw_auc"] for r in sub]
        fpw = [max(v, 1 - v) for v in pwsub]
        polarities = "".join("+" if v >= 0.5 else "-" for v in rauc)
        print(f"  {d:>3} {st.mean(rauc):>14.4f} {st.mean(f10):>11.4f} {st.mean(pwsub):>13.4f} {st.mean(fpw):>10.4f}  {polarities}")

    # --- 4) sanity: what is the BEST raw_auc anywhere (any epoch) ---
    print("\n--- 4) BEST raw_auc across ALL 10 epochs of each run ---")
    bany = [r["best_raw_auc_any"] for r in rows]
    print(f"  best_any: mean={st.mean(bany):.4f}  median={st.median(bany):.4f}  min={min(bany):.4f}  max={max(bany):.4f}")
    print(f"  best_any >=0.90: {sum(1 for v in bany if v>=0.90)}/{n}")
    print(f"  best_any >=0.95: {sum(1 for v in bany if v>=0.95)}/{n}")


if __name__ == "__main__":
    main()
