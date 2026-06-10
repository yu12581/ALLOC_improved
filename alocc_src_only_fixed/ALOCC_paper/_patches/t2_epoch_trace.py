"""Epoch-by-epoch trace: for each (class, variant), print auc / raw_auc / auc_gain / ssim_ic / ssim_oc / score_gap.

Aim: reveal whether the collapse is (a) training dynamics, (b) selection strategy, or (c) both.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALOCC_paper")
CLASSES = [2, 6, 1]
VARIANTS = [
    ("OFF",   lambda c: ROOT / f"s1_c{c}_off"       / "experiment" / "summary.json"),
    ("S1",    lambda c: ROOT / f"s1_c{c}_r16_p03"   / "experiment" / "summary.json"),
    ("T2-SN", lambda c: ROOT / f"t2_c{c}_sn_r16_p03"/ "experiment" / "summary.json"),
]

def rec_get(r, k):
    # records might be dicts or PowerShell-stringified dicts; normalize
    if isinstance(r, dict):
        return r.get(k)
    return None


for c in CLASSES:
    print(f"\n========== class {c} ==========")
    for vname, p in VARIANTS:
        path = p(c)
        if not path.exists():
            print(f"[{vname}] missing: {path.name}")
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        best = j.get("best_epoch")
        print(f"\n  [{vname}]  best_epoch={best}  selection={j.get('switches', {}).get('selection_strategy')}")
        hdr = f"    {'ep':>3} | {'auc':>7} | {'raw_auc':>7} | {'auc_gain':>9} | {'ssim_ic':>7} | {'ssim_oc':>7} | {'ssim_gap':>8} | {'score_gap':>9} | {'raw_sg':>7} | {'dist':>7}"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for r in j.get("records", []):
            if not isinstance(r, dict):
                continue
            ep = r.get("epoch")
            mark = " *" if ep == best else "  "
            print(f"    {ep:>3}{mark}| "
                  f"{r.get('auc', 0):>7.4f} | "
                  f"{r.get('raw_auc', 0):>7.4f} | "
                  f"{r.get('auc_gain', 0):>+9.4f} | "
                  f"{r.get('ssim_ic', 0):>7.4f} | "
                  f"{r.get('ssim_oc', 0):>7.4f} | "
                  f"{r.get('ssim_gap', 0):>8.4f} | "
                  f"{r.get('score_gap', 0):>+9.5f} | "
                  f"{r.get('raw_score_gap', 0):>+7.4f} | "
                  f"{r.get('distortion_score', 0):>7.4f}")
