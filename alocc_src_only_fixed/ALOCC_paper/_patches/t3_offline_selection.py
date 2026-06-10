"""T3 offline selection verification.

Replay alternative epoch-selection rules on recorded `records[]` from
9 summary.json files (3 classes x 3 configs). No training, no GPU.

Strategies:
  D0 = distortion  (= ssim_gap * refined_auc)  [current default]
  S1 = acc_auc     (= (acc + auc) / 2)         [ALOCC pre-X3 default]
  S2 = raw_auc_max (raw_auc max)               [ignore refinement confound]
  R1 = earliest epoch meeting (ssim_oc <= 0.15 AND raw_auc >= 0.60)
  R2 = earliest epoch meeting (ssim_oc <= 0.15 AND auc_gain >= 0)
  R3 = earliest epoch meeting (ssim_oc <= 0.15 AND raw_auc >= 0.60 AND raw_score_gap > 0)
  PK = ideal-oracle: max raw_auc among epochs with ssim_oc <= 0.15 (upper bound)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALOCC_paper")
CLASSES = [2, 6, 1]
CONFIGS = [
    ("OFF",   lambda c: ROOT / f"s1_c{c}_off"        / "experiment" / "summary.json"),
    ("S1",    lambda c: ROOT / f"s1_c{c}_r16_p03"    / "experiment" / "summary.json"),
    ("T2-SN", lambda c: ROOT / f"t2_c{c}_sn_r16_p03" / "experiment" / "summary.json"),
]
SSIM_REDLINE = 0.15


def _load(path: Path):
    if not path.exists():
        return None
    j = json.loads(path.read_text(encoding="utf-8"))
    recs = [r for r in j.get("records", []) if isinstance(r, dict)]
    return j, recs


def pick_distortion(recs):
    return max(recs, key=lambda r: r.get("distortion_score", -1))


def pick_acc_auc(recs):
    return max(recs, key=lambda r: (r.get("acc", 0) + r.get("auc", 0)) / 2)


def pick_raw_auc_max(recs):
    return max(recs, key=lambda r: r.get("raw_auc", 0))


def pick_redline(recs, *, require_auc_gain_pos=False, require_raw_sg_pos=False,
                 raw_auc_min=0.60):
    eligible = []
    for r in recs:
        if r.get("ssim_oc", 1) > SSIM_REDLINE:
            continue
        if r.get("raw_auc", 0) < raw_auc_min:
            continue
        if require_auc_gain_pos and r.get("auc_gain", -1) < 0:
            continue
        if require_raw_sg_pos and r.get("raw_score_gap", -1) <= 0:
            continue
        eligible.append(r)
    if not eligible:
        return None
    return min(eligible, key=lambda r: r.get("epoch", 999))


def pick_oracle(recs):
    eligible = [r for r in recs if r.get("ssim_oc", 1) <= SSIM_REDLINE]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.get("raw_auc", 0))


def summarise(r):
    if r is None:
        return "          |       |       |      |       |       "
    return (f" ep{r.get('epoch', 0):>2} | "
            f"{r.get('auc', 0):>.4f} | "
            f"{r.get('raw_auc', 0):>.4f} | "
            f"{r.get('ssim_oc', 0):>.3f} | "
            f"{r.get('auc_gain', 0):>+.3f} | "
            f"{r.get('raw_score_gap', 0):>+.4f}")


def main():
    strategies = [
        ("D0 distortion",      pick_distortion),
        ("S1 acc_auc",         pick_acc_auc),
        ("S2 raw_auc_max",     pick_raw_auc_max),
        ("R1 oc<=.15 & raw>=.6", lambda r: pick_redline(r)),
        ("R2 oc & gain>=0",    lambda r: pick_redline(r, require_auc_gain_pos=True)),
        ("R3 oc & raw>=.6 & rsg>0", lambda r: pick_redline(r, require_raw_sg_pos=True)),
        ("PK oracle",          pick_oracle),
    ]

    hdr = f"{'strategy':<24} | {'epoch':>4} | {'auc':>6} | {'raw_auc':>7} | {'ssim_oc':>7} | {'a_gain':>7} | {'raw_sg':>7}"
    all_results = {}

    for c in CLASSES:
        print(f"\n========== class {c} ==========")
        all_results[c] = {}
        for cname, p in CONFIGS:
            loaded = _load(p(c))
            if loaded is None:
                print(f"\n  [{cname}] MISSING")
                continue
            j, recs = loaded
            print(f"\n  [{cname}] (current best via distortion = ep{j.get('best_epoch')})")
            print(f"  {hdr}")
            print("  " + "-" * 80)
            all_results[c][cname] = {}
            for sname, fn in strategies:
                try:
                    r = fn(recs)
                except Exception as e:
                    print(f"  {sname:<24} | ERROR: {e}")
                    continue
                print(f"  {sname:<24} |{summarise(r)}")
                if r is not None:
                    all_results[c][cname][sname] = {
                        "epoch": r.get("epoch"),
                        "auc": r.get("auc"),
                        "raw_auc": r.get("raw_auc"),
                        "ssim_oc": r.get("ssim_oc"),
                        "auc_gain": r.get("auc_gain"),
                        "raw_score_gap": r.get("raw_score_gap"),
                    }
                else:
                    all_results[c][cname][sname] = None

    out = ROOT / "t3_offline_selection.json"
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
