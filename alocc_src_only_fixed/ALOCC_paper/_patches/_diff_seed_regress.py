"""[A4-SEED] Field-by-field diff between regression run and stored baseline.

Bitwise identity means every field in `best_metrics` must match exactly.
The `switches` dict is allowed to differ by the `seed` key (added by the patch).
"""
from __future__ import annotations
import json, pathlib, math

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")
BASELINE = ROOT / "s1_c1_r16_p03_redline" / "experiment" / "summary.json"
REGRESS = ROOT / "s1_c1_r16_p03_redline_regress" / "experiment" / "summary.json"

b = json.loads(BASELINE.read_text(encoding="utf-8"))
r = json.loads(REGRESS.read_text(encoding="utf-8"))


def cmp(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        diffs = []
        for k in sorted(keys):
            sub = cmp(a.get(k, "<MISSING>"), b.get(k, "<MISSING>"), f"{path}.{k}")
            diffs.extend(sub)
        return diffs
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return []
        if a != b:
            return [(path, a, b, abs(a - b))]
        return []
    if a != b:
        return [(path, a, b, None)]
    return []


print(f"baseline: {BASELINE}")
print(f"regress : {REGRESS}")
print(f"baseline best_epoch={b.get('best_epoch')}  regress best_epoch={r.get('best_epoch')}")
print()

bm_diffs = cmp(b.get("best_metrics", {}), r.get("best_metrics", {}), "best_metrics")
if bm_diffs:
    print(f"[BEST_METRICS] {len(bm_diffs)} differences:")
    for path, va, vb, delta in bm_diffs:
        d_str = f"  |Δ|={delta:.2e}" if delta is not None else ""
        print(f"  {path}: baseline={va!r}  regress={vb!r}{d_str}")
else:
    print("[BEST_METRICS] BITWISE IDENTICAL (0 differences)")

print()
sw_b = b.get("switches", {})
sw_r = r.get("switches", {})
sw_keys_only_in_regress = set(sw_r) - set(sw_b)
sw_keys_only_in_baseline = set(sw_b) - set(sw_r)
sw_common_diffs = cmp({k: sw_b[k] for k in sw_b if k in sw_r},
                      {k: sw_r[k] for k in sw_r if k in sw_b}, "switches")
print(f"[SWITCHES] added by patch: {sorted(sw_keys_only_in_regress)}")
print(f"[SWITCHES] removed by patch: {sorted(sw_keys_only_in_baseline)}")
if sw_common_diffs:
    print(f"[SWITCHES] common-key diffs: {len(sw_common_diffs)}")
    for path, va, vb, delta in sw_common_diffs:
        print(f"  {path}: baseline={va!r}  regress={vb!r}")
else:
    print("[SWITCHES] common keys identical")

print()
if not bm_diffs and not sw_common_diffs:
    added = sorted(sw_keys_only_in_regress)
    if added == ["seed"] and sw_r.get("seed") is None:
        print("VERDICT: PASS (bitwise identical; only 'seed=None' added to switches)")
    elif not added and not sw_keys_only_in_baseline:
        print("VERDICT: PASS (fully identical switches + metrics)")
    else:
        print("VERDICT: PASS with switches diff (review above)")
else:
    print("VERDICT: FAIL (non-trivial differences; investigate)")
