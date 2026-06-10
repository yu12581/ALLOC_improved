"""[A1-SEL] ADR-007 regression: post-patch, the 3 legacy strategies must
pick the same epoch as the original summary.json against the same records.

Also exercise the new `redline` branch on the 9 existing record sets to
confirm it matches the offline t3 verification.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Trae_coding\ALLOC\ALOCC-master")
from mnist_experiment_runner import _select_records  # noqa: E402

ROOT = Path(r"D:\Trae_coding\ALOCC_paper")
SUMMARIES = [
    ("OFF",   "s1_c{c}_off"),
    ("S1",    "s1_c{c}_r16_p03"),
    ("T2-SN", "t2_c{c}_sn_r16_p03"),
]
CLASSES = [2, 6, 1]


def _run(records, strategy, ctx=None, **extra):
    """Mirror the kwargs evaluate_checkpoints supplies; ctx carries original
    epoch window + thresholds so legacy runs replay deterministically."""
    ctx = ctx or {}
    return _select_records(
        records=records,
        selection_strategy=strategy,
        selection_epoch_start=ctx.get("epoch_start"),
        selection_epoch_end=ctx.get("epoch_end"),
        selection_min_acc=ctx.get("min_acc"),
        selection_min_auc=ctx.get("min_auc"),
        selection_min_auc_hard=bool(ctx.get("min_auc_hard", False)),
        selection_log_fallback=False,
        distortion_alpha=float(ctx.get("distortion_alpha", 1.0)),
        distortion_beta=float(ctx.get("distortion_beta", 1.0)),
        **extra,
    )


def main():
    failures = 0
    total = 0
    print(f"{'cfg':<6} {'cls':>3} {'strategy':<10} {'orig_ep':>7} {'now_ep':>7}  result")
    print("-" * 60)
    for tag, tpl in SUMMARIES:
        for c in CLASSES:
            p = ROOT / tpl.format(c=c) / "experiment" / "summary.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text(encoding="utf-8"))
            records = j["records"]
            orig_info = j["selection_info"]
            orig_strategy = orig_info["strategy"]
            orig_best = j["best_epoch"]
            ctx = orig_info  # epoch_start/end/min_acc/min_auc live here
            for strategy in ("acc_auc", "paper", "distortion"):
                total += 1
                sel, info = _run(records, strategy,
                                 ctx=ctx if strategy == orig_strategy else None)
                got = sel["epoch"]
                if strategy == orig_strategy:
                    ok = (got == orig_best)
                    tag_r = "PASS" if ok else "FAIL"
                    if not ok:
                        failures += 1
                    print(f"{tag:<6} {c:>3} {strategy:<10} {orig_best:>7} {got:>7}  {tag_r} (orig)")
                else:
                    print(f"{tag:<6} {c:>3} {strategy:<10} {'-':>7} {got:>7}  info")
    print()
    print(f"legacy-parity failures: {failures}")
    print()
    # redline cross-check on same 9 record sets
    print(f"{'cfg':<6} {'cls':>3} {'redline':<12} {'epoch':>5} {'raw_auc':>8} {'ssim_oc':>8}  fallback")
    print("-" * 70)
    rl_expected = {
        ("OFF", 2): None, ("OFF", 6): 1, ("OFF", 1): 1,
        ("S1", 2): 1,    ("S1", 6): 1,   ("S1", 1): 1,
        ("T2-SN", 2): 1, ("T2-SN", 6): None, ("T2-SN", 1): None,
    }
    rl_mismatch = 0
    for tag, tpl in SUMMARIES:
        for c in CLASSES:
            p = ROOT / tpl.format(c=c) / "experiment" / "summary.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text(encoding="utf-8"))
            sel, info = _run(j["records"], "redline",
                             redline_ssim_oc_max=0.15, redline_raw_auc_min=0.60)
            fb = info.get("redline_fallback_triggered", False)
            exp = rl_expected.get((tag, c))
            if exp is None:
                # expected fallback; any epoch OK but fallback must be True
                status = "OK(fallback)" if fb else "UNEXPECTED-NONFB"
                if not fb:
                    rl_mismatch += 1
            else:
                ok = (sel["epoch"] == exp and not fb)
                status = "PASS" if ok else f"FAIL (exp ep{exp})"
                if not ok:
                    rl_mismatch += 1
            print(f"{tag:<6} {c:>3} {'redline':<12} {sel['epoch']:>5} "
                  f"{sel.get('raw_auc', 0):>8.4f} {sel.get('ssim_oc', 0):>8.3f}  "
                  f"{fb!s:<5} {status}")
    print()
    print(f"redline mismatches: {rl_mismatch}")
    if failures == 0 and rl_mismatch == 0:
        print("\n[A1-SEL] ADR-007 OK + redline agrees with offline t3")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
