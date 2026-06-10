"""[A1-SEL] Add `redline` selection strategy to runner + fig6_7.

Idempotent patch. Writes `.redline.bak` backup on first apply.
Changes:
  1. mnist_experiment_runner.py
     - _select_records: add redline_ssim_oc_max / redline_raw_auc_min params
       + redline branch + selection_info audit fields.
     - evaluate_checkpoints: pass-through.
     - run_experiment: pass-through from args + switches echo.
     - parse_args: extend --selection-strategy choices + 2 new flags.
  2. run_paper_mnist_figure6_7.py
     - Args dataclass: 2 new fields.
     - argparse: extend choices + 2 new flags.
     - evaluate_checkpoints() call: pass-through.
     - summary dict: 2 new keys.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
RUNNER = ROOT / "mnist_experiment_runner.py"
FIG = ROOT / "run_paper_mnist_figure6_7.py"
SUFFIX = ".redline.bak"

SENTINEL = "[A1-SEL]"


def _backup(p: Path):
    bak = p.with_suffix(p.suffix + SUFFIX)
    if not bak.exists():
        bak.write_bytes(p.read_bytes())
        print(f"backup: {bak.name}")


def _patch_text(src: str, old: str, new: str, label: str) -> str:
    if new in src:
        print(f"  [{label}] already applied; skip")
        return src
    if old not in src:
        raise RuntimeError(f"[{label}] anchor not found: {old[:80]!r}")
    print(f"  [{label}] patched")
    return src.replace(old, new, 1)


def patch_runner():
    src = RUNNER.read_text(encoding="utf-8")

    # 1) _select_records signature: add two new kwargs.
    old_sig = (
        "def _select_records(records, selection_strategy: str, selection_epoch_start, "
        "selection_epoch_end, selection_min_acc, selection_min_auc, "
        "selection_min_auc_hard: bool = False, selection_log_fallback: bool = True, "
        "distortion_alpha: float = 1.0, distortion_beta: float = 1.0):"
    )
    new_sig = (
        "def _select_records(records, selection_strategy: str, selection_epoch_start, "
        "selection_epoch_end, selection_min_acc, selection_min_auc, "
        "selection_min_auc_hard: bool = False, selection_log_fallback: bool = True, "
        "distortion_alpha: float = 1.0, distortion_beta: float = 1.0, "
        "redline_ssim_oc_max: float = 0.15, redline_raw_auc_min: float = 0.60):  # [A1-SEL]"
    )
    src = _patch_text(src, old_sig, new_sig, "runner:_select_records/sig")

    # 2) Inject redline branch + audit fields in selection_info.
    old_branch = (
        '    elif selection_strategy == "distortion":'
    )
    # Keep legacy `distortion` body; insert a new elif BEFORE else.
    # Use the `else:` anchor to insert `elif redline:` right above it.
    old_else = (
        '    else:\n'
        '        raise ValueError(f"Unknown selection strategy: {selection_strategy}")'
    )
    new_else = (
        '    elif selection_strategy == "redline":  # [A1-SEL]\n'
        '        rl_eligible = [\n'
        '            r for r in eligible\n'
        '            if r.get("ssim_oc", 1.0) <= redline_ssim_oc_max\n'
        '            and r.get("raw_auc", 0.0) >= redline_raw_auc_min\n'
        '        ]\n'
        '        redline_fallback_triggered = False\n'
        '        redline_fallback_reason = None\n'
        '        if rl_eligible:\n'
        '            best_record = min(\n'
        '                rl_eligible,\n'
        '                key=lambda r: (\n'
        '                    r["epoch"],\n'
        '                    -r.get("raw_auc", 0.0),\n'
        '                    -r.get("auc", 0.0),\n'
        '                    -r.get("score_gap", 0.0),\n'
        '                    r.get("ssim_oc", 1.0),\n'
        '                    r.get("eer", 1.0),\n'
        '                ),\n'
        '            )\n'
        '        else:\n'
        '            redline_fallback_triggered = True\n'
        '            redline_fallback_reason = (\n'
        '                f"No epoch in {[r[\'epoch\'] for r in eligible]} satisfied "\n'
        '                f"ssim_oc<={redline_ssim_oc_max} AND raw_auc>={redline_raw_auc_min}"\n'
        '            )\n'
        '            if selection_min_auc_hard:\n'
        '                raise RuntimeError(f"[A1-SEL] redline has no eligible epoch; hard-fail on. {redline_fallback_reason}")\n'
        '            if selection_log_fallback:\n'
        '                print(f"[A1-SEL][selection] WARNING redline fallback to distortion key: {redline_fallback_reason}", file=sys.stderr)\n'
        '            best_record = max(\n'
        '                eligible,\n'
        '                key=lambda record: (\n'
        '                    record["distortion_score"],\n'
        '                    record["auc_gain"],\n'
        '                    record["ssim_gap"],\n'
        '                    -record["ssim_oc"],\n'
        '                    record["paper_score"],\n'
        '                    record["acc"],\n'
        '                    record["auc"],\n'
        '                    -record["eer"],\n'
        '                ),\n'
        '            )\n'
        '    else:\n'
        '        raise ValueError(f"Unknown selection strategy: {selection_strategy}")'
    )
    src = _patch_text(src, old_else, new_else, "runner:_select_records/redline-branch")

    # 3) selection_info: add redline audit fields.
    old_info_tail = (
        '        "distortion_alpha": float(distortion_alpha),\n'
        '        "distortion_beta": float(distortion_beta),\n'
        '    }\n'
        '    return best_record, selection_info'
    )
    new_info_tail = (
        '        "distortion_alpha": float(distortion_alpha),\n'
        '        "distortion_beta": float(distortion_beta),\n'
        '        "redline_ssim_oc_max": float(redline_ssim_oc_max),  # [A1-SEL]\n'
        '        "redline_raw_auc_min": float(redline_raw_auc_min),  # [A1-SEL]\n'
        '        "redline_fallback_triggered": bool(locals().get("redline_fallback_triggered", False)),  # [A1-SEL]\n'
        '        "redline_fallback_reason": locals().get("redline_fallback_reason", None),  # [A1-SEL]\n'
        '    }\n'
        '    return best_record, selection_info'
    )
    src = _patch_text(src, old_info_tail, new_info_tail, "runner:_select_records/info-tail")

    RUNNER.write_text(src, encoding="utf-8")



def patch_runner_part2():
    """Pass-through in evaluate_checkpoints + run_experiment + parse_args."""
    src = RUNNER.read_text(encoding="utf-8")

    # 4) evaluate_checkpoints signature.
    old_ec_sig = (
        "    distortion_alpha: float = 1.0,\n"
        "    distortion_beta: float = 1.0,\n"
        '    paper_score_normalization: str = "relative",\n'
        "):"
    )
    new_ec_sig = (
        "    distortion_alpha: float = 1.0,\n"
        "    distortion_beta: float = 1.0,\n"
        '    paper_score_normalization: str = "relative",\n'
        "    redline_ssim_oc_max: float = 0.15,  # [A1-SEL]\n"
        "    redline_raw_auc_min: float = 0.60,  # [A1-SEL]\n"
        "):"
    )
    src = _patch_text(src, old_ec_sig, new_ec_sig, "runner:evaluate_checkpoints/sig")

    # 5) evaluate_checkpoints body: forward two new kwargs to _select_records.
    old_ec_call = (
        "        distortion_alpha=distortion_alpha,\n"
        "        distortion_beta=distortion_beta,\n"
        "    )\n"
        "    best_epoch = best_metrics[\"epoch\"]"
    )
    new_ec_call = (
        "        distortion_alpha=distortion_alpha,\n"
        "        distortion_beta=distortion_beta,\n"
        "        redline_ssim_oc_max=redline_ssim_oc_max,  # [A1-SEL]\n"
        "        redline_raw_auc_min=redline_raw_auc_min,  # [A1-SEL]\n"
        "    )\n"
        "    best_epoch = best_metrics[\"epoch\"]"
    )
    src = _patch_text(src, old_ec_call, new_ec_call, "runner:evaluate_checkpoints/call")

    # 6) run_experiment -> evaluate_checkpoints call: forward from args.
    old_re_call = (
        '        paper_score_normalization=getattr(args, "paper_score_normalization", "relative"),\n'
        "    )\n"
        "\n"
        "    summary = {"
    )
    new_re_call = (
        '        paper_score_normalization=getattr(args, "paper_score_normalization", "relative"),\n'
        '        redline_ssim_oc_max=float(getattr(args, "redline_ssim_oc_max", 0.15)),  # [A1-SEL]\n'
        '        redline_raw_auc_min=float(getattr(args, "redline_raw_auc_min", 0.60)),  # [A1-SEL]\n'
        "    )\n"
        "\n"
        "    summary = {"
    )
    src = _patch_text(src, old_re_call, new_re_call, "runner:run_experiment/call")

    # 7) Summary switches echo. Anchor on the final S1-BOT switch (stable
    # whether or not T2-SN has been applied/rolled-back).
    old_sw = (
        '            "bottleneck_noise_type": str(getattr(args, "bottleneck_noise_type", "dropout")),\n'
        "        },\n"
        '        "selection_info": selection_info,'
    )
    new_sw = (
        '            "bottleneck_noise_type": str(getattr(args, "bottleneck_noise_type", "dropout")),\n'
        '            "redline_ssim_oc_max": float(getattr(args, "redline_ssim_oc_max", 0.15)),  # [A1-SEL]\n'
        '            "redline_raw_auc_min": float(getattr(args, "redline_raw_auc_min", 0.60)),  # [A1-SEL]\n'
        "        },\n"
        '        "selection_info": selection_info,'
    )
    src = _patch_text(src, old_sw, new_sw, "runner:summary/switches")

    # 8) argparse: extend --selection-strategy choices + 2 new flags.
    old_cli = (
        '    parser.add_argument("--selection-strategy", choices=["acc_auc", "paper", "distortion"], default="acc_auc")'
    )
    new_cli = (
        '    parser.add_argument("--selection-strategy", choices=["acc_auc", "paper", "distortion", "redline"], default="acc_auc")  # [A1-SEL]\n'
        '    parser.add_argument("--redline-ssim-oc-max", type=float, default=0.15,\n'
        '                        help="[A1-SEL] ssim_oc upper bound for redline eligibility (default 0.15).")\n'
        '    parser.add_argument("--redline-raw-auc-min", type=float, default=0.60,\n'
        '                        help="[A1-SEL] raw_auc lower bound for redline eligibility (default 0.60).")'
    )
    src = _patch_text(src, old_cli, new_cli, "runner:argparse/redline")

    RUNNER.write_text(src, encoding="utf-8")


def patch_fig():
    src = FIG.read_text(encoding="utf-8")

    # Args dataclass: anchor on the last [S1-BOT] field (stable
    # whether or not T2-SN has been applied/rolled-back).
    old_dc = (
        '    bottleneck_noise_type: str = "dropout"'
    )
    new_dc = (
        '    bottleneck_noise_type: str = "dropout"\n'
        "    redline_ssim_oc_max: float = 0.15  # [A1-SEL]\n"
        "    redline_raw_auc_min: float = 0.60  # [A1-SEL]"
    )
    src = _patch_text(src, old_dc, new_dc, "fig:Args")

    old_cli = (
        '    parser.add_argument("--selection-strategy", choices=["acc_auc", "paper", "distortion"], default="paper")'
    )
    new_cli = (
        '    parser.add_argument("--selection-strategy", choices=["acc_auc", "paper", "distortion", "redline"], default="paper")  # [A1-SEL]\n'
        '    parser.add_argument("--redline-ssim-oc-max", type=float, default=0.15,\n'
        '                        help="[A1-SEL] ssim_oc upper bound for redline eligibility.")\n'
        '    parser.add_argument("--redline-raw-auc-min", type=float, default=0.60,\n'
        '                        help="[A1-SEL] raw_auc lower bound for redline eligibility.")'
    )
    src = _patch_text(src, old_cli, new_cli, "fig:argparse")

    old_inv = (
        "        selection_log_fallback=args.selection_log_fallback,"
    )
    new_inv = (
        "        selection_log_fallback=args.selection_log_fallback,\n"
        "        redline_ssim_oc_max=float(getattr(args, \"redline_ssim_oc_max\", 0.15)),  # [A1-SEL]\n"
        "        redline_raw_auc_min=float(getattr(args, \"redline_raw_auc_min\", 0.60)),  # [A1-SEL]"
    )
    src = _patch_text(src, old_inv, new_inv, "fig:call")

    old_sum = (
        '        "selection_log_fallback": args.selection_log_fallback,'
    )
    new_sum = (
        '        "selection_log_fallback": args.selection_log_fallback,\n'
        '        "redline_ssim_oc_max": float(getattr(args, "redline_ssim_oc_max", 0.15)),  # [A1-SEL]\n'
        '        "redline_raw_auc_min": float(getattr(args, "redline_raw_auc_min", 0.60)),  # [A1-SEL]'
    )
    src = _patch_text(src, old_sum, new_sum, "fig:summary")

    FIG.write_text(src, encoding="utf-8")


if __name__ == "__main__":
    print(f"[A1-SEL] applying redline selection patch to {RUNNER.name} and {FIG.name}")
    _backup(RUNNER)
    _backup(FIG)
    patch_runner()
    patch_runner_part2()
    patch_fig()
    print("[A1-SEL] done")
