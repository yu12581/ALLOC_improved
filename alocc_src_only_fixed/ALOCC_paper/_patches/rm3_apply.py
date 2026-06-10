"""RM-3 unified patch (ADR-007 compliant, regression-safe).

- RM-3a: new `distortion` selection strategy computing
  `ssim_gap^alpha * refined_auc^beta`. Guarded by --selection-strategy
  distortion + --distortion-alpha (default 1.0) + --distortion-beta (1.0).
- RM-3b: absolute anchor normalization for paper_score via
  --paper-score-normalization absolute. Default stays 'relative' =>
  bitwise regression with baselines. Hybrid: only the 7 North-Star-ish
  keys (ssim_gap, ssim_oc, acc, refined_auc, auc_gain, score_gap,
  score_gap_gain) use anchor mapping; auxiliary keys remain relative.
- RM-3c: lower --selection-min-auc default 0.95 -> 0.60. _run_all.ps1
  explicitly passes 0.95 so baseline regression is unaffected.

Idempotent: reapply is a no-op. Writes `*.rm3.bak` backups on first run.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
RUNNER = ROOT / "mnist_experiment_runner.py"
PIPELINE = ROOT / "run_paper_mnist_figure6_7.py"


def _apply(path: Path, old: str, new: str, tag: str, sentinel: str | None = None) -> int:
    text = path.read_text(encoding="utf-8")
    if sentinel is None and old in new:
        diff = new.replace(old, "", 1)
        stripped = diff.strip()
        sentinel = stripped.splitlines()[0] if stripped else None
    if sentinel is not None and sentinel in text:
        print(f"[{tag}] already applied in {path.name}")
        return 0
    old_count = text.count(old)
    if old_count == 0:
        if new in text:
            print(f"[{tag}] already applied in {path.name}")
            return 0
        print(f"[{tag}] anchor missing in {path.name}", file=sys.stderr)
        return 2
    if old_count != 1:
        print(
            f"[{tag}] anchor count={old_count} (expected 1) in {path.name}",
            file=sys.stderr,
        )
        return 2
    bak = path.with_suffix(path.suffix + ".rm3.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[{tag}] backup: {bak.name}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[{tag}] patched: {path.name}")
    return 0


PATCHES: list[tuple[str, Path, str, str]] = []


# ---------------------------------------------------------------------------
# RM-3a/3b: insert absolute anchor table + absolute normalizer + distortion
# score attacher, placed just before `_normalize_metric`.
# ---------------------------------------------------------------------------
PATCHES.append((
    "RM-3a/3b insert helpers",
    RUNNER,
    "def _normalize_metric(records, key: str, invert: bool = False):\n",
    "PAPER_SCORE_ABSOLUTE_ANCHORS = {\n"
    "    # key: (zero_line, one_line). Direction encoded by order.\n"
    "    # Anchors approved 2026-04-19 (user): see PROJECT_LOG.md \u00a79.5.\n"
    "    \"acc\":            (0.80, 0.95),\n"
    "    \"auc_gain\":       (0.00, 0.15),\n"
    "    \"ssim_gap\":       (0.00, 0.30),\n"
    "    \"ssim_oc\":        (0.50, 0.10),\n"
    "    \"score_gap\":      (0.00, 0.20),\n"
    "    \"score_gap_gain\": (0.00, 0.10),\n"
    "    \"refined_auc\":    (0.50, 0.95),\n"
    "}\n"
    "\n"
    "\n"
    "def _normalize_metric_absolute(records, key: str, zero_line: float, one_line: float):\n"
    "    \"\"\"Linearly map each record[key] to [0,1] using absolute anchors (RM-3b).\"\"\"\n"
    "    scale = one_line - zero_line\n"
    "    normalized = {}\n"
    "    for record in records:\n"
    "        value = float(record[key])\n"
    "        if scale == 0:\n"
    "            normalized[record[\"epoch\"]] = 0.5\n"
    "            continue\n"
    "        frac = (value - zero_line) / scale\n"
    "        frac = max(0.0, min(1.0, frac))\n"
    "        normalized[record[\"epoch\"]] = frac\n"
    "    return normalized\n"
    "\n"
    "\n"
    "def _attach_distortion_score(records, alpha: float = 1.0, beta: float = 1.0):\n"
    "    \"\"\"RM-3a distortion score: max(ssim_gap,0)^alpha * max(refined_auc,0)^beta.\"\"\"\n"
    "    for record in records:\n"
    "        gap = max(0.0, float(record[\"ssim_gap\"]))\n"
    "        rauc = max(0.0, float(record[\"refined_auc\"]))\n"
    "        score = (gap ** float(alpha)) * (rauc ** float(beta))\n"
    "        record[\"distortion_score\"] = float(score)\n"
    "        record[\"distortion_components\"] = {\n"
    "            \"ssim_gap\": float(record[\"ssim_gap\"]),\n"
    "            \"refined_auc\": float(record[\"refined_auc\"]),\n"
    "            \"alpha\": float(alpha),\n"
    "            \"beta\": float(beta),\n"
    "        }\n"
    "\n"
    "\n"
    "def _normalize_metric(records, key: str, invert: bool = False):\n",
))

# ---------------------------------------------------------------------------
# RM-3b: _attach_paper_score grows a `normalization` kwarg (default "relative"
# preserves bitwise behavior). Absolute mode only rewires keys present in the
# anchor table; auxiliary keys fall back to relative normalization.
# ---------------------------------------------------------------------------
PATCHES.append((
    "RM-3b _attach_paper_score",
    RUNNER,
    "def _attach_paper_score(records):\n"
    "    if not records:\n"
    "        return\n"
    "\n"
    "    normalized = {\n"
    "        \"acc\": _normalize_metric(records, \"acc\"),\n"
    "        \"auc\": _normalize_metric(records, \"auc\"),\n"
    "        \"eer\": _normalize_metric(records, \"eer\", invert=True),\n"
    "        \"raw_auc\": _normalize_metric(records, \"raw_auc\"),\n"
    "        \"ssim_ic\": _normalize_metric(records, \"ssim_ic\"),\n"
    "        \"ssim_oc\": _normalize_metric(records, \"ssim_oc\", invert=True),\n"
    "        \"vif_oc\": _normalize_metric(records, \"vif_oc\", invert=True),\n"
    "        \"ssim_gap\": _normalize_metric(records, \"ssim_gap\"),\n"
    "        \"vif_gap\": _normalize_metric(records, \"vif_gap\"),\n"
    "        \"gmsd_gap\": _normalize_metric(records, \"gmsd_gap\"),\n"
    "        \"score_gap\": _normalize_metric(records, \"score_gap\"),\n"
    "        \"raw_score_gap\": _normalize_metric(records, \"raw_score_gap\"),\n"
    "        \"score_gap_gain\": _normalize_metric(records, \"score_gap_gain\"),\n"
    "        \"auc_gain\": _normalize_metric(records, \"auc_gain\"),\n"
    "    }\n",
    "def _attach_paper_score(records, normalization: str = \"relative\"):\n"
    "    if not records:\n"
    "        return\n"
    "\n"
    "    def _norm(key, invert=False):\n"
    "        if normalization == \"absolute\" and key in PAPER_SCORE_ABSOLUTE_ANCHORS:\n"
    "            z, o = PAPER_SCORE_ABSOLUTE_ANCHORS[key]\n"
    "            return _normalize_metric_absolute(records, key, z, o)\n"
    "        return _normalize_metric(records, key, invert=invert)\n"
    "\n"
    "    normalized = {\n"
    "        \"acc\": _norm(\"acc\"),\n"
    "        \"auc\": _norm(\"auc\"),\n"
    "        \"eer\": _norm(\"eer\", invert=True),\n"
    "        \"raw_auc\": _norm(\"raw_auc\"),\n"
    "        \"ssim_ic\": _norm(\"ssim_ic\"),\n"
    "        \"ssim_oc\": _norm(\"ssim_oc\", invert=True),\n"
    "        \"vif_oc\": _norm(\"vif_oc\", invert=True),\n"
    "        \"ssim_gap\": _norm(\"ssim_gap\"),\n"
    "        \"vif_gap\": _norm(\"vif_gap\"),\n"
    "        \"gmsd_gap\": _norm(\"gmsd_gap\"),\n"
    "        \"score_gap\": _norm(\"score_gap\"),\n"
    "        \"raw_score_gap\": _norm(\"raw_score_gap\"),\n"
    "        \"score_gap_gain\": _norm(\"score_gap_gain\"),\n"
    "        \"auc_gain\": _norm(\"auc_gain\"),\n"
    "    }\n",
))

# ---------------------------------------------------------------------------
# RM-3a: grow _select_records signature with distortion_alpha/beta, add the
# `distortion` branch, and echo the new params into selection_info.
# ---------------------------------------------------------------------------
PATCHES.append((
    "RM-3a _select_records signature",
    RUNNER,
    "def _select_records(records, selection_strategy: str, selection_epoch_start, selection_epoch_end, selection_min_acc, selection_min_auc, selection_min_auc_hard: bool = False, selection_log_fallback: bool = True):\n",
    "def _select_records(records, selection_strategy: str, selection_epoch_start, selection_epoch_end, selection_min_acc, selection_min_auc, selection_min_auc_hard: bool = False, selection_log_fallback: bool = True, distortion_alpha: float = 1.0, distortion_beta: float = 1.0):\n",
))

PATCHES.append((
    "RM-3a _select_records distortion branch",
    RUNNER,
    "    elif selection_strategy == \"paper\":\n"
    "        best_record = max(\n"
    "            eligible,\n"
    "            key=lambda record: (\n"
    "                record[\"paper_score\"],\n"
    "                record[\"auc_gain\"],\n"
    "                record[\"score_gap_gain\"],\n"
    "                record[\"ssim_gap\"],\n"
    "                -record[\"ssim_oc\"],\n"
    "                record[\"score_gap\"],\n"
    "                record[\"gmsd_gap\"],\n"
    "                record[\"acc\"],\n"
    "                record[\"auc\"],\n"
    "                -record[\"eer\"],\n"
    "            ),\n"
    "        )\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown selection strategy: {selection_strategy}\")\n",
    "    elif selection_strategy == \"paper\":\n"
    "        best_record = max(\n"
    "            eligible,\n"
    "            key=lambda record: (\n"
    "                record[\"paper_score\"],\n"
    "                record[\"auc_gain\"],\n"
    "                record[\"score_gap_gain\"],\n"
    "                record[\"ssim_gap\"],\n"
    "                -record[\"ssim_oc\"],\n"
    "                record[\"score_gap\"],\n"
    "                record[\"gmsd_gap\"],\n"
    "                record[\"acc\"],\n"
    "                record[\"auc\"],\n"
    "                -record[\"eer\"],\n"
    "            ),\n"
    "        )\n"
    "    elif selection_strategy == \"distortion\":\n"
    "        best_record = max(\n"
    "            eligible,\n"
    "            key=lambda record: (\n"
    "                record[\"distortion_score\"],\n"
    "                record[\"auc_gain\"],\n"
    "                record[\"ssim_gap\"],\n"
    "                -record[\"ssim_oc\"],\n"
    "                record[\"paper_score\"],\n"
    "                record[\"acc\"],\n"
    "                record[\"auc\"],\n"
    "                -record[\"eer\"],\n"
    "            ),\n"
    "        )\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown selection strategy: {selection_strategy}\")\n",
))

PATCHES.append((
    "RM-3a _select_records selection_info echo",
    RUNNER,
    "        \"log_fallback\": bool(selection_log_fallback),\n"
    "        \"fallback_triggered\": bool(fallback_triggered),\n"
    "        \"fallback_reason\": fallback_reason,\n"
    "        \"candidate_epochs\": [record[\"epoch\"] for record in candidates],\n"
    "        \"eligible_epochs\": [record[\"epoch\"] for record in eligible],\n"
    "    }\n",
    "        \"log_fallback\": bool(selection_log_fallback),\n"
    "        \"fallback_triggered\": bool(fallback_triggered),\n"
    "        \"fallback_reason\": fallback_reason,\n"
    "        \"candidate_epochs\": [record[\"epoch\"] for record in candidates],\n"
    "        \"eligible_epochs\": [record[\"epoch\"] for record in eligible],\n"
    "        \"distortion_alpha\": float(distortion_alpha),\n"
    "        \"distortion_beta\": float(distortion_beta),\n"
    "    }\n",
))

# ---------------------------------------------------------------------------
# RM-3: evaluate_checkpoints signature grows by 3 kwargs (all defaulted so the
# existing call surface is unchanged). Also wire paper_score normalization +
# distortion score attachment into its body and forward the new kwargs into
# _select_records.
# ---------------------------------------------------------------------------
PATCHES.append((
    "RM-3 evaluate_checkpoints signature",
    RUNNER,
    "def evaluate_checkpoints(\n"
    "    model,\n"
    "    checkpoint_dir: Path,\n"
    "    dataloader,\n"
    "    epochs: int,\n"
    "    inner_class: int,\n"
    "    selection_strategy: str,\n"
    "    selection_epoch_start,\n"
    "    selection_epoch_end,\n"
    "    selection_min_acc,\n"
    "    selection_min_auc,\n"
    "    selection_min_auc_hard: bool = False,\n"
    "    selection_log_fallback: bool = True,\n"
    "):\n",
    "def evaluate_checkpoints(\n"
    "    model,\n"
    "    checkpoint_dir: Path,\n"
    "    dataloader,\n"
    "    epochs: int,\n"
    "    inner_class: int,\n"
    "    selection_strategy: str,\n"
    "    selection_epoch_start,\n"
    "    selection_epoch_end,\n"
    "    selection_min_acc,\n"
    "    selection_min_auc,\n"
    "    selection_min_auc_hard: bool = False,\n"
    "    selection_log_fallback: bool = True,\n"
    "    distortion_alpha: float = 1.0,\n"
    "    distortion_beta: float = 1.0,\n"
    "    paper_score_normalization: str = \"relative\",\n"
    "):\n",
))

PATCHES.append((
    "RM-3 evaluate_checkpoints body",
    RUNNER,
    "    _attach_paper_score(records)\n"
    "    best_metrics, selection_info = _select_records(\n"
    "        records=records,\n"
    "        selection_strategy=selection_strategy,\n"
    "        selection_epoch_start=selection_epoch_start,\n"
    "        selection_epoch_end=selection_epoch_end,\n"
    "        selection_min_acc=selection_min_acc,\n"
    "        selection_min_auc=selection_min_auc,\n"
    "        selection_min_auc_hard=selection_min_auc_hard,\n"
    "        selection_log_fallback=selection_log_fallback,\n"
    "    )\n",
    "    _attach_paper_score(records, normalization=paper_score_normalization)\n"
    "    _attach_distortion_score(records, alpha=distortion_alpha, beta=distortion_beta)\n"
    "    best_metrics, selection_info = _select_records(\n"
    "        records=records,\n"
    "        selection_strategy=selection_strategy,\n"
    "        selection_epoch_start=selection_epoch_start,\n"
    "        selection_epoch_end=selection_epoch_end,\n"
    "        selection_min_acc=selection_min_acc,\n"
    "        selection_min_auc=selection_min_auc,\n"
    "        selection_min_auc_hard=selection_min_auc_hard,\n"
    "        selection_log_fallback=selection_log_fallback,\n"
    "        distortion_alpha=distortion_alpha,\n"
    "        distortion_beta=distortion_beta,\n"
    "    )\n",
))




# ---------------------------------------------------------------------------
# RM-3: run_experiment forwards the 3 new args to evaluate_checkpoints, and
# echoes them into summary.switches so consumers can see the RM-3 state.
# ---------------------------------------------------------------------------
PATCHES.append((
    "RM-3 run_experiment forward",
    RUNNER,
    "        selection_min_auc_hard=bool(getattr(args, \"selection_min_auc_hard\", False)),\n"
    "        selection_log_fallback=bool(getattr(args, \"selection_log_fallback\", True)),\n"
    "    )\n",
    "        selection_min_auc_hard=bool(getattr(args, \"selection_min_auc_hard\", False)),\n"
    "        selection_log_fallback=bool(getattr(args, \"selection_log_fallback\", True)),\n"
    "        distortion_alpha=float(getattr(args, \"distortion_alpha\", 1.0)),\n"
    "        distortion_beta=float(getattr(args, \"distortion_beta\", 1.0)),\n"
    "        paper_score_normalization=getattr(args, \"paper_score_normalization\", \"relative\"),\n"
    "    )\n",
))

PATCHES.append((
    "RM-3 run_experiment switches echo",
    RUNNER,
    "        \"switches\": {\n"
    "            \"selection_min_auc_hard\": bool(getattr(args, \"selection_min_auc_hard\", False)),\n"
    "            \"selection_log_fallback\": bool(getattr(args, \"selection_log_fallback\", True)),\n"
    "            \"selection_fallback_triggered\": bool(selection_info.get(\"fallback_triggered\", False)),\n"
    "            \"stop_recon_threshold_active\": getattr(args, \"stop_recon_threshold\", None) is not None,\n"
    "            \"d_outclass_loss_active\": float(getattr(args, \"d_outclass_loss_scale\", 0.0) or 0.0) > 0.0,\n"
    "            \"g_outclass_distortion_active\": float(getattr(args, \"g_outclass_distortion_scale\", 0.0) or 0.0) > 0.0,\n"
    "        },\n",
    "        \"switches\": {\n"
    "            \"selection_min_auc_hard\": bool(getattr(args, \"selection_min_auc_hard\", False)),\n"
    "            \"selection_log_fallback\": bool(getattr(args, \"selection_log_fallback\", True)),\n"
    "            \"selection_fallback_triggered\": bool(selection_info.get(\"fallback_triggered\", False)),\n"
    "            \"stop_recon_threshold_active\": getattr(args, \"stop_recon_threshold\", None) is not None,\n"
    "            \"d_outclass_loss_active\": float(getattr(args, \"d_outclass_loss_scale\", 0.0) or 0.0) > 0.0,\n"
    "            \"g_outclass_distortion_active\": float(getattr(args, \"g_outclass_distortion_scale\", 0.0) or 0.0) > 0.0,\n"
    "            \"selection_strategy\": str(args.selection_strategy),\n"
    "            \"selection_min_auc\": (float(args.selection_min_auc) if args.selection_min_auc is not None else None),\n"
    "            \"paper_score_normalization\": str(getattr(args, \"paper_score_normalization\", \"relative\")),\n"
    "            \"distortion_alpha\": float(getattr(args, \"distortion_alpha\", 1.0)),\n"
    "            \"distortion_beta\": float(getattr(args, \"distortion_beta\", 1.0)),\n"
    "        },\n",
))


# ---------------------------------------------------------------------------
# Pipeline (run_paper_mnist_figure6_7.py): Args dataclass + CLI argparse +
# Args() construction + pipeline_summary.json echo. All new fields default
# to RM-3-OFF semantics (relative, alpha=beta=1.0); selection_min_auc default
# drops 0.95 -> 0.60 (RM-3c, baseline PS1 still forces 0.95).
# ---------------------------------------------------------------------------
PATCHES.append((
    "RM-3 pipeline Args dataclass",
    PIPELINE,
    "    selection_min_auc_hard: bool = False\n"
    "    selection_log_fallback: bool = True\n",
    "    selection_min_auc_hard: bool = False\n"
    "    selection_log_fallback: bool = True\n"
    "    distortion_alpha: float = 1.0\n"
    "    distortion_beta: float = 1.0\n"
    "    paper_score_normalization: str = \"relative\"\n",
))

PATCHES.append((
    "RM-3a/b pipeline argparse (strategy + new flags)",
    PIPELINE,
    "    parser.add_argument(\"--selection-strategy\", choices=[\"acc_auc\", \"paper\"], default=\"paper\")\n",
    "    parser.add_argument(\"--selection-strategy\", choices=[\"acc_auc\", \"paper\", \"distortion\"], default=\"paper\")\n"
    "    parser.add_argument(\"--distortion-alpha\", type=float, default=1.0,\n"
    "                        help=\"[RM-3a] exponent for ssim_gap in distortion strategy (default 1.0)\")\n"
    "    parser.add_argument(\"--distortion-beta\", type=float, default=1.0,\n"
    "                        help=\"[RM-3a] exponent for refined_auc in distortion strategy (default 1.0)\")\n"
    "    parser.add_argument(\"--paper-score-normalization\", choices=[\"relative\", \"absolute\"], default=\"relative\",\n"
    "                        help=\"[RM-3b] paper_score normalization mode; 'absolute' uses approved anchors\")\n",
))

PATCHES.append((
    "RM-3c pipeline min_auc default 0.95 -> 0.60",
    PIPELINE,
    "    parser.add_argument(\"--selection-min-auc\", type=float, default=0.95)\n",
    "    parser.add_argument(\"--selection-min-auc\", type=float, default=0.60,\n"
    "                        help=\"[RM-3c] threshold for paper selection; default lowered 0.95->0.60 (2026-04-19)\")\n",
))

PATCHES.append((
    "RM-3 pipeline Args() wire",
    PIPELINE,
    "        selection_min_auc_hard=args.selection_min_auc_hard,\n"
    "        selection_log_fallback=args.selection_log_fallback,\n"
    "    )\n",
    "        selection_min_auc_hard=args.selection_min_auc_hard,\n"
    "        selection_log_fallback=args.selection_log_fallback,\n"
    "        distortion_alpha=args.distortion_alpha,\n"
    "        distortion_beta=args.distortion_beta,\n"
    "        paper_score_normalization=args.paper_score_normalization,\n"
    "    )\n",
))

PATCHES.append((
    "RM-3 pipeline pipeline_summary echo",
    PIPELINE,
    "        \"selection_min_auc\": args.selection_min_auc,\n"
    "        \"selection_min_auc_hard\": args.selection_min_auc_hard,\n"
    "        \"selection_log_fallback\": args.selection_log_fallback,\n"
    "    }\n",
    "        \"selection_min_auc\": args.selection_min_auc,\n"
    "        \"selection_min_auc_hard\": args.selection_min_auc_hard,\n"
    "        \"selection_log_fallback\": args.selection_log_fallback,\n"
    "        \"distortion_alpha\": args.distortion_alpha,\n"
    "        \"distortion_beta\": args.distortion_beta,\n"
    "        \"paper_score_normalization\": args.paper_score_normalization,\n"
    "    }\n",
))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    rc_total = 0
    for tag, path, old, new in PATCHES:
        rc = _apply(path, old, new, tag)
        if rc != 0:
            rc_total = rc
    if rc_total == 0:
        print("\n[rm3_apply] all patches applied (or already present)")
    else:
        print(f"\n[rm3_apply] finished with rc={rc_total}", file=sys.stderr)
    return rc_total


if __name__ == "__main__":
    sys.exit(main())

