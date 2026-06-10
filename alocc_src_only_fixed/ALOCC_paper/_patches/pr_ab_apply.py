"""PR-A + PR-B unified patch (ADR-007 compliant, regression-safe).

- PR-A (mnist_experiment_runner.py + run_paper_mnist_figure6_7.py):
  transparent selection fallback with --selection-min-auc-hard /
  --selection-log-fallback switches; emits `switches` + `fallback_*` in
  summary.json. Defaults preserve current silent-fallback behavior.
- PR-B (model.py): wire stop_recon_threshold / stop_min_epoch into
  ALOCC_LOSS_CLS._train, mirroring ALOCC._train / ALOCC_LOSS._train.

Idempotent: reapply is a no-op. Writes `*.pr_ab.bak` backups on first run.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
RUNNER = ROOT / "mnist_experiment_runner.py"
PIPELINE = ROOT / "run_paper_mnist_figure6_7.py"
MODEL = ROOT / "model.py"


def _apply(path: Path, old: str, new: str, tag: str) -> int:
    text = path.read_text(encoding="utf-8")
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
    bak = path.with_suffix(path.suffix + ".pr_ab.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[{tag}] backup: {bak.name}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[{tag}] patched: {path.name}")
    return 0


PATCHES: list[tuple[str, Path, str, str]] = []


# ---------- PR-B: model.py ALOCC_LOSS_CLS._train early stop ----------
PRB_OLD = (
    "                # 保存检查点\n"
    "                if (i + 1) % step == 0:\n"
    "                    self._save_checkpoint(os.path.join(checkpoint_dir,f\"{i+1}.pth\"))\n"
    "        return int(epoch)\n"
)
PRB_NEW = (
    "                # 保存检查点\n"
    "                if (i + 1) % step == 0:\n"
    "                    self._save_checkpoint(os.path.join(checkpoint_dir,f\"{i+1}.pth\"))\n"
    "                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):\n"
    "                    mean_g_r = float(epoch_g_r_total / cnt)\n"
    "                    if mean_g_r < float(stop_recon_threshold):\n"
    "                        if (i + 1) % step != 0:\n"
    "                            self._save_checkpoint(os.path.join(checkpoint_dir, f\"{i+1}.pth\"))\n"
    "                        return int(i + 1)\n"
    "        return int(epoch)\n"
)
PATCHES.append(("PR-B", MODEL, PRB_OLD, PRB_NEW))


# ---------- PR-A runner: import sys (for stderr logging) ----------
PRA_RUNNER_IMPORT_OLD = (
    "import argparse\n"
    "import json\n"
    "import os\n"
    "import shutil\n"
    "from pathlib import Path\n"
)
PRA_RUNNER_IMPORT_NEW = (
    "import argparse\n"
    "import json\n"
    "import os\n"
    "import shutil\n"
    "import sys\n"
    "from pathlib import Path\n"
)
PATCHES.append(("PR-A[import]", RUNNER, PRA_RUNNER_IMPORT_OLD, PRA_RUNNER_IMPORT_NEW))


# ---------- PR-A runner: _select_records signature + fallback diagnostics ----------
PRA_SELECT_OLD = (
    "def _select_records(records, selection_strategy: str, selection_epoch_start, selection_epoch_end, selection_min_acc, selection_min_auc):\n"
    "    candidates = records\n"
    "    if selection_epoch_start is not None:\n"
    "        candidates = [record for record in candidates if record[\"epoch\"] >= selection_epoch_start]\n"
    "    if selection_epoch_end is not None:\n"
    "        candidates = [record for record in candidates if record[\"epoch\"] <= selection_epoch_end]\n"
    "    if not candidates:\n"
    "        raise ValueError(\"No checkpoint records remain after applying the epoch selection window\")\n"
    "\n"
    "    eligible = candidates\n"
    "    if selection_min_acc is not None:\n"
    "        eligible = [record for record in eligible if record[\"acc\"] >= selection_min_acc]\n"
    "    if selection_min_auc is not None:\n"
    "        eligible = [record for record in eligible if record[\"auc\"] >= selection_min_auc]\n"
    "    if not eligible:\n"
    "        eligible = candidates\n"
)
PRA_SELECT_NEW = (
    "def _select_records(records, selection_strategy: str, selection_epoch_start, selection_epoch_end, selection_min_acc, selection_min_auc, selection_min_auc_hard: bool = False, selection_log_fallback: bool = True):\n"
    "    candidates = records\n"
    "    if selection_epoch_start is not None:\n"
    "        candidates = [record for record in candidates if record[\"epoch\"] >= selection_epoch_start]\n"
    "    if selection_epoch_end is not None:\n"
    "        candidates = [record for record in candidates if record[\"epoch\"] <= selection_epoch_end]\n"
    "    if not candidates:\n"
    "        raise ValueError(\"No checkpoint records remain after applying the epoch selection window\")\n"
    "\n"
    "    eligible = candidates\n"
    "    if selection_min_acc is not None:\n"
    "        eligible = [record for record in eligible if record[\"acc\"] >= selection_min_acc]\n"
    "    if selection_min_auc is not None:\n"
    "        eligible = [record for record in eligible if record[\"auc\"] >= selection_min_auc]\n"
    "    fallback_triggered = False\n"
    "    fallback_reason = None\n"
    "    if not eligible:\n"
    "        fallback_triggered = True\n"
    "        reason_parts = []\n"
    "        if selection_min_acc is not None:\n"
    "            reason_parts.append(f\"acc>={selection_min_acc}\")\n"
    "        if selection_min_auc is not None:\n"
    "            reason_parts.append(f\"auc>={selection_min_auc}\")\n"
    "        constraint = \"+\".join(reason_parts) if reason_parts else \"(no thresholds set)\"\n"
    "        best_auc = max((r[\"auc\"] for r in candidates), default=float(\"nan\"))\n"
    "        fallback_reason = (\n"
    "            f\"No candidate epoch in {[r['epoch'] for r in candidates]} satisfied {constraint}; \"\n"
    "            f\"best auc in window = {best_auc:.4f}\"\n"
    "        )\n"
    "        if selection_min_auc_hard:\n"
    "            raise RuntimeError(f\"[PR-A] selection_min_auc_hard=True and fallback would trigger. {fallback_reason}\")\n"
    "        if selection_log_fallback:\n"
    "            print(f\"[PR-A][selection] WARNING fallback to full candidate window: {fallback_reason}\", file=sys.stderr)\n"
    "        eligible = candidates\n"
)
PATCHES.append(("PR-A[_select]", RUNNER, PRA_SELECT_OLD, PRA_SELECT_NEW))


# ---------- PR-A runner: selection_info dict expansion ----------
PRA_SELINFO_OLD = (
    "    selection_info = {\n"
    "        \"strategy\": selection_strategy,\n"
    "        \"epoch_start\": selection_epoch_start,\n"
    "        \"epoch_end\": selection_epoch_end,\n"
    "        \"min_acc\": selection_min_acc,\n"
    "        \"min_auc\": selection_min_auc,\n"
    "        \"candidate_epochs\": [record[\"epoch\"] for record in candidates],\n"
    "        \"eligible_epochs\": [record[\"epoch\"] for record in eligible],\n"
    "    }\n"
)
PRA_SELINFO_NEW = (
    "    selection_info = {\n"
    "        \"strategy\": selection_strategy,\n"
    "        \"epoch_start\": selection_epoch_start,\n"
    "        \"epoch_end\": selection_epoch_end,\n"
    "        \"min_acc\": selection_min_acc,\n"
    "        \"min_auc\": selection_min_auc,\n"
    "        \"min_auc_hard\": bool(selection_min_auc_hard),\n"
    "        \"log_fallback\": bool(selection_log_fallback),\n"
    "        \"fallback_triggered\": bool(fallback_triggered),\n"
    "        \"fallback_reason\": fallback_reason,\n"
    "        \"candidate_epochs\": [record[\"epoch\"] for record in candidates],\n"
    "        \"eligible_epochs\": [record[\"epoch\"] for record in eligible],\n"
    "    }\n"
)
PATCHES.append(("PR-A[selinfo]", RUNNER, PRA_SELINFO_OLD, PRA_SELINFO_NEW))


# ---------- PR-A runner: evaluate_checkpoints signature ----------
PRA_EVALSIG_OLD = (
    "    selection_strategy: str,\n"
    "    selection_epoch_start,\n"
    "    selection_epoch_end,\n"
    "    selection_min_acc,\n"
    "    selection_min_auc,\n"
    "):\n"
    "    records = []\n"
)
PRA_EVALSIG_NEW = (
    "    selection_strategy: str,\n"
    "    selection_epoch_start,\n"
    "    selection_epoch_end,\n"
    "    selection_min_acc,\n"
    "    selection_min_auc,\n"
    "    selection_min_auc_hard: bool = False,\n"
    "    selection_log_fallback: bool = True,\n"
    "):\n"
    "    records = []\n"
)
PATCHES.append(("PR-A[evalsig]", RUNNER, PRA_EVALSIG_OLD, PRA_EVALSIG_NEW))


# ---------- PR-A runner: evaluate_checkpoints -> _select_records call ----------
PRA_EVALCALL_OLD = (
    "    best_metrics, selection_info = _select_records(\n"
    "        records=records,\n"
    "        selection_strategy=selection_strategy,\n"
    "        selection_epoch_start=selection_epoch_start,\n"
    "        selection_epoch_end=selection_epoch_end,\n"
    "        selection_min_acc=selection_min_acc,\n"
    "        selection_min_auc=selection_min_auc,\n"
    "    )\n"
)
PRA_EVALCALL_NEW = (
    "    best_metrics, selection_info = _select_records(\n"
    "        records=records,\n"
    "        selection_strategy=selection_strategy,\n"
    "        selection_epoch_start=selection_epoch_start,\n"
    "        selection_epoch_end=selection_epoch_end,\n"
    "        selection_min_acc=selection_min_acc,\n"
    "        selection_min_auc=selection_min_auc,\n"
    "        selection_min_auc_hard=selection_min_auc_hard,\n"
    "        selection_log_fallback=selection_log_fallback,\n"
    "    )\n"
)
PATCHES.append(("PR-A[evalcall]", RUNNER, PRA_EVALCALL_OLD, PRA_EVALCALL_NEW))


# ---------- PR-A runner: run_experiment -> evaluate_checkpoints call ----------
PRA_RUNCALL_OLD = (
    "        selection_min_acc=args.selection_min_acc,\n"
    "        selection_min_auc=args.selection_min_auc,\n"
    "    )\n"
)
PRA_RUNCALL_NEW = (
    "        selection_min_acc=args.selection_min_acc,\n"
    "        selection_min_auc=args.selection_min_auc,\n"
    "        selection_min_auc_hard=bool(getattr(args, \"selection_min_auc_hard\", False)),\n"
    "        selection_log_fallback=bool(getattr(args, \"selection_log_fallback\", True)),\n"
    "    )\n"
)
PATCHES.append(("PR-A[runcall]", RUNNER, PRA_RUNCALL_OLD, PRA_RUNCALL_NEW))


# ---------- PR-A runner: summary.json switches block (ADR-007 \u00a79.2) ----------
PRA_SUMMARY_OLD = (
    "        \"test_outlier_labels\": args.test_outlier_labels,\n"
    "        \"selection_info\": selection_info,\n"
)
PRA_SUMMARY_NEW = (
    "        \"test_outlier_labels\": args.test_outlier_labels,\n"
    "        \"switches\": {\n"
    "            \"selection_min_auc_hard\": bool(getattr(args, \"selection_min_auc_hard\", False)),\n"
    "            \"selection_log_fallback\": bool(getattr(args, \"selection_log_fallback\", True)),\n"
    "            \"selection_fallback_triggered\": bool(selection_info.get(\"fallback_triggered\", False)),\n"
    "            \"stop_recon_threshold_active\": getattr(args, \"stop_recon_threshold\", None) is not None,\n"
    "            \"d_outclass_loss_active\": float(getattr(args, \"d_outclass_loss_scale\", 0.0) or 0.0) > 0.0,\n"
    "            \"g_outclass_distortion_active\": float(getattr(args, \"g_outclass_distortion_scale\", 0.0) or 0.0) > 0.0,\n"
    "        },\n"
    "        \"selection_info\": selection_info,\n"
)
PATCHES.append(("PR-A[summary]", RUNNER, PRA_SUMMARY_OLD, PRA_SUMMARY_NEW))


# ---------- PR-A runner: argparse flags ----------
PRA_CLI_RUNNER_OLD = (
    "    parser.add_argument(\"--selection-min-acc\", type=float, default=None)\n"
    "    parser.add_argument(\"--selection-min-auc\", type=float, default=None)\n"
    "    return parser.parse_args()\n"
)
PRA_CLI_RUNNER_NEW = (
    "    parser.add_argument(\"--selection-min-acc\", type=float, default=None)\n"
    "    parser.add_argument(\"--selection-min-auc\", type=float, default=None)\n"
    "    parser.add_argument(\"--selection-min-auc-hard\", action=\"store_true\", default=False,\n"
    "                        help=\"[PR-A] Raise instead of silently falling back when no epoch meets --selection-min-auc.\")\n"
    "    parser.add_argument(\"--selection-log-fallback\", dest=\"selection_log_fallback\", action=\"store_true\", default=True,\n"
    "                        help=\"[PR-A] Log a stderr warning when fallback triggers (default: on).\")\n"
    "    parser.add_argument(\"--no-selection-log-fallback\", dest=\"selection_log_fallback\", action=\"store_false\",\n"
    "                        help=\"[PR-A] Disable the fallback warning.\")\n"
    "    return parser.parse_args()\n"
)
PATCHES.append(("PR-A[cli-runner]", RUNNER, PRA_CLI_RUNNER_OLD, PRA_CLI_RUNNER_NEW))


# ---------- PR-A pipeline: Args dataclass fields ----------
PRA_ARGS_FIELDS_OLD = (
    "    selection_min_acc: float | None\n"
    "    selection_min_auc: float | None\n"
    "\n"
    "\n"
    "def _figure7_scores("
)
PRA_ARGS_FIELDS_NEW = (
    "    selection_min_acc: float | None\n"
    "    selection_min_auc: float | None\n"
    "    selection_min_auc_hard: bool = False\n"
    "    selection_log_fallback: bool = True\n"
    "\n"
    "\n"
    "def _figure7_scores("
)
PATCHES.append(("PR-A[args-fields]", PIPELINE, PRA_ARGS_FIELDS_OLD, PRA_ARGS_FIELDS_NEW))


# ---------- PR-A pipeline: Args(...) construction ----------
PRA_ARGS_CTOR_OLD = (
    "        selection_min_acc=None,\n"
    "        selection_min_auc=args.selection_min_auc,\n"
    "    )\n"
)
PRA_ARGS_CTOR_NEW = (
    "        selection_min_acc=None,\n"
    "        selection_min_auc=args.selection_min_auc,\n"
    "        selection_min_auc_hard=args.selection_min_auc_hard,\n"
    "        selection_log_fallback=args.selection_log_fallback,\n"
    "    )\n"
)
PATCHES.append(("PR-A[args-ctor]", PIPELINE, PRA_ARGS_CTOR_OLD, PRA_ARGS_CTOR_NEW))


# ---------- PR-A pipeline: argparse flags ----------
PRA_CLI_PIPELINE_OLD = (
    "    parser.add_argument(\"--selection-min-auc\", type=float, default=0.95)\n"
    "    parser.add_argument(\"--triplet-count\", type=int, default=12)\n"
)
PRA_CLI_PIPELINE_NEW = (
    "    parser.add_argument(\"--selection-min-auc\", type=float, default=0.95)\n"
    "    parser.add_argument(\"--selection-min-auc-hard\", action=\"store_true\", default=False,\n"
    "                        help=\"[PR-A] Hard-fail instead of silent fallback.\")\n"
    "    parser.add_argument(\"--selection-log-fallback\", dest=\"selection_log_fallback\", action=\"store_true\", default=True,\n"
    "                        help=\"[PR-A] Warn on fallback (default: on).\")\n"
    "    parser.add_argument(\"--no-selection-log-fallback\", dest=\"selection_log_fallback\", action=\"store_false\",\n"
    "                        help=\"[PR-A] Disable fallback warning.\")\n"
    "    parser.add_argument(\"--triplet-count\", type=int, default=12)\n"
)
PATCHES.append(("PR-A[cli-pipeline]", PIPELINE, PRA_CLI_PIPELINE_OLD, PRA_CLI_PIPELINE_NEW))


# ---------- PR-A pipeline: pipeline_summary.json passthrough ----------
PRA_PIPELINE_SUMMARY_OLD = (
    "        \"selection_min_auc\": args.selection_min_auc,\n"
    "    }\n"
)
PRA_PIPELINE_SUMMARY_NEW = (
    "        \"selection_min_auc\": args.selection_min_auc,\n"
    "        \"selection_min_auc_hard\": args.selection_min_auc_hard,\n"
    "        \"selection_log_fallback\": args.selection_log_fallback,\n"
    "    }\n"
)
PATCHES.append(("PR-A[pipeline-summary]", PIPELINE, PRA_PIPELINE_SUMMARY_OLD, PRA_PIPELINE_SUMMARY_NEW))


def main() -> int:
    rc = 0
    for tag, path, old, new in PATCHES:
        if not path.exists():
            print(f"[{tag}] target missing: {path}", file=sys.stderr)
            rc = 3
            continue
        rc |= _apply(path, old, new, tag)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
