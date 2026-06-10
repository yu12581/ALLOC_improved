"""PR-A verification: force the selection fallback path and check the new
switches/diagnostics surface.

Scenario 1: --selection-min-auc 0.9999 (unreachable) with default
            --selection-log-fallback on
            => expect stderr warning + summary switches.selection_fallback_triggered=true
Scenario 2: --selection-min-auc 0.9999 --selection-min-auc-hard
            => expect non-zero rc (RuntimeError), no summary written
Scenario 3: --selection-min-auc 0.9999 --no-selection-log-fallback
            => expect NO stderr warning, summary.switches.selection_log_fallback=false
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
PY = SRC / ".venv" / "Scripts" / "python.exe"
OUT_ROOT = REPO / "ALOCC_paper" / "_patches" / "_verify_pra"

BASE_CMD = [
    str(PY),
    "mnist_experiment_runner.py",
    "--variant", "alocc_tiny",
    "--specific", "1",
    "--epochs", "3",
    "--train-count", "256",
    "--test-inlier-count", "100",
    "--test-outlier-labels", "6", "7",
    "--batch-size", "64",
    "--eval-batch-size", "128",
    "--noise-std", "0.31",
    "--r-alpha", "0.2",
    "--lr", "0.002",
    "--selection-strategy", "acc_auc",
    "--selection-min-auc", "0.9999",
]


def _run(label: str, extra_flags: list[str], expect_rc_nonzero: bool = False) -> dict:
    out = OUT_ROOT / label
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = BASE_CMD + ["--output-dir", str(out)] + extra_flags
    print(f"\n[verify-pr-a][{label}] cmd tail: {' '.join(extra_flags) or '(defaults)'}")
    proc = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True)
    print(f"[verify-pr-a][{label}] rc={proc.returncode}")
    stderr_snip = proc.stderr.strip().splitlines()[-6:] if proc.stderr else []
    for ln in stderr_snip:
        print(f"    [stderr] {ln}")
    if expect_rc_nonzero:
        assert proc.returncode != 0, f"[{label}] expected non-zero rc, got 0"
        assert "selection_min_auc_hard" in proc.stderr, f"[{label}] missing hard-fail diagnostic in stderr"
        return {"rc": proc.returncode, "stderr": proc.stderr}
    assert proc.returncode == 0, f"[{label}] unexpected rc={proc.returncode}\n{proc.stderr}"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return {"rc": proc.returncode, "stderr": proc.stderr, "summary": summary}


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    r1 = _run("s1_default", [])
    s = r1["summary"]
    sw = s["switches"]
    si = s["selection_info"]
    print(f"  switches.selection_fallback_triggered = {sw['selection_fallback_triggered']}")
    print(f"  selection_info.fallback_reason = {si['fallback_reason']}")
    assert sw["selection_fallback_triggered"] is True, "S1: fallback_triggered should be true"
    assert si["fallback_reason"] is not None, "S1: fallback_reason should be set"
    assert "[PR-A][selection] WARNING" in r1["stderr"], "S1: stderr warning not emitted"
    assert sw["selection_log_fallback"] is True
    assert sw["selection_min_auc_hard"] is False

    r2 = _run("s2_hard", ["--selection-min-auc-hard"], expect_rc_nonzero=True)
    print(f"  [s2_hard] rc={r2['rc']} (expected non-zero) OK")

    r3 = _run("s3_silent", ["--no-selection-log-fallback"])
    s = r3["summary"]
    sw = s["switches"]
    print(f"  switches.selection_log_fallback = {sw['selection_log_fallback']}")
    print(f"  switches.selection_fallback_triggered = {sw['selection_fallback_triggered']}")
    assert sw["selection_log_fallback"] is False, "S3: log_fallback should be false"
    assert sw["selection_fallback_triggered"] is True, "S3: fallback still triggered"
    assert "[PR-A][selection] WARNING" not in r3["stderr"], "S3: warning should be silenced"

    print("\n[verify-pr-a] ALL SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
