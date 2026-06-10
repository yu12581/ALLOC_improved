"""PR-B verification: --stop-recon-threshold wiring in ALOCC_LOSS_CLS._train.

Scenario 1: stop_recon_threshold=10.0 stop_min_epoch=2 (trivially triggers at ep2)
            => trained_epochs should equal 2; switches.stop_recon_threshold_active=true
Scenario 2: no --stop-recon-threshold (default None)
            => trained_epochs == epochs (all consumed); switches.stop_recon_threshold_active=false
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
OUT_ROOT = REPO / "ALOCC_paper" / "_patches" / "_verify_prb"

BASE_CMD = [
    str(PY),
    "mnist_experiment_runner.py",
    "--variant", "alocc_loss_cls",
    "--specific", "1",
    "--epochs", "5",
    "--train-count", "512",
    "--test-inlier-count", "100",
    "--test-outlier-labels", "6", "7",
    "--batch-size", "64",
    "--eval-batch-size", "128",
    "--noise-std", "0.31",
    "--r-alpha", "0.2",
    "--lr", "0.002",
    "--d-outclass-loss-scale", "0.1",
    "--out-per-class-count", "32",
    "--selection-strategy", "acc_auc",
]


def _run(label: str, extra_flags: list[str]) -> dict:
    out = OUT_ROOT / label
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = BASE_CMD + ["--output-dir", str(out)] + extra_flags
    print(f"\n[verify-pr-b][{label}] flags: {' '.join(extra_flags) or '(defaults)'}")
    proc = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True)
    print(f"[verify-pr-b][{label}] rc={proc.returncode}")
    assert proc.returncode == 0, f"[{label}] rc={proc.returncode}\n{proc.stderr}"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return summary


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    s1 = _run("s1_stop_on", ["--stop-recon-threshold", "10.0", "--stop-min-epoch", "2"])
    te1 = int(s1["trained_epochs"])
    sw1 = s1["switches"]
    print(f"  trained_epochs={te1}  epochs={s1['epochs']}")
    print(f"  switches.stop_recon_threshold_active={sw1['stop_recon_threshold_active']}")
    assert sw1["stop_recon_threshold_active"] is True, "S1: stop active flag mismatch"
    assert te1 == 2, f"S1: trained_epochs should be 2, got {te1}"
    ckpt2 = (OUT_ROOT / "s1_stop_on" / "2.pth").exists()
    assert ckpt2, "S1: expected checkpoint 2.pth saved at stop-epoch"

    s2 = _run("s2_stop_off", [])
    te2 = int(s2["trained_epochs"])
    sw2 = s2["switches"]
    print(f"  trained_epochs={te2}  epochs={s2['epochs']}")
    print(f"  switches.stop_recon_threshold_active={sw2['stop_recon_threshold_active']}")
    assert sw2["stop_recon_threshold_active"] is False, "S2: stop flag should be off"
    assert te2 == int(s2["epochs"]), f"S2: trained_epochs should equal epochs, got {te2}"

    print("\n[verify-pr-b] ALL SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
