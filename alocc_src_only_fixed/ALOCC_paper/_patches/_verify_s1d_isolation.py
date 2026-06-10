"""[BASELINE-A][ADR-007] S1D-side isolation check.

Constructs ALOCC_LOSS (S1D's class) WITHOUT the tf_verbatim_rmsprop flag and asserts:
  1. self.tf_verbatim_rmsprop == False
  2. optim_D.param_groups[0]['alpha'] == 0.99   (PyTorch default)
  3. optim_D.param_groups[0]['eps']   == 1e-8   (PyTorch default)
  4. optim_G has the same defaults.
Then constructs ALOCC (Baseline A's class) WITH tf_verbatim_rmsprop=True and asserts
the verbatim values land on both optimizers.

This makes ADR-007 enforcement testable without spinning up a real training loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
sys.path.insert(0, str(REPO))

from model import ALOCC, ALOCC_LOSS  # type: ignore  # noqa: E402


def show(name: str, model) -> dict:
    pg_d = model.optim_D.param_groups[0]
    pg_g = model.optim_G.param_groups[0]
    info = {
        "tf_verbatim_rmsprop_attr": getattr(model, "tf_verbatim_rmsprop", "<absent>"),
        "optim_D.alpha": pg_d["alpha"],
        "optim_D.eps":   pg_d["eps"],
        "optim_G.alpha": pg_g["alpha"],
        "optim_G.eps":   pg_g["eps"],
    }
    print(f"--- {name} ---")
    for k, v in info.items():
        print(f"  {k:32s} = {v}")
    return info


def main() -> int:
    # Case A: S1D defaults (no flag) -> historical PyTorch RMSprop.
    s1d = ALOCC_LOSS(in_h=28, out_h=28, lr=0.002)
    a = show("S1D / ALOCC_LOSS (no flag)", s1d)

    # Case B: Baseline A path (flag on) -> TF1.15 verbatim.
    base = ALOCC(in_h=28, out_h=28, lr=0.002, tf_verbatim_rmsprop=True)
    b = show("Baseline A / ALOCC (flag=True)", base)

    fails: list[str] = []

    if a["tf_verbatim_rmsprop_attr"] is not False:
        fails.append("S1D attr != False")
    if abs(a["optim_D.alpha"] - 0.99) > 1e-12:
        fails.append(f"S1D optim_D.alpha={a['optim_D.alpha']} (expected 0.99)")
    if abs(a["optim_D.eps"] - 1e-8) > 1e-20:
        fails.append(f"S1D optim_D.eps={a['optim_D.eps']} (expected 1e-8)")
    if abs(a["optim_G.alpha"] - 0.99) > 1e-12:
        fails.append(f"S1D optim_G.alpha={a['optim_G.alpha']} (expected 0.99)")
    if abs(a["optim_G.eps"] - 1e-8) > 1e-20:
        fails.append(f"S1D optim_G.eps={a['optim_G.eps']} (expected 1e-8)")

    if b["tf_verbatim_rmsprop_attr"] is not True:
        fails.append("Baseline-A attr != True")
    if abs(b["optim_D.alpha"] - 0.9) > 1e-12:
        fails.append(f"BL-A optim_D.alpha={b['optim_D.alpha']} (expected 0.9)")
    if abs(b["optim_D.eps"] - 1e-10) > 1e-20:
        fails.append(f"BL-A optim_D.eps={b['optim_D.eps']} (expected 1e-10)")
    if abs(b["optim_G.alpha"] - 0.9) > 1e-12:
        fails.append(f"BL-A optim_G.alpha={b['optim_G.alpha']} (expected 0.9)")
    if abs(b["optim_G.eps"] - 1e-10) > 1e-20:
        fails.append(f"BL-A optim_G.eps={b['optim_G.eps']} (expected 1e-10)")

    print()
    if fails:
        print("[FAIL]")
        for f in fails:
            print("  -", f)
        return 1
    print("[PASS] ADR-007 isolation verified:")
    print("  - S1D (ALOCC_LOSS, no flag)         -> alpha=0.99 eps=1e-8 (project-historical)")
    print("  - Baseline A (ALOCC, flag=True)     -> alpha=0.9  eps=1e-10 (TF1.15 verbatim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
