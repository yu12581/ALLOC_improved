"""T2 rollback: restore the 4 files touched by t2_step1/step2 from *.t2_sn.bak.

Files covered:
  - model.py
  - mnist_experiment_runner.py
  - export_mnist_triplets.py
  - run_paper_mnist_figure6_7.py

Idempotent. Removes the *.t2_sn.bak file after restore.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
FILES = [
    ROOT / "model.py",
    ROOT / "mnist_experiment_runner.py",
    ROOT / "export_mnist_triplets.py",
    ROOT / "run_paper_mnist_figure6_7.py",
]
SENTINEL = "[T2-SN]"


def main() -> None:
    residues = []
    for f in FILES:
        bak = Path(str(f) + ".t2_sn.bak")
        if bak.exists():
            shutil.copyfile(bak, f)
            bak.unlink()
            print(f"[restore] {f.name} <- .t2_sn.bak (bak removed)")
        else:
            print(f"[skip]    {f.name} (no .t2_sn.bak)")
        src = f.read_text(encoding="utf-8")
        if SENTINEL in src:
            residues.append(f.name)
    if residues:
        print(f"[FAIL] sentinel still present in: {residues}")
        raise SystemExit(1)
    print("[ok] no T2-SN residue")


if __name__ == "__main__":
    main()
