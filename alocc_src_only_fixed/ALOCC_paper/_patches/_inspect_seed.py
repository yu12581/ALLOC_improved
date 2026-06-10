"""Quick inspector for seed-related markers in the patched ALOCC source tree."""
from __future__ import annotations
import pathlib

BASE = pathlib.Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
FILES = [
    "utils.py",
    "mnist_experiment_runner.py",
    "run_paper_mnist_figure6_7.py",
    "export_mnist_triplets.py",
]
NEEDLES = ("A4-SEED", "set_random_seed", "_CURRENT_SEED", "--seed", "args.seed")

for f in FILES:
    p = BASE / f
    text = p.read_text(encoding="utf-8")
    print(f"=== {f} ===")
    for i, line in enumerate(text.splitlines(), 1):
        if any(n in line for n in NEEDLES):
            print(f"{i:4}: {line.rstrip()}")
    print()
