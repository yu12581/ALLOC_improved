"""A2 rollback: remove all S1-BOT (low-rank noisy bottleneck) code from ALOCC-master.

Reason: either RM-1 Round 1 fails qualitatively, or design needs a fresh restart.
All four touched files have `.s1_bot.bak` siblings -> restore is a pure copy.

Scope (surgical, only touches S1-BOT artefacts):
  1. Restore 4 files from *.s1_bot.bak
     (model.py, mnist_experiment_runner.py, export_mnist_triplets.py,
      run_paper_mnist_figure6_7.py)
  2. AST parse to confirm the restored files are valid Python
  3. Self-check: no "[S1-BOT]" / "LowRankNoisyBottleneck" / "bottleneck_rank"
     sentinel may remain anywhere in ALOCC-master *.py (excluding venv/__pycache__)
  4. Delete the 4 *.s1_bot.bak files once self-check passes
  5. Leave older *.rm1_l3.bak / *.pr_ab.bak / *.rm3*.bak UNTOUCHED

Idempotent: re-running after success is a no-op (backups gone -> early exit).
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")

BAK_PAIRS = [
    (ROOT / "model.py.s1_bot.bak",                     ROOT / "model.py"),
    (ROOT / "mnist_experiment_runner.py.s1_bot.bak",   ROOT / "mnist_experiment_runner.py"),
    (ROOT / "export_mnist_triplets.py.s1_bot.bak",     ROOT / "export_mnist_triplets.py"),
    (ROOT / "run_paper_mnist_figure6_7.py.s1_bot.bak", ROOT / "run_paper_mnist_figure6_7.py"),
]


def all_baks_missing() -> bool:
    return all(not bak.exists() for bak, _ in BAK_PAIRS)


def restore_from_bak() -> None:
    for bak, tgt in BAK_PAIRS:
        if not bak.exists():
            raise FileNotFoundError(f"missing backup: {bak}")
        shutil.copyfile(bak, tgt)
        print(f"[restore] {tgt.name}  <-  {bak.name}")


def syntax_check() -> None:
    import ast
    for _, tgt in BAK_PAIRS:
        ast.parse(tgt.read_text(encoding="utf-8"))
    print("[ok]      AST parse OK for all 4 restored files")


def self_check() -> None:
    leftover = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).parts
        if rel and rel[0] in {".venv", "venv", "__pycache__"}:
            continue
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "[S1-BOT]" in s or "LowRankNoisyBottleneck" in s or "bottleneck_rank" in s:
            leftover.append(p)
    if leftover:
        print("[FAIL] S1-BOT residue found in:")
        for p in leftover:
            print(f"       {p}")
        sys.exit(2)
    print("[ok]      no [S1-BOT] / LowRankNoisyBottleneck / bottleneck_rank in *.py")


def delete_bak_files() -> None:
    for bak, _ in BAK_PAIRS:
        if bak.exists():
            bak.unlink()
            print(f"[delete]  {bak.name}")


def main() -> None:
    if all_baks_missing():
        print("[skip] no .s1_bot.bak found; either not patched or already rolled back.")
        return
    restore_from_bak()
    syntax_check()
    self_check()
    delete_bak_files()
    print("\n[done] A2 (S1) rollback complete.")


if __name__ == "__main__":
    main()
