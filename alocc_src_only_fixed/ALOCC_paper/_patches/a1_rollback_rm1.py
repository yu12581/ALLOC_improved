"""A1 rollback: remove all RM1-L3 (MANet) code from ALOCC-master.

Reason: first author of the source paper conflicts with lab author circle;
full withdrawal required to avoid duplicate-publication risk.

Scope (surgical, only touches RM1-L3 artefacts):
  1. Restore 3 files from *.rm1_l3.bak  (model.py, mnist_experiment_runner.py,
     run_paper_mnist_figure6_7.py)
  2. Remove RM1-L3 additions from export_mnist_triplets.py (no .bak exists
     because step 3c skipped the backup -> reconstruct original signature)
  3. Delete self_attention.py
  4. Delete the 3 *.rm1_l3.bak files once restore is verified
  5. Leave older *.pr_ab.bak / *.pr_r.bak / *.rm3*.bak UNTOUCHED
  6. Self-check: no "RM1-L3" sentinel may remain anywhere in ALOCC-master
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")

BAK_PAIRS = [
    (ROOT / "model.py.rm1_l3.bak",                   ROOT / "model.py"),
    (ROOT / "mnist_experiment_runner.py.rm1_l3.bak", ROOT / "mnist_experiment_runner.py"),
    (ROOT / "run_paper_mnist_figure6_7.py.rm1_l3.bak", ROOT / "run_paper_mnist_figure6_7.py"),
]

SELF_ATTN = ROOT / "self_attention.py"
EXPORT    = ROOT / "export_mnist_triplets.py"


def restore_from_bak() -> None:
    for bak, tgt in BAK_PAIRS:
        if not bak.exists():
            raise FileNotFoundError(f"missing backup: {bak}")
        shutil.copyfile(bak, tgt)
        print(f"[restore] {tgt.name}  <-  {bak.name}")


def patch_export_mnist_triplets() -> None:
    """Revert build_model(variant, use_attention=False, attention_heads=4) to
    build_model(variant) by string replacement; strip all **kw usage.
    Idempotent: if no RM1-L3 tokens remain, return no-op."""
    txt = EXPORT.read_text(encoding="utf-8")
    if "RM1-L3" not in txt and "use_attention" not in txt and "attention_heads" not in txt and "**kw" not in txt:
        print(f"[skip]    {EXPORT.name}  (already rolled back)")
        return
    before = txt

    repls = [
        (
            "def build_model(variant: str, use_attention: bool = False, "
            "attention_heads: int = 4):  # [RM1-L3]\n"
            "    kw = {\"use_attention\": use_attention, "
            "\"attention_heads\": attention_heads}  # [RM1-L3]\n",
            "def build_model(variant: str):\n",
        ),
        ("ALOCC(in_h=28, out_h=28, **kw)",                       "ALOCC(in_h=28, out_h=28)"),
        ("ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, **kw)",   "ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8)"),
        ("ALOCC_LOSS(in_h=28, out_h=28, **kw)",                  "ALOCC_LOSS(in_h=28, out_h=28)"),
        ("ALOCC_LOSS_CLS(in_h=28, out_h=28, classify=True, **kw)", "ALOCC_LOSS_CLS(in_h=28, out_h=28, classify=True)"),
    ]
    for old, new in repls:
        if old not in txt:
            raise RuntimeError(f"rollback target not found in export_mnist_triplets.py:\n{old!r}")
        txt = txt.replace(old, new, 1)

    if "RM1-L3" in txt or "use_attention" in txt or "attention_heads" in txt:
        raise RuntimeError("stray RM1-L3 tokens remain after export_mnist_triplets rollback")

    if txt != before:
        EXPORT.write_text(txt, encoding="utf-8")
        print(f"[patch]   {EXPORT.name}  (surgical strip of 5 RM1-L3 sites)")


def delete_self_attention() -> None:
    if SELF_ATTN.exists():
        SELF_ATTN.unlink()
        print(f"[delete]  {SELF_ATTN.name}")


def delete_bak_files() -> None:
    for bak, _ in BAK_PAIRS:
        if bak.exists():
            bak.unlink()
            print(f"[delete]  {bak.name}")


def self_check() -> None:
    leftover = []
    for p in ROOT.rglob("*.py"):
        # only scan the project source, not the bundled venv / site-packages
        rel = p.relative_to(ROOT).parts
        if rel and rel[0] in {".venv", "venv", "__pycache__"}:
            continue
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "RM1-L3" in s or "SelfAttention2d" in s or "use_balance_loss" in s:
            leftover.append(p)
    if leftover:
        print("[FAIL] RM1-L3 residue found in:")
        for p in leftover:
            print(f"       {p}")
        sys.exit(2)
    print("[ok]      no RM1-L3 / SelfAttention2d / use_attention / use_balance_loss in *.py")


def syntax_check() -> None:
    import ast
    for _, tgt in BAK_PAIRS:
        ast.parse(tgt.read_text(encoding="utf-8"))
    ast.parse(EXPORT.read_text(encoding="utf-8"))
    print("[ok]      AST parse OK for all 4 restored files")


def main() -> None:
    restore_from_bak()
    patch_export_mnist_triplets()
    delete_self_attention()
    syntax_check()
    self_check()
    delete_bak_files()
    print("\n[done] A1 rollback complete.")


if __name__ == "__main__":
    main()
