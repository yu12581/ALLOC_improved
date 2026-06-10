"""PR-R: append `return int(epoch)` to ALOCC_LOSS_CLS._train.

Idempotent: runs detect-then-patch. Writes a backup next to the target.
"""
from pathlib import Path
import shutil
import sys

TARGET = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\model.py")
BACKUP = TARGET.with_suffix(".py.pr_r.bak")
SENTINEL_PREV = 'self._save_checkpoint(os.path.join(checkpoint_dir,f"{i+1}.pth"))'
RETURN_LINE = "        return int(epoch)\n"


def main() -> int:
    if not TARGET.exists():
        print(f"[PR-R] target missing: {TARGET}", file=sys.stderr)
        return 2

    text = TARGET.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    cls_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("class ALOCC_LOSS_CLS")),
        None,
    )
    if cls_idx is None:
        print("[PR-R] class ALOCC_LOSS_CLS not found", file=sys.stderr)
        return 3

    tail = lines[cls_idx:]
    if any("return int(epoch)" in ln for ln in tail):
        print("[PR-R] already patched (return int(epoch) exists in ALOCC_LOSS_CLS)")
        return 0

    last_sent_rel = None
    for i in range(len(tail) - 1, -1, -1):
        if SENTINEL_PREV in tail[i]:
            last_sent_rel = i
            break
    if last_sent_rel is None:
        print("[PR-R] sentinel line not found inside ALOCC_LOSS_CLS", file=sys.stderr)
        return 4

    insert_idx = cls_idx + last_sent_rel + 1

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"[PR-R] backup written: {BACKUP}")

    new_lines = lines[:insert_idx] + [RETURN_LINE] + lines[insert_idx:]
    TARGET.write_text("".join(new_lines), encoding="utf-8")
    print(f"[PR-R] inserted `return int(epoch)` at line {insert_idx + 1} (1-based)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
