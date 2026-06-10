"""Generate a diff archive comparing the workspace shadow copy vs the live code.

Outputs ALOCC_paper/_patches/workspace_shadow_diff.md summarizing:
- Which .py files differ between the two trees (with line counts + MD5)
- Which files are unique to one side
- Per-file unified diff (truncated to 200 lines each) for divergent files

Purpose: archive what we lose before deleting the stale workspace shadow copy.
"""
from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

LIVE = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
SHADOW = Path(r"d:\codeVS\ALOCC_paper\ALLOC\ALOCC-master")
OUT = Path(r"d:\codeVS\ALOCC_paper\_patches\workspace_shadow_diff.md")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return -1


def list_py(root: Path) -> set[str]:
    return {
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
    }


def truncate_diff(live_text: str, shadow_text: str, max_lines: int = 200) -> str:
    diff = list(
        difflib.unified_diff(
            shadow_text.splitlines(keepends=False),
            live_text.splitlines(keepends=False),
            fromfile="shadow",
            tofile="live",
            lineterm="",
            n=2,
        )
    )
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... (truncated, total {len(diff)} lines) ..."]
    return "\n".join(diff)


def main() -> None:
    live_files = list_py(LIVE)
    shadow_files = list_py(SHADOW)

    only_live = sorted(live_files - shadow_files)
    only_shadow = sorted(shadow_files - live_files)
    common = sorted(live_files & shadow_files)

    identical: list[str] = []
    differ: list[tuple[str, str, str, int, int]] = []

    for rel in common:
        lp = LIVE / rel
        sp = SHADOW / rel
        lh, sh = md5(lp), md5(sp)
        if lh == sh:
            identical.append(rel)
        else:
            differ.append((rel, lh, sh, count_lines(lp), count_lines(sp)))

    lines: list[str] = []
    lines.append("# Workspace Shadow Copy — Diff Archive\n")
    lines.append(
        "Generated before deleting `d:\\codeVS\\ALOCC_paper\\ALLOC\\ALOCC-master\\` "
        "(the stale workspace shadow copy). Authoritative tree is "
        "`D:\\Trae_coding\\ALLOC\\ALOCC-master\\`.\n"
    )
    lines.append(f"- Python files in live tree:    **{len(live_files)}**")
    lines.append(f"- Python files in shadow tree:  **{len(shadow_files)}**")
    lines.append(f"- Identical (MD5 match):        **{len(identical)}**")
    lines.append(f"- Diverging (same name, diff):  **{len(differ)}**")
    lines.append(f"- Only in live (shadow missing): **{len(only_live)}**")
    lines.append(f"- Only in shadow (live missing): **{len(only_shadow)}**\n")

    if only_live:
        lines.append("## Files only in LIVE (never existed in shadow)\n")
        for f in only_live:
            lines.append(f"- `{f}`")
        lines.append("")
    if only_shadow:
        lines.append("## Files only in SHADOW (will be lost on delete)\n")
        for f in only_shadow:
            lp_size = (SHADOW / f).stat().st_size
            lines.append(f"- `{f}` ({lp_size} bytes)")
        lines.append("")

    if identical:
        lines.append(f"## Identical files ({len(identical)})\n")
        lines.append("<details><summary>click to expand</summary>\n")
        for f in identical:
            lines.append(f"- `{f}`")
        lines.append("\n</details>\n")

    if differ:
        lines.append(f"## Diverging files ({len(differ)})\n")
        lines.append("| File | live MD5 | shadow MD5 | live lines | shadow lines |")
        lines.append("|---|---|---|---:|---:|")
        for rel, lh, sh, ll, sl in differ:
            lines.append(
                f"| `{rel}` | `{lh[:8]}…` | `{sh[:8]}…` | {ll} | {sl} |"
            )
        lines.append("")

        lines.append("## Per-file unified diffs (truncated to 200 lines each)\n")
        for rel, *_ in differ:
            lines.append(f"### `{rel}`\n")
            lt = (LIVE / rel).read_text(encoding="utf-8", errors="replace")
            st = (SHADOW / rel).read_text(encoding="utf-8", errors="replace")
            lines.append("```diff")
            lines.append(truncate_diff(lt, st))
            lines.append("```\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        f"  identical={len(identical)}  differ={len(differ)}  "
        f"only_live={len(only_live)}  only_shadow={len(only_shadow)}"
    )


if __name__ == "__main__":
    main()
