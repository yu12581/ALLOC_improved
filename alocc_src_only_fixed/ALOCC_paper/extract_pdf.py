import sys
from pypdf import PdfReader

src = sys.argv[1]
out = sys.argv[2]
r = PdfReader(src)
with open(out, "w", encoding="utf-8") as f:
    f.write(f"# Pages: {len(r.pages)}\n\n")
    for i, p in enumerate(r.pages):
        f.write(f"\n===== Page {i+1} =====\n")
        try:
            f.write(p.extract_text() or "")
        except Exception as e:
            f.write(f"[extract error: {e}]")
        f.write("\n")
print("OK", len(r.pages))
