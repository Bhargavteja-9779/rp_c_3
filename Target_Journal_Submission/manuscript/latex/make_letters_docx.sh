#!/usr/bin/env bash
# Word versions of the cover letter, title page and highlights (pandoc expands the number macros).
set -euo pipefail
cd "$(dirname "$0")"
for f in cover_letter title_page; do pandoc "$f.tex" -f latex -o "$f.docx"; done
pandoc highlights.txt -f markdown -o highlights.docx
python3 - <<'PY'
import docx, re
for f in ("cover_letter.docx", "title_page.docx"):
    t = "\n".join(p.text for p in docx.Document(f).paragraphs)
    bad = re.findall(r"\\[A-Za-z]+", t)
    print(f, "macro leaks:", bad[:5] if bad else 0)
PY
