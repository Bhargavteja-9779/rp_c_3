#!/usr/bin/env python3
"""Editable Word version of the main manuscript (pandoc + pre/post-processing).

Pre-processing of main.tex: inline \\input tables, resolve every \\ref (main and supplement) from the .aux
files of the LaTeX build, number captions as "Fig. N" / "Table N", use PNG figures, and turn the
algorithm box into a numbered list. Pandoc expands the number macros and renders equations as native
Word equations; citations are formatted by citeproc with the Springer basic (author-date) style.
Post-processing: fixed table widths, compact table font, centred figures.
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "code"
BUILD = HERE.parent / "build"


def aux_labels(path):
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", path.read_text(errors="ignore")):
        labels[m.group(1)] = m.group(2)
    return labels


def main():
    tex = (HERE / "main.tex").read_text()
    lab = aux_labels(HERE / "main.aux")
    lab.update({"S-" + k: v for k, v in aux_labels(HERE / "supplement.aux").items()})
    # inline tables
    tex = re.sub(r"\\input\{(\.\./\.\./code/tables/tab_[^}]*)\}", lambda m: (HERE / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))).read_text(), tex)
    # numbered captions for figures and tables
    def number(env, prefix):
        nonlocal tex
        def rep(m):
            block = m.group(0)
            lm = re.search(r"\\label\{([^}]*)\}", block)
            n = lab.get(lm.group(1), "?") if lm else "?"
            return block.replace("\\caption{", f"\\caption{{\\textbf{{{prefix} {n}}} ", 1)
        tex = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", rep, tex, flags=re.S)
    number("figure", "Fig."); number("table", "Table")
    # references
    tex = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab.get(m.group(1), "??"), tex)
    tex = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + lab.get(m.group(1), "??") + ")", tex)
    tex = tex.replace("\\input{../../code/tables/macros.tex}", (CODE / "tables" / "macros.tex").read_text())
    tex = re.sub(r"\\includegraphics\[([^\]]*)\]\{([^}]*)\.pdf\}", lambda m: f"\\includegraphics[{m.group(1)}]{{{CODE}/figures/{m.group(2)}.png}}", tex)
    # algorithm box -> numbered steps
    def alg(m):
        body = m.group(0)
        cap = re.search(r"\\caption\{(.*?)\}\\label", body, re.S).group(1)
        steps = re.findall(r"\\(?:State|Require|For|EndFor)\b([^\n]*)", body)
        items = "\n".join(f"\\item {s.strip()}" for s in steps if s.strip())
        return f"\\paragraph{{Algorithm 1. {cap}}}\n\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}"
    tex = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", alg, tex, flags=re.S)
    tex = tex.replace("\\Comment{", "\\emph{")
    # constructs the pandoc LaTeX reader does not handle
    tex = re.sub(r"\\(begin|end)\{(figure|table)\*\}", r"\\\1{\2}", tex)
    tex = re.sub(r"\\shortstack\{([^}]*)\}", lambda m: m.group(1).replace("\\\\", " ").replace("-\\", ""), tex)
    tex = re.sub(r"P\{([0-9.]+cm)\}", r"p{\1}", tex)
    tex = re.sub(r"\\newcolumntype\{P\}.*\n", "", tex)
    for cmd in ("\\maketitle", "\\backmatter", "\\raggedbottom"):
        tex = tex.replace(cmd, "")
    flat = HERE / "_word_flat.tex"; flat.write_text(tex)
    out = HERE / "main_word.docx"
    subprocess.run(["pandoc", str(flat), "-o", str(out), "--citeproc", f"--bibliography={BUILD / 'references.json'}",
                    f"--csl={BUILD / 'springer-basic-author-date.csl'}", f"--reference-doc={BUILD / 'reference_springer.docx'}",
                    "--resource-path", str(HERE)], check=True)
    flat.unlink()
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    d = Document(out)
    for tb in d.tables:
        tb.autofit = False
        pr = tb._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); pr.append(lay)
        for row in tb.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.space_after = Pt(0)
                    for run in par.runs:
                        run.font.size = Pt(8)
    for par in d.paragraphs:
        if par._p.xpath(".//w:drawing"):
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.save(out)
    text = "\n".join(p.text for p in d.paragraphs)
    leaks = re.findall(r"\\[A-Za-z]{3,}|\?\?", text)
    print(f"written {out.name}; tables={len(d.tables)}; macro/?? leaks={len(leaks)} {leaks[:5]}")


if __name__ == "__main__":
    main()
