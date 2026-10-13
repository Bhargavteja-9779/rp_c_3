#!/usr/bin/env python3
"""Build the JKSU-CIS manuscript (anonymised DOCX + PDF) from machine-generated results.

Every number in the text is produced by evaluating a {{ expression }} against
code/results/statistics.json, code/results/metrics.json and the CSV tables; a missing
value aborts the build (nothing is typed by hand). Tables are generated from the same
files. Citations ([@key]) are resolved by pandoc citeproc with the Springer Basic
(author-date) style from references.json.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUB = HERE.parents[1]                       # Target_Journal_Submission/
CODE = SUB / "code"
OUT = SUB / "manuscript"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else CODE   # root holding results/, tables/, figures/

S = json.load(open(RES / "results" / "statistics.json"))
M = json.load(open(RES / "results" / "metrics.json"))
T = {p.stem: p for p in (RES / "tables").glob("*.csv")}


# ------------------------------------------------------------------ formatting helpers
def f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("missing value in manuscript expression")
    return f"{x:.{nd}f}"


def f2(x): return f(x, 2)
def f1(x): return f(x, 1)
def pct(x, nd=1): return f"{100 * x:.{nd}f} %"


def pv(p):
    return "< 0.001" if p < 1e-3 else f"= {p:.3f}"


def pw(key, metric, comp):
    """pairwise record of LOOM vs competitor"""
    for r in S[key][metric]["pairwise"]:
        if r["competitor"] == comp:
            return r
    raise KeyError((key, metric, comp))


def med(key, metric, method):
    return S[key][metric]["medians"][method]


def rank(key, metric, method):
    return S[key][metric]["mean_ranks"][method]


def wdl(key, metric, comp):
    r = pw(key, metric, comp)
    return f"{r['ref_wins']}/{r['draws']}/{r['ref_losses']}"


def csv(name):
    return pd.read_csv(T[name])


def md_table(df, caption, label=None, colfmt=None):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + f"\n\n: {caption}\n"


def tab_compare(key, metrics, methods, caption, names=None, lower=None):
    """One row per method: median of each metric, mean rank (first metric), W/D/L and Holm p
    (first metric) of LOOM vs method."""
    lower = lower or {}
    m0 = metrics[0]
    rows = []
    for m in methods:
        if m not in S[key][m0]["medians"]:
            continue
        row = {"Method": (names or {}).get(m, m)}
        for mt in metrics:
            if mt in S[key] and m in S[key][mt]["medians"]:
                row[mt.upper() if mt != "shift_auroc" else "Shift AUROC"] = f(S[key][mt]["medians"][m])
        row["Mean rank"] = f2(S[key][m0]["mean_ranks"][m])
        if m == "LOOM":
            row["W/D/L"] = "–"; row["r_rb"] = "–"; row["p (Holm)"] = "–"
        else:
            r = pw(key, m0, m)
            row["W/D/L"] = f"{r['ref_wins']}/{r['draws']}/{r['ref_losses']}"
            row["r_rb"] = f2(r["rank_biserial"])
            row["p (Holm)"] = pv(r["p_holm"]).replace("= ", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    return md_table(df, caption)


CTX = {k: v for k, v in globals().items() if not k.startswith("__")}


def render(text):
    def repl(mo):
        expr = mo.group(1).strip()
        try:
            return str(eval(expr, CTX))
        except Exception as e:
            raise RuntimeError(f"manuscript expression failed: {{{{{expr}}}}}: {e!r}")
    return re.sub(r"\{\{(.+?)\}\}", repl, text, flags=re.S)


def build(anonymous=True):
    parts = []
    for name in ["front.md", "sections_static.md", "sections_results.md", "back.md"]:
        p = HERE / name
        if p.exists():
            parts.append(p.read_text())
    md = render("\n\n".join(parts))
    md = md.replace("](figures/", f"]({RES / 'figures'}/")
    (HERE / "manuscript_rendered.md").write_text(md)
    docx = OUT / "Final_Manuscript.docx"
    cmd = ["pandoc", str(HERE / "manuscript_rendered.md"), "-o", str(docx), "--citeproc",
           f"--bibliography={HERE / 'references.json'}", f"--csl={HERE / 'springer-basic-author-date.csl'}",
           f"--reference-doc={HERE / 'reference_springer.docx'}", "--resource-path", str(RES),
           "-M", "link-citations=false", "-M", "reference-section-title=References"]
    subprocess.run(cmd, check=True)
    postprocess(docx)
    print("written", docx)
    return docx


def postprocess(path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, Cm
    d = Document(path)
    # page numbers in the footer
    for s in d.sections:
        p = s.footer.paragraphs[0] if s.footer.paragraphs else s.footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for t, txt in (("begin", None), (None, " PAGE "), ("end", None)):
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r._r.append(e)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)
    # three-line tables, 8 pt, compact
    for tb in d.tables:
        tbl = tb._tbl
        pr = tbl.tblPr
        b = pr.find(qn("w:tblBorders"))
        if b is not None:
            pr.remove(b)
        b = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom"):
            e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000"); b.append(e)
        for edge in ("left", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); b.append(e)
        pr.append(b)
        for i, row in enumerate(tb.rows):
            for cell in row.cells:
                if i == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    bd = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom")
                    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000"); bd.append(e); tcPr.append(bd)
                for par in cell.paragraphs:
                    par.paragraph_format.space_after = Pt(0); par.paragraph_format.line_spacing = 1.0
                    for run in par.runs:
                        run.font.size = Pt(8); run.font.name = "Times New Roman"
                        if i == 0:
                            run.font.bold = True
    # figure paragraphs centred
    for par in d.paragraphs:
        if par._p.xpath(".//w:drawing"):
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.save(path)


def to_pdf(docx):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return docx.with_suffix(".pdf")


if __name__ == "__main__":
    dx = build()
    print("pdf:", to_pdf(dx))
