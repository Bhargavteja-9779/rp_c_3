"""Create reference_springer.docx (pandoc template): Times New Roman 10 pt, A4, decimal headings."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
d = Document("ref_default.docx")
S = {s.name: s for s in d.styles}

def setfont(st, size=10, bold=None, italic=None):
    st.font.name = "Times New Roman"; st.font.size = Pt(size); st.font.color.rgb = RGBColor(0, 0, 0)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    if bold is not None: st.font.bold = bold
    if italic is not None: st.font.italic = italic

for nm in ["Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Bibliography", "Footnote Text",
           "Image Caption", "Table Caption", "Caption", "Abstract", "Author", "Date", "Subtitle", "Abstract Title", "Figure", "Captioned Figure"]:
    if nm in S: setfont(S[nm], 10)
setfont(S["Title"], 14, bold=True)
S["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
for i, sz in [(1, 12), (2, 11), (3, 10)]:
    st = S[f"Heading {i}"]; setfont(st, sz, bold=True, italic=(i == 3))
    st.paragraph_format.space_before = Pt(12 if i == 1 else 8); st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.keep_with_next = True
for nm in ["Body Text", "First Paragraph"]:
    pf = S[nm].paragraph_format; pf.line_spacing = 1.5; pf.space_after = Pt(4); pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
S["Compact"].paragraph_format.line_spacing = 1.0
for nm in ["Image Caption", "Table Caption", "Caption"]:
    pf = S[nm].paragraph_format; pf.alignment = WD_ALIGN_PARAGRAPH.LEFT; S[nm].font.italic = False; pf.space_after = Pt(8)
S["Abstract"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf = S["Bibliography"].paragraph_format; pf.left_indent = Cm(0.5); pf.first_line_indent = Cm(-0.5); pf.line_spacing = 1.15; pf.space_after = Pt(3)
for s in d.sections:
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.5)
d.save("reference_springer.docx"); print("reference_springer.docx written")
