#!/usr/bin/env python3
"""Convert research_design/deviations.md into a LaTeX longtable (deviations_table.tex) for the supplement."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
md = (HERE.parents[1] / "research_design" / "deviations.md").read_text().splitlines()
rows = [l for l in md if re.match(r"^\| D\d+", l)]


def tex(s):
    s = s.strip().replace("**", "")
    s = re.sub(r"`([^`]*)`", r"\\texttt{\1}", s)
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("~", "$\\sim$"), ("≤", "$\\le$"), ("×", "$\\times$"),
                 ("→", "$\\to$"), ("−", "--"), ("–", "--"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s.replace("\\texttt{", "\\texttt{").replace("10\\^{}13", "$10^{13}$")


out = ["\\begin{longtable}{P{0.5cm}P{1.4cm}P{3.3cm}P{2.6cm}P{1.5cm}P{2.5cm}}", "\\toprule",
       "ID & Date & Change & Reason & Timing & Effect \\\\", "\\midrule", "\\endhead"]
for r in rows:
    cells = [tex(c) for c in r.strip().strip("|").split("|")]
    out.append(" & ".join(cells) + " \\\\")
out += ["\\bottomrule", "\\end{longtable}"]
(HERE / "deviations_table.tex").write_text("\n".join(out) + "\n")
print(len(rows), "deviations")
