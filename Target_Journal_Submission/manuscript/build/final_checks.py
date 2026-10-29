#!/usr/bin/env python3
"""Automated consistency checks on the final build; prints PASS/FAIL lines (used for REVIEW/final_audit.md)."""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = ROOT / "code"
md = (HERE / "manuscript_rendered.md").read_text()
body = md.split("# References")[0]
res = []


def check(name, ok, detail=""):
    res.append((name, bool(ok), detail))


check("no unresolved expressions", "MISSING" not in md and "{{" not in md)
ab = re.search(r"\*\*Abstract\*\* (.*)", md).group(1)
check("abstract 150-250 words", 150 <= len(ab.split()) <= 250, f"{len(ab.split())} words")
kw = re.search(r"\*\*Keywords\*\* (.*)", md).group(1).split("·")
check("4-6 keywords", 4 <= len(kw) <= 6, f"{len(kw)}")
hl = [l[2:] for l in (HERE / "narr_highlights.md").read_text().splitlines() if l.startswith("- ")]
check("highlights 3-5, <= 85 characters", 3 <= len(hl) <= 5 and all(len(h) <= 85 for h in hl), f"{[len(h) for h in hl]}")
first = {}
for m in re.finditer(r"Fig\. (\d+)", body):
    first.setdefault(int(m.group(1)), m.start())
order = [k for k, _ in sorted(first.items(), key=lambda x: x[1])]
check("figures cited in order", order == sorted(order), f"{order}")
caps = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\*\*", body)]
check("every figure has a caption and is cited", sorted(caps) == sorted(first), f"captions {caps}")
tabs = [int(x) for x in re.findall(r"\*\*Table (\d+)\*\*", body)]
tfirst = {}
for m in re.finditer(r"Table (\d+)", body):
    tfirst.setdefault(int(m.group(1)), m.start())
check("tables cited in order", [k for k, _ in sorted(tfirst.items(), key=lambda x: x[1])] == sorted(tfirst), f"{sorted(tfirst)}")
heads = re.findall(r"^## (\d\.\d) ", body, flags=re.M)
refs_sec = set(re.findall(r"Section (\d\.\d)", body))
check("cross-referenced sections exist", refs_sec <= set(heads), f"{sorted(refs_sec - set(heads))}")
refs = {r["id"] for r in json.load(open(HERE / "references.json"))}
cited = {c for g in re.findall(r"\[(@[^\]]+)\]", md) for c in re.findall(r"@([\w:-]+)", g)}
cited |= set(re.findall(r"(?<![\w\[])@([A-Za-z][\w-]*\d{4}\w*)", md))
check("all citations in bibliography", cited <= refs, f"{sorted(cited - refs)}")
check("all references cited", refs <= cited, f"{sorted(refs - cited)}")
for f in ["Final_Manuscript.docx", "Final_Manuscript.pdf", "Title_Page.docx", "Cover_Letter.docx",
          "alternative_ESWA/Manuscript_ESWA.docx"]:
    check(f"exists: manuscript/{f}", (ROOT / "manuscript" / f).exists())
from docx import Document  # noqa: E402
cp = Document(ROOT / "manuscript" / "Final_Manuscript.docx").core_properties
check("anonymous DOCX metadata", not cp.author and not cp.last_modified_by, f"author={cp.author!r}")
check("no author-identifying strings", not re.search(r"bhargav|@gmail", md, re.I))
for f in ["Online_Resource_1.pdf", "Online_Resource_2.pdf", "Online_Resource_3.xlsx"]:
    check(f"exists: {f}", (ROOT / "supplementary" / "online_resources" / f).exists())
S = json.load(open(CODE / "results" / "statistics.json"))
check("all 114 datasets analysed", S["n_datasets"] == 114, f"{S['n_datasets']}")
need = ["E1_clean", "E4_shifted", "E2b_raw_clean", "E2b_raw_shifted", "E3_ablation_clean", "E3_ablation_shifted",
        "E5_multirocket_clean", "E5_hydra_clean", "E7_train_size", "E9_conformal", "E10_natural_shift", "E11_case_study"]
check("all experiments present in statistics.json", all(k in S for k in need), f"{[k for k in need if k not in S]}")
n_q = S["E2b_raw_clean"]["nll"]
qn = [r["n"] for r in n_q["pairwise"] if r["competitor"] in ("QUANT", "Arsenal")]
check("QUANT on all 114 datasets; Arsenal on 113 (NaN probabilities on Fungi, disclosed in 5.2)",
      qn == [114, 113] and "Fungi" in md, f"n={qn}")
t = subprocess.run(["python3", "-m", "pytest", "-q", "tests"], cwd=CODE, capture_output=True, text=True)
check("unit tests pass", t.returncode == 0, t.stdout.strip().splitlines()[-1] if t.stdout else t.stderr[-200:])
for name, ok, d in res:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {d}")
print(f"\n{sum(ok for _, ok, _ in res)}/{len(res)} checks passed")
