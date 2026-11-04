#!/usr/bin/env python3
"""Convert the verified CSL-JSON bibliography (../build/references.json) into references.bib.

Entry types are mapped explicitly; acronyms and proper nouns in titles are protected with braces so that
the sentence-case conversion of the Springer Nature style keeps them (e.g. {ROCKET}, {MiniRocket}, {Bayesian}).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPER = {"Bayesian", "Bayes", "Laplace", "Gaussian", "Python", "ImageNet", "Parkinson's", "Parkinson’s", "AlexNet",
          "PhysioBank", "PhysioToolkit", "PhysioNet", "ReLU", "ANOVA", "Bake", "UCR", "Dirichlet", "Hessian", "Sherman", "Morrison"}
TYPE = {"journal-article": "article", "article-journal": "article", "paper-conference": "inproceedings",
        "proceedings-article": "inproceedings", "chapter": "incollection", "book": "book", "article": "misc"}


def protect(title):
    out = []
    for w in title.split(" "):
        core = re.sub(r"[^\w’'-]", "", w)
        if core and (any(c.isupper() for c in core[1:]) or core in PROPER or (core.isupper() and len(core) > 1)
                     or re.match(r"^[A-Z]+[a-z]*\+$", w)):
            out.append("{" + w + "}")
        else:
            out.append(w)
    return " ".join(out)


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def main():
    refs = json.load(open(HERE.parent / "build" / "references.json"))
    out = []
    for r in refs:
        t = TYPE[r["type"]]
        f = {}
        f["author"] = " and ".join(f"{a['family']}, {a.get('given', '')}".strip(", ") if "family" in a else "{" + a["literal"] + "}"
                                   for a in r.get("author", []))
        f["title"] = protect(esc(r["title"]))
        f["year"] = str(r["issued"]["date-parts"][0][0])
        ct = esc(r.get("container-title", ""))
        if t == "article":
            f["journal"] = ct
        elif t in ("inproceedings", "incollection"):
            f["booktitle"] = ct
        elif t == "misc" and ct:
            f["howpublished"] = ct
        for k in ("volume", "page", "publisher", "DOI", "URL"):
            if r.get(k):
                f[{"page": "pages", "DOI": "doi", "URL": "url"}.get(k, k)] = esc(str(r[k])).replace("-", "--") if k == "page" else str(r[k])
        if r.get("issue"):
            f["number"] = str(r["issue"])
        if "editor" in r:
            f["editor"] = " and ".join(f"{a['family']}, {a.get('given', '')}" for a in r["editor"])
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)
        out.append(f"@{t}{{{r['id']},\n{body}\n}}")
    (HERE / "references.bib").write_text("\n\n".join(out) + "\n")
    print(len(out), "entries")


if __name__ == "__main__":
    main()
