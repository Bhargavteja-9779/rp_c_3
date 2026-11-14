"""Add the references of the second revision to build/references.json (idempotent).

DOI entries are taken from Crossref (CSL-JSON); the two conference papers were checked on their official
proceedings pages (PMLR v119, NeurIPS 2018). Verification results: refs/revision2_verified.json.
"""
import json, urllib.request
from pathlib import Path
B = Path(__file__).resolve().parents[1] / "build" / "references.json"
DOIS = {"wagner2020ptbxl": "10.1038/s41597-020-0495-6", "strodthoff2021ptbxl": "10.1109/JBHI.2020.3022989",
        "mcnemar1947": "10.1007/BF02295996"}
MANUAL = [
 {"id": "mozannar2020", "type": "paper-conference", "title": "Consistent estimators for learning to defer to an expert",
  "author": [{"family": "Mozannar", "given": "Hussein"}, {"family": "Sontag", "given": "David"}],
  "container-title": "Proceedings of the 37th International Conference on Machine Learning", "collection-title": "PMLR",
  "volume": "119", "page": "7076-7087", "issued": {"date-parts": [[2020]]},
  "_verified": "https://proceedings.mlr.press/v119/mozannar20b.html"},
 {"id": "madras2018", "type": "paper-conference", "title": "Predict responsibly: improving fairness and accuracy by learning to defer",
  "author": [{"family": "Madras", "given": "David"}, {"family": "Pitassi", "given": "Toniann"}, {"family": "Zemel", "given": "Richard"}],
  "container-title": "Advances in Neural Information Processing Systems", "volume": "31", "issued": {"date-parts": [[2018]]},
  "_verified": "https://papers.nips.cc/paper_files/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html"},
]
items = json.load(open(B)); have = {e["id"] for e in items}; log = []
for k, d in DOIS.items():
    if k in have: continue
    r = urllib.request.Request(f"https://api.crossref.org/works/{d}/transform/application/vnd.citationstyles.csl+json",
                               headers={"User-Agent": "research-agent"})
    it = json.load(urllib.request.urlopen(r, timeout=60))
    keep = {x: it[x] for x in ("type", "title", "author", "container-title", "volume", "issue", "page", "issued", "DOI", "publisher") if x in it}
    if isinstance(keep.get("container-title"), list): keep["container-title"] = keep["container-title"][0]
    keep["id"] = k; items.append(keep); log.append({"id": k, "doi": d, "source": "crossref", "title": keep["title"]})
for m in MANUAL:
    if m["id"] in have: continue
    src = m.pop("_verified"); items.append(m); log.append({"id": m["id"], "source": src, "title": m["title"]})
json.dump(items, open(B, "w"), indent=1, ensure_ascii=False)
p = Path(__file__).with_name("revision2_verified.json")
old = json.load(open(p)) if p.exists() else []
json.dump(old + log, open(p, "w"), indent=1)
print(len(items), "references;", len(log), "added")
