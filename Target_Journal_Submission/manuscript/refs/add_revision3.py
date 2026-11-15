"""Add the references of the third revision to build/references.json (idempotent).

DOI entries are taken from Crossref (CSL-JSON). Silva et al. (2011) was checked on the Computing in Cardiology
archive PDF (vol. 38, pp. 273-276) and Gal & Ghahramani (2016) on its PMLR page. Log: refs/revision3_verified.json.
"""
import json, urllib.request
from pathlib import Path
B = Path(__file__).resolve().parents[1] / "build" / "references.json"
DOIS = {"clifford2012sqi": "10.1088/0967-3334/33/9/1419", "wang2017baseline": "10.1109/IJCNN.2017.7966039",
        "takara2025minirocket": "10.1016/j.eswa.2025.128662", "wang2025kernelgroup": "10.1016/j.eswa.2025.128693",
        "sensoy2025risk": "10.1016/j.eswa.2024.125906", "han2025evidential": "10.1016/j.eswa.2025.126533",
        "kinalioglu2027mmap": "10.1016/j.eswa.2026.133628", "munir2027ccaid": "10.1016/j.eswa.2026.134403",
        "wang2026wearable": "10.1016/j.eswa.2026.132601", "zhang2025llmtsfd": "10.1016/j.eswa.2024.125861"}
MANUAL = [
 {"id": "silva2011challenge", "type": "paper-conference",
  "title": "Improving the quality of ECGs collected using mobile phones: the PhysioNet/Computing in Cardiology Challenge 2011",
  "author": [{"family": "Silva", "given": "Ikaro"}, {"family": "Moody", "given": "George B."}, {"family": "Celi", "given": "Leo"}],
  "container-title": "Computing in Cardiology", "volume": "38", "page": "273-276", "issued": {"date-parts": [[2011]]},
  "_verified": "https://www.cinc.org/archives/2011/pdf/0273.pdf"},
 {"id": "gal2016dropout", "type": "paper-conference",
  "title": "Dropout as a Bayesian approximation: representing model uncertainty in deep learning",
  "author": [{"family": "Gal", "given": "Yarin"}, {"family": "Ghahramani", "given": "Zoubin"}],
  "container-title": "Proceedings of the 33rd International Conference on Machine Learning", "collection-title": "PMLR",
  "volume": "48", "page": "1050-1059", "issued": {"date-parts": [[2016]]},
  "_verified": "https://proceedings.mlr.press/v48/gal16.html"},
]
SENTENCE = {"wang2017baseline": "Time series classification from scratch with deep neural networks: a strong baseline"}
items = json.load(open(B)); have = {e["id"] for e in items}; log = []
for k, d in DOIS.items():
    if k in have: continue
    r = urllib.request.Request(f"https://api.crossref.org/works/{d}/transform/application/vnd.citationstyles.csl+json",
                               headers={"User-Agent": "research-agent"})
    it = json.load(urllib.request.urlopen(r, timeout=60))
    keep = {x: it[x] for x in ("type", "title", "author", "container-title", "volume", "issue", "page", "issued", "DOI", "publisher") if x in it}
    if isinstance(keep.get("container-title"), list): keep["container-title"] = keep["container-title"][0]
    if "page" not in keep and it.get("article-number"): keep["page"] = str(it["article-number"])
    if k in SENTENCE: keep["title"] = SENTENCE[k]
    keep["id"] = k; items.append(keep); log.append({"id": k, "doi": d, "source": "crossref", "title": keep["title"]})
for m in MANUAL:
    if m["id"] in have: continue
    m = dict(m); src = m.pop("_verified"); items.append(m); log.append({"id": m["id"], "source": src, "title": m["title"]})
json.dump(items, open(B, "w"), indent=1, ensure_ascii=False)
p = Path(__file__).with_name("revision3_verified.json")
old = json.load(open(p)) if p.exists() else []
json.dump(old + log, open(p, "w"), indent=1)
print(len(items), "references;", len(log), "added")
