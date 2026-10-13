"""Build references.json (CSL-JSON) from Crossref (DOI references) plus manually entered,
page-verified non-DOI references. Keys are used in the manuscript as [@key]."""
import json, time, urllib.request, urllib.parse
DOI = {
 "dempster2020rocket": "10.1007/s10618-020-00701-z", "dempster2021minirocket": "10.1145/3447548.3467231",
 "tan2022multirocket": "10.1007/s10618-022-00844-1", "dempster2023hydra": "10.1007/s10618-023-00939-3",
 "dempster2024quant": "10.1007/s10618-024-01036-9", "middlehurst2021hc2": "10.1007/s10994-021-06057-9",
 "middlehurst2024bakeoff": "10.1007/s10618-024-01022-1", "dau2019ucr": "10.1109/JAS.2019.1911747",
 "bagnall2017bakeoff": "10.1007/s10618-016-0483-9", "ismailfawaz2019review": "10.1007/s10618-019-00619-1",
 "ismailfawaz2020inception": "10.1007/s10618-020-00710-y", "niculescu2005": "10.1145/1102351.1102430",
 "zadrozny2002": "10.1145/775047.775151", "silvafilho2023": "10.1007/s10994-023-06336-7",
 "mackay1992": "10.1162/neco.1992.4.5.720", "hoerl1970": "10.1080/00401706.1970.10488634",
 "golub1979": "10.1080/00401706.1979.10489751", "allen1974": "10.1080/00401706.1974.10489157",
 "hoaglin1978": "10.1080/00031305.1978.10479237", "cawley2004": "10.1016/j.neunet.2004.07.002",
 "suykens1999": "10.1023/A:1018628609742", "vovk2022": "10.1007/978-3-031-06649-8",
 "barber2021": "10.1214/20-AOS1965", "sadinle2019": "10.1080/01621459.2017.1395341",
 "angelopoulos2023": "10.1561/2200000101", "lei2018": "10.1080/01621459.2017.1307116",
 "wilcoxon1945": "10.2307/3001968", "kerby2014": "10.2466/11.IT.3.1", "tibshirani2002": "10.2202/1544-6115.1000",
 "ruiz2021": "10.1007/s10618-020-00727-3", "iwana2021": "10.1371/journal.pone.0254841",
 "um2017": "10.1145/3136755.3136817", "ragab2023": "10.1145/3587937", "abdar2021": "10.1016/j.inffus.2021.05.008",
 "gawlikowski2023": "10.1007/s10462-023-10562-9", "xu2023": "10.1109/TPAMI.2023.3272339",
}
items = []
for k, d in DOI.items():
    url = f"https://api.crossref.org/works/{urllib.parse.quote(d)}/transform/application/vnd.citationstyles.csl+json"
    for a in range(4):
        try:
            it = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-agent (mailto:research.agent@example.org)"}), timeout=60)); break
        except Exception:
            time.sleep(3)
    keep = {x: it[x] for x in ("type", "title", "author", "editor", "container-title", "container-title-short", "volume", "issue", "page",
                                "issued", "DOI", "publisher", "publisher-place", "number", "event") if x in it}
    if isinstance(keep.get("title"), list): keep["title"] = keep["title"][0]
    if isinstance(keep.get("container-title"), list): keep["container-title"] = keep["container-title"][0] if keep["container-title"] else ""
    keep.pop("container-title-short", None)
    keep["id"] = k
    items.append(keep); time.sleep(0.2)
def A(*names):
    out = []
    for n in names:
        fam, giv = n.split(",")
        out.append({"family": fam.strip(), "given": giv.strip()})
    return out
MANUAL = [
 dict(id="guo2017", type="paper-conference", title="On calibration of modern neural networks", author=A("Guo, Chuan", "Pleiss, Geoff", "Sun, Yu", "Weinberger, Kilian Q."),
      **{"container-title": "Proceedings of the 34th International Conference on Machine Learning", "collection-title": "PMLR", "volume": "70", "page": "1321-1330", "issued": {"date-parts": [[2017]]}}),
 dict(id="platt1999", type="chapter", title="Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods",
      author=A("Platt, John C."), editor=A("Smola, Alexander J.", "Bartlett, Peter", "Schölkopf, Bernhard", "Schuurmans, Dale"),
      **{"container-title": "Advances in Large Margin Classifiers", "publisher": "MIT Press", "publisher-place": "Cambridge, MA", "page": "61-74", "issued": {"date-parts": [[2000]]}}),
 dict(id="kull2019", type="paper-conference", title="Beyond temperature scaling: obtaining well-calibrated multi-class probabilities with Dirichlet calibration",
      author=A("Kull, Meelis", "Perello-Nieto, Miquel", "Kängsepp, Markus", "Silva Filho, Telmo", "Song, Hao", "Flach, Peter"),
      **{"container-title": "Advances in Neural Information Processing Systems", "volume": "32", "issued": {"date-parts": [[2019]]}}),
 dict(id="ovadia2019", type="paper-conference", title="Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift",
      author=A("Ovadia, Yaniv", "Fertig, Emily", "Ren, Jie", "Nado, Zachary", "Sculley, D.", "Nowozin, Sebastian", "Dillon, Joshua V.", "Lakshminarayanan, Balaji", "Snoek, Jasper"),
      **{"container-title": "Advances in Neural Information Processing Systems", "volume": "32", "issued": {"date-parts": [[2019]]}}),
 dict(id="lakshminarayanan2017", type="paper-conference", title="Simple and scalable predictive uncertainty estimation using deep ensembles",
      author=A("Lakshminarayanan, Balaji", "Pritzel, Alexander", "Blundell, Charles"), **{"container-title": "Advances in Neural Information Processing Systems", "volume": "30", "issued": {"date-parts": [[2017]]}}),
 dict(id="liu2020sngp", type="paper-conference", title="Simple and principled uncertainty estimation with deterministic deep learning via distance awareness",
      author=A("Liu, Jeremiah Zhe", "Lin, Zi", "Padhy, Shreyas", "Tran, Dustin", "Bedrax-Weiss, Tania", "Lakshminarayanan, Balaji"),
      **{"container-title": "Advances in Neural Information Processing Systems", "volume": "33", "issued": {"date-parts": [[2020]]}}),
 dict(id="kristiadi2020", type="paper-conference", title="Being Bayesian, even just a bit, fixes overconfidence in ReLU networks",
      author=A("Kristiadi, Agustinus", "Hein, Matthias", "Hennig, Philipp"),
      **{"container-title": "Proceedings of the 37th International Conference on Machine Learning", "collection-title": "PMLR", "volume": "119", "page": "5436-5446", "issued": {"date-parts": [[2020]]}}),
 dict(id="vanamersfoort2020", type="paper-conference", title="Uncertainty estimation using a single deep deterministic neural network",
      author=A("van Amersfoort, Joost", "Smith, Lewis", "Teh, Yee Whye", "Gal, Yarin"),
      **{"container-title": "Proceedings of the 37th International Conference on Machine Learning", "collection-title": "PMLR", "volume": "119", "page": "9690-9700", "issued": {"date-parts": [[2020]]}}),
 dict(id="demsar2006", type="article-journal", title="Statistical comparisons of classifiers over multiple data sets", author=A("Demšar, Janez"),
      **{"container-title": "Journal of Machine Learning Research", "volume": "7", "page": "1-30", "issued": {"date-parts": [[2006]]}}),
 dict(id="benavoli2016", type="article-journal", title="Should we really use post-hoc tests based on mean-ranks?", author=A("Benavoli, Alessio", "Corani, Giorgio", "Mangili, Francesca"),
      **{"container-title": "Journal of Machine Learning Research", "volume": "17", "issue": "5", "page": "1-10", "issued": {"date-parts": [[2016]]}}),
 dict(id="holm1979", type="article-journal", title="A simple sequentially rejective multiple test procedure", author=A("Holm, Sture"),
      **{"container-title": "Scandinavian Journal of Statistics", "volume": "6", "issue": "2", "page": "65-70", "issued": {"date-parts": [[1979]]}}),
 dict(id="romano2020", type="paper-conference", title="Classification with valid and adaptive coverage", author=A("Romano, Yaniv", "Sesia, Matteo", "Candès, Emmanuel J."),
      **{"container-title": "Advances in Neural Information Processing Systems", "volume": "33", "issued": {"date-parts": [[2020]]}}),
 dict(id="hendrycks2019", type="paper-conference", title="Benchmarking neural network robustness to common corruptions and perturbations", author=A("Hendrycks, Dan", "Dietterich, Thomas"),
      **{"container-title": "International Conference on Learning Representations", "issued": {"date-parts": [[2019]]}}),
 dict(id="hendrycks2017", type="paper-conference", title="A baseline for detecting misclassified and out-of-distribution examples in neural networks", author=A("Hendrycks, Dan", "Gimpel, Kevin"),
      **{"container-title": "International Conference on Learning Representations", "issued": {"date-parts": [[2017]]}}),
 dict(id="geifman2017", type="paper-conference", title="Selective classification for deep neural networks", author=A("Geifman, Yonatan", "El-Yaniv, Ran"),
      **{"container-title": "Advances in Neural Information Processing Systems", "volume": "30", "issued": {"date-parts": [[2017]]}}),
 dict(id="aeon2024", type="article-journal", title="aeon: a Python toolkit for learning from time series",
      author=A("Middlehurst, Matthew", "Ismail-Fawaz, Ali", "Guillaume, Antoine", "Holder, Christopher", "Guijo-Rubio, David", "Bulatova, Guzal", "Tsaprounis, Leonidas", "Mentel, Lukasz", "Walter, Martin", "Schäfer, Patrick", "Bagnall, Anthony"),
      **{"container-title": "Journal of Machine Learning Research", "volume": "25", "issue": "289", "page": "1-10", "issued": {"date-parts": [[2024]]}}),
 dict(id="pedregosa2011", type="article-journal", title="Scikit-learn: machine learning in Python",
      author=A("Pedregosa, Fabian", "Varoquaux, Gaël", "Gramfort, Alexandre", "Michel, Vincent", "Thirion, Bertrand", "Grisel, Olivier", "Blondel, Mathieu", "Prettenhofer, Peter", "Weiss, Ron", "Dubourg, Vincent", "Vanderplas, Jake", "Passos, Alexandre", "Cournapeau, David", "Brucher, Matthieu", "Perrot, Matthieu", "Duchesnay, Édouard"),
      **{"container-title": "Journal of Machine Learning Research", "volume": "12", "page": "2825-2830", "issued": {"date-parts": [[2011]]}}),
 dict(id="dempster2025preval", type="article", title="Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data",
      author=A("Dempster, Angus", "Webb, Geoffrey I.", "Schmidt, Daniel F."), DOI="10.48550/arXiv.2401.15610",
      **{"container-title": "arXiv preprint arXiv:2401.15610 (version 2)", "issued": {"date-parts": [[2025]]}}),
 dict(id="liu2023sngp", type="article-journal", title="A simple approach to improve single-model deep uncertainty via distance-awareness",
      author=A("Liu, Jeremiah Zhe", "Padhy, Shreyas", "Ren, Jie", "Lin, Zi", "Wen, Yeming", "Jerfel, Ghassen", "Nado, Zachary", "Snoek, Jasper", "Tran, Dustin", "Lakshminarayanan, Balaji"),
      **{"container-title": "Journal of Machine Learning Research", "volume": "24", "issue": "42", "page": "1-63", "issued": {"date-parts": [[2023]]}}),
 dict(id="angelopoulos2021raps", type="paper-conference", title="Uncertainty sets for image classifiers using conformal prediction",
      author=A("Angelopoulos, Anastasios N.", "Bates, Stephen", "Malik, Jitendra", "Jordan, Michael I."),
      **{"container-title": "International Conference on Learning Representations", "issued": {"date-parts": [[2021]]}}),
]
items += MANUAL
json.dump(items, open("references.json", "w"), indent=1, ensure_ascii=False)
print(len(items), "references;", sum("DOI" in i for i in items), "with DOI")
for i in items:
    if not i.get("author") and i.get("type") != "book": print("check authors:", i["id"])
