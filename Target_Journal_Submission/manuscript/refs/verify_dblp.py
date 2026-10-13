"""Verify non-DOI references (proceedings, JMLR, arXiv) via the dblp search API."""
import json, time, urllib.request, urllib.parse
Q = {
 "guo2017": "On Calibration of Modern Neural Networks Guo Pleiss",
 "kull2019": "Beyond temperature scaling Dirichlet calibration Kull",
 "ovadia2019": "Can you trust your model's uncertainty evaluating predictive uncertainty under dataset shift",
 "lakshminarayanan2017": "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
 "liu2020": "Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness",
 "liu2023jmlr": "A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness",
 "kristiadi2020": "Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks",
 "demsar2006": "Statistical Comparisons of Classifiers over Multiple Data Sets Demsar",
 "benavoli2016": "Should We Really Use Post-Hoc Tests Based on Mean-Ranks",
 "romano2020": "Classification with Valid and Adaptive Coverage Romano Sesia",
 "hendrycks2019": "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
 "geifman2017": "Selective Classification for Deep Neural Networks Geifman",
 "hendrycks2017": "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks",
 "aeon2024": "aeon: a Python Toolkit for Learning from Time Series",
 "sklearn2011": "Scikit-learn: Machine Learning in Python Pedregosa",
 "preval": "Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression",
 "vanamersfoort2020": "Uncertainty Estimation Using a Single Deep Deterministic Neural Network",
 "angelopoulos2021raps": "Uncertainty Sets for Image Classifiers using Conformal Prediction",
 "rahimi2007": "Random Features for Large-Scale Kernel Machines",
}
out = {}
for k, q in Q.items():
    try:
        r = json.load(urllib.request.urlopen("https://dblp.uni-trier.de/search/publ/api?" + urllib.parse.urlencode({"q": q, "format": "json", "h": 5}), timeout=60))
        hits = r["result"]["hits"].get("hit", [])
        out[k] = [h["info"] for h in hits]
        for h in hits[:3]:
            i = h["info"]; a = i.get("authors", {}).get("author", [])
            a = [x["text"] if isinstance(x, dict) else x for x in (a if isinstance(a, list) else [a])]
            print(f"{k:22s} | {i.get('year')} | {i.get('venue')} {i.get('volume','')} {i.get('pages','')} | {i.get('title')[:85]} | {', '.join(a[:3])} | {i.get('doi','')} {i.get('ee','') if isinstance(i.get('ee'),str) else ''}")
    except Exception as e:
        print(k, "ERR", e)
    time.sleep(4)
json.dump(out, open("dblp_verified.json", "w"), indent=1)
