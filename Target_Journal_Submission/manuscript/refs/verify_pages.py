"""Confirm non-DOI references on their official landing pages (title + an author surname)."""
import json, re, html, time, urllib.request
REFS = {
 "guo2017": ("https://proceedings.mlr.press/v70/guo17a.html", "On Calibration of Modern Neural Networks", "Pleiss"),
 "kull2019": ("https://arxiv.org/abs/1910.12656", "Beyond temperature scaling", "Kull"),
 "ovadia2019": ("https://arxiv.org/abs/1906.02530", "Can You Trust Your Model", "Ovadia"),
 "lakshminarayanan2017": ("https://arxiv.org/abs/1612.01474", "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", "Lakshminarayanan"),
 "liu2020": ("https://arxiv.org/abs/2006.10108", "Simple and Principled Uncertainty Estimation", "Liu"),
 "kristiadi2020": ("https://proceedings.mlr.press/v119/kristiadi20a.html", "Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks", "Kristiadi"),
 "vanamersfoort2020": ("https://proceedings.mlr.press/v119/van-amersfoort20a.html", "Uncertainty Estimation Using a Single Deep Deterministic Neural Network", "Amersfoort"),
 "demsar2006": ("https://www.jmlr.org/papers/v7/demsar06a.html", "Statistical Comparisons of Classifiers over Multiple Data Sets", "Dem"),
 "benavoli2016": ("https://www.jmlr.org/papers/v17/benavoli16a.html", "Should We Really Use Post-Hoc Tests Based on Mean-Ranks", "Benavoli"),
 "aeon2024": ("https://www.jmlr.org/papers/v25/23-1444.html", "aeon", "Middlehurst"),
 "sklearn2011": ("https://www.jmlr.org/papers/v12/pedregosa11a.html", "Scikit-learn: Machine Learning in Python", "Pedregosa"),
 "romano2020": ("https://arxiv.org/abs/2006.02544", "Classification with Valid and Adaptive Coverage", "Romano"),
 "hendrycks2019": ("https://arxiv.org/abs/1903.12261", "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations", "Hendrycks"),
 "hendrycks2017": ("https://arxiv.org/abs/1610.02136", "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", "Hendrycks"),
 "geifman2017": ("https://arxiv.org/abs/1705.08500", "Selective Classification for Deep Neural Networks", "Geifman"),
 "preval": ("https://arxiv.org/abs/2401.15610", "Prevalidated", "Dempster"),
 "angelopoulos2021": ("https://arxiv.org/abs/2009.14193", "Uncertainty Sets for Image Classifiers using Conformal Prediction", "Angelopoulos"),
 "liu2023jmlr": ("https://www.jmlr.org/papers/v24/22-0479.html", "Distance-Awareness", "Liu"),
}
res = {}
for k, (url, title, author) in REFS.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research reference check)"})
        t = html.unescape(urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore"))
        ok_t = title.lower() in re.sub(r"\s+", " ", t).lower(); ok_a = author.lower() in t.lower()
        m = re.search(r"<title>(.*?)</title>", t, re.S)
        extra = ""
        jr = re.search(r'(Journal of Machine Learning Research \d+ \(\d{4}\)[^<]{0,40}|Proceedings of the \d+\w+ International Conference on Machine Learning[^<]{0,80}|Comments:</td>\s*<td[^>]*>([^<]{0,160}))', t)
        if jr: extra = re.sub(r"\s+", " ", jr.group(0))[:170]
        res[k] = dict(url=url, title_found=ok_t, author_found=ok_a, page_title=re.sub(r"\s+", " ", m.group(1)).strip()[:150] if m else "", note=extra)
        print(f"{'OK ' if ok_t and ok_a else 'CHK'} {k:22s} {res[k]['page_title'][:95]} || {extra}")
    except Exception as e:
        res[k] = dict(url=url, error=str(e)); print("ERR", k, e)
    time.sleep(1)
json.dump(res, open("landing_pages_verified.json", "w"), indent=1)
