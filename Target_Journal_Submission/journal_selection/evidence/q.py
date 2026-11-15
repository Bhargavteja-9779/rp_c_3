import json, sys, time, urllib.request, urllib.parse
qs = ["time series classification uncertainty", "time series classification", "ECG signal quality", "ECG classification noise robust",
      "reject option classification", "selective prediction confidence", "calibration uncertainty classification", "MiniRocket ROCKET convolutional kernels",
      "sensor fault detection classification", "human-in-the-loop deferral expert", "out-of-distribution detection time series", "conformal prediction"]
out = {}
for q in qs:
    u = "https://api.crossref.org/journals/0957-4174/works?" + urllib.parse.urlencode({"query.bibliographic": q, "filter": "from-pub-date:2025-01-01,type:journal-article", "rows": 25, "select": "DOI,title,published,is-referenced-by-count,author"})
    r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script (mailto:none)"}), timeout=60))
    out[q] = [dict(doi=i["DOI"], title=i.get("title", [""])[0], year=i["published"]["date-parts"][0][0], cites=i.get("is-referenced-by-count", 0)) for i in r["message"]["items"]]
    time.sleep(1)
json.dump(out, open("crossref_eswa.json", "w"), indent=1)
for q, L in out.items():
    print("##", q)
    for i in L[:12]: print(f"  {i['year']} c{i['cites']:>3} {i['doi']} {i['title'][:120]}")
