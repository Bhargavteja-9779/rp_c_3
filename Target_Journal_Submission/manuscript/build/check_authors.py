import json, re, html, time, urllib.request, unicodedata
pages = json.load(open("../refs/landing_pages_verified.json"))
key_map = {"guo2017": "guo2017", "kull2019": "kull2019", "ovadia2019": "ovadia2019", "lakshminarayanan2017": "lakshminarayanan2017",
           "liu2020sngp": "liu2020", "kristiadi2020": "kristiadi2020", "vanamersfoort2020": "vanamersfoort2020", "demsar2006": "demsar2006",
           "benavoli2016": "benavoli2016", "romano2020": "romano2020", "hendrycks2019": "hendrycks2019", "hendrycks2017": "hendrycks2017",
           "geifman2017": "geifman2017", "aeon2024": "aeon2024", "pedregosa2011": "sklearn2011", "dempster2025preval": "preval",
           "liu2023sngp": "liu2023jmlr", "angelopoulos2021raps": "angelopoulos2021"}
norm = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ").strip()
refs = {r["id"]: r for r in json.load(open("references.json"))}
for k, pk in key_map.items():
    url = pages[pk]["url"]
    t = html.unescape(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8", "ignore"))
    found = re.findall(r'<meta name="citation_author" content="([^"]+)"', t)
    fams = []
    for a in found:
        fams.append(norm(a.split(",")[0]) if "," in a else norm(a.split()[-1]))
    mine = [norm(a["family"]) for a in refs[k]["author"]]
    ok = [m.split()[-1] for m in mine] == [f.split()[-1] for f in fams]
    print(("OK " if ok else "MISMATCH ") + k, "| page:", found if not ok else len(found))
    time.sleep(1)
