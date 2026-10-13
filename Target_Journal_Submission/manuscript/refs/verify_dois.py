"""Verify bibliographic metadata of every DOI-bearing reference against Crossref."""
import json, sys, time, urllib.request, urllib.parse
DOIS = """10.1007/s10618-020-00701-z
10.1145/3447548.3467231
10.1007/s10618-022-00844-1
10.1007/s10618-023-00939-3
10.1007/s10618-024-01036-9
10.1007/s10994-021-06057-9
10.1007/s10618-024-01022-1
10.1109/JAS.2019.1911747
10.1007/s10618-016-0483-9
10.1007/s10618-019-00619-1
10.1007/s10618-020-00710-y
10.1145/1102351.1102430
10.1145/775047.775151
10.1007/s10994-023-06336-7
10.1162/neco.1992.4.5.720
10.1080/00401706.1970.10488634
10.1080/00401706.1979.10489751
10.1080/00401706.1974.10489157
10.1080/00031305.1978.10479237
10.1016/j.neunet.2004.07.002
10.1023/A:1018628609742
10.1007/978-3-031-06649-8
10.1214/20-AOS1965
10.1080/01621459.2017.1395341
10.1561/2200000101
10.1080/01621459.2017.1307116
10.2307/3001968
10.2466/11.IT.3.1
10.2202/1544-6115.1000
10.1007/s10618-020-00727-3
10.1371/journal.pone.0254841
10.1145/3136755.3136817
10.1145/3587937
10.1016/j.inffus.2021.05.008
10.1007/s10462-023-10562-9
10.1109/TPAMI.2023.3272339""".split()
out = {}
for d in DOIS:
    try:
        m = json.load(urllib.request.urlopen(f"https://api.crossref.org/works/{urllib.parse.quote(d)}?mailto=research.agent@example.org", timeout=60))["message"]
        auth = [f"{a.get('family','')} {''.join(x[0] for x in a.get('given','').replace('-',' ').split())}".strip() for a in m.get("author", [])]
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        out[d] = dict(title=m.get("title", [""])[0], authors=auth, year=yr, venue=(m.get("container-title") or [""])[0],
                      volume=m.get("volume"), issue=m.get("issue"), page=m.get("page"), article_number=m.get("article-number"),
                      publisher=m.get("publisher"), type=m.get("type"))
        print(f"OK {d} | {yr} | {', '.join(auth[:3])}{' et al' if len(auth)>3 else ''} | {out[d]['title'][:80]} | {out[d]['venue']} {m.get('volume')}:{m.get('page')}")
    except Exception as e:
        out[d] = dict(error=str(e)); print("FAIL", d, e)
    time.sleep(0.3)
json.dump(out, open("crossref_verified.json", "w"), indent=1)
