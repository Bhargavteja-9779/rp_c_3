"""Empirical submission->acceptance (received->accepted) from Crossref assertions.
usage: python accept_time.py ISSN:Name ...  (articles published-online 2026-01-01..)"""
import sys, json, time, urllib.request, urllib.parse, statistics, datetime as dt
def parse(s):
    for f in ('%d %B %Y','%Y-%m-%d','%d %b %Y','%B %d, %Y'):
        try: return dt.datetime.strptime(s.strip(), f)
        except Exception: pass
    return None
def fetch(issn, maxn=1000):
    items, cursor = [], '*'
    while len(items) < maxn:
        q = urllib.parse.urlencode({'filter':'from-online-pub-date:2026-01-01,type:journal-article','rows':500,'cursor':cursor,'select':'DOI,assertion,title','mailto':'research.agent@example.org'})
        for attempt in range(4):
            try:
                d = json.load(urllib.request.urlopen(f'https://api.crossref.org/journals/{issn}/works?{q}', timeout=60))['message']; break
            except Exception as e:
                time.sleep(3*(attempt+1)); d=None
        if not d or not d['items']: break
        items += d['items']; cursor = d.get('next-cursor')
        if len(d['items']) < 500: break
    return items
out = {}
for arg in sys.argv[1:]:
    issn, name = arg.split(':',1)
    its = fetch(issn); days=[]
    for it in its:
        a = {x.get('name','').lower(): x.get('value','') for x in it.get('assertion',[])}
        r, c = parse(a.get('received','')), parse(a.get('accepted',''))
        if r and c and c>=r: days.append((c-r).days)
    if days:
        days.sort(); q = statistics.quantiles(days, n=4)
        out[name] = dict(issn=issn, n_articles=len(its), n_with_dates=len(days), median=statistics.median(days), q1=q[0], q3=q[2], pct_le_72=round(100*sum(d<=72 for d in days)/len(days),1))
    else:
        out[name] = dict(issn=issn, n_articles=len(its), n_with_dates=0)
    print(name, out[name], flush=True)
json.dump(out, open('accept_times.json','a'), indent=1)
