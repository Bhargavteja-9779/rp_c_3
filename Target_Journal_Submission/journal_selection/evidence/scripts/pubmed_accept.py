"""Empirical received->accepted days from PubMed history dates.
usage: python pubmed_accept.py "Journal TA":Name ... (publication date 2025/07-2026)"""
import sys, time, urllib.request, urllib.parse, statistics, datetime as dt, json, xml.etree.ElementTree as ET
E='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
def get(url):
    for a in range(5):
        try: return urllib.request.urlopen(url, timeout=90).read()
        except Exception: time.sleep(2+3*a)
    return b''
res={}
for arg in sys.argv[1:]:
    ta,name=arg.rsplit(':',1)
    term=f'"{ta}"[ta] AND ("2025/07/01"[dp] : "2026/12/31"[dp]) AND journal article[pt]'
    q=urllib.parse.urlencode({'db':'pubmed','term':term,'retmax':600,'retmode':'json','sort':'pub_date'})
    ids=json.loads(get(E+'esearch.fcgi?'+q))['esearchresult']['idlist']
    days=[]
    for i in range(0,len(ids),200):
        x=get(E+'efetch.fcgi?'+urllib.parse.urlencode({'db':'pubmed','id':','.join(ids[i:i+200]),'retmode':'xml'}))
        root=ET.fromstring(x)
        for art in root.findall('.//PubmedArticle'):
            h={}
            for p in art.findall('.//History/PubMedPubDate'):
                try: h[p.get('PubStatus')]=dt.date(int(p.findtext('Year')),int(p.findtext('Month')),int(p.findtext('Day')))
                except Exception: pass
            if 'received' in h and 'accepted' in h and h['accepted']>=h['received']:
                days.append((h['accepted']-h['received']).days)
        time.sleep(0.4)
    if days:
        qs=statistics.quantiles(days,n=4) if len(days)>3 else [0,0,0]
        res[name]=dict(n_ids=len(ids),n_dates=len(days),median=statistics.median(days),q1=qs[0],q3=qs[2],pct_le_72=round(100*sum(d<=72 for d in days)/len(days),1))
    else: res[name]=dict(n_ids=len(ids),n_dates=0)
    print(name,res[name],flush=True); time.sleep(0.5)
