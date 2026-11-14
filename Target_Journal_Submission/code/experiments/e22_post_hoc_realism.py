"""E22 (post hoc, deviation D27; designed after the E20 seed-0 results were known): two descriptive analyses.

1. PTB-XL by annotation type: for each type of expert-annotated noise (baseline drift, static noise, burst
   noise, electrode problems), the AUROC of 1 - max probability for separating records with that annotation
   from clean records, and the share of those records in the 20 % review queue; per method, per seed.
2. Effect of LOOM vs PreVal across the triage studies, ordered by realism of the faults: difference in the share
   of faulty recordings caught at a 20 % budget (LOOM - PreVal), with 95 % bootstrap CIs.
   E11/E13 (simulated faults) and E12 (recorded noise added): unit = dataset (mean over resamples), bootstrap
   over datasets. E20 (expert-annotated noise): unit = record, per seed; CI by bootstrap over records, seed 0.
No hypothesis is tested. Output: results/e22/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import ptbxl                                                     # noqa: E402

OUT = ROOT / "results" / "e22"
METHODS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"]
N_BOOT = 10_000


def queue(conf, b=0.2):
    k = int(round(b * len(conf))); q = np.zeros(len(conf), bool)
    q[np.argsort(conf, kind="stable")[:k]] = True
    return q


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def leads_in(text):
    """Leads named in a PTB-XL noise annotation: single leads, ranges such as 'I-AVF', or 'alles' (all leads).
    Returns None when the text names no recognisable lead (e.g. a bare comma)."""
    import re
    t = str(text).upper().replace("ALLES", "I-V6").replace("ALL", "I-V6")
    out = set()
    for tok in re.split(r"[,\s;]+", t):
        tok = tok.strip("?.: ")
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            if a in LEADS and b in LEADS and LEADS.index(a) <= LEADS.index(b):
                out.update(LEADS[LEADS.index(a):LEADS.index(b) + 1])
        elif tok in LEADS:
            out.add(tok)
    return out or None


def lead_coverage():
    """Number of leads named in the noise annotations of the noisy test records (union over annotation types)."""
    db = ptbxl.metadata(); X, y, noisy, fold, ids = ptbxl.load()
    meta = db.loc[ids[(fold >= 9) & noisy]]
    counts, unparsed = [], 0
    for _, r in meta.iterrows():
        u, any_text = set(), False
        for c in ptbxl.NOISE_COLS:
            if isinstance(r[c], str) and r[c].strip():
                any_text = True; L = leads_in(r[c])
                if L is None:
                    unparsed += 1; u = None; break
                u |= L
        if any_text and u is not None:
            counts.append(len(u))
    c = np.array(counts)
    return dict(n_records=int(len(meta)), n_parsed=int(len(c)), n_unparsed=int(unparsed), median_leads=float(np.median(c)),
                share_le3=float((c <= 3).mean()), share_all12=float((c == 12).mean()))


def by_type():
    from sklearn.metrics import roc_auc_score
    db = ptbxl.metadata()
    runs = {p.stem: dict(np.load(p)) for p in sorted((ROOT / "results" / "e20").glob("*_s[0-9].npz"))}
    out = {}
    for name, o in runs.items():
        meta = db.loc[o["ecg_id"]]; clean = ~o["noisy"].astype(bool)
        res = {}
        for c in ptbxl.NOISE_COLS:
            has = meta[c].fillna("").astype(str).str.strip().ne("").to_numpy()
            sel = clean | has
            res[c] = {"n": int(has.sum())}
            for m in METHODS:
                conf = o[f"P_{m}"].max(1)
                res[c][m] = dict(auroc=float(roc_auc_score(has[sel], 1 - conf[sel])),
                                 caught=float(queue(conf)[has].mean()))
        out[name] = res
    agg = {}
    for train in ("clean", "all"):
        keys = [k for k in out if k.startswith(train + "_")]
        if not keys:
            continue
        agg[train] = {"n_seeds": len(keys)}
        for c in ptbxl.NOISE_COLS:
            agg[train][c] = {"n": out[keys[0]][c]["n"]}
            for m in METHODS:
                agg[train][c][m] = {k: float(np.mean([out[s][c][m][k] for s in keys])) for k in ("auroc", "caught")}
    return {"per_run": out, "mean_over_seeds": agg}


def boot_mean(d, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def effects():
    rows = []
    for study, path, filt in (("E11", "case_study.json", None), ("E13", "case_study_confirm.json", None),
                              ("E12", "real_noise.json", "stream")):
        t = pd.DataFrame(json.load(open(ROOT / "results" / path)))
        if filt:
            t = t[t.cond == filt]
        else:
            t = t[t.stream == "stream"]
        key = "fault_caught@20"
        P = t[t.method.isin(["LOOM", "PreVal"])].groupby(["dataset", "method"])[key].mean().unstack()
        m, lo, hi = boot_mean(P["LOOM"] - P["PreVal"])
        rows.append(dict(study=study, unit="dataset", n=int(len(P)), diff=m, ci_low=lo, ci_high=hi,
                         loom=float(P["LOOM"].mean()), preval=float(P["PreVal"].mean())))
    runs = sorted((ROOT / "results" / "e20").glob("clean_s[0-9].npz"))
    per_seed = []
    for p in runs:
        o = dict(np.load(p)); nz = o["noisy"].astype(bool)
        per_seed.append(float(queue(o["P_LOOM"].max(1))[nz].mean() - queue(o["P_PreVal"].max(1))[nz].mean()))
    o = dict(np.load(runs[0])); nz = o["noisy"].astype(bool)
    dl = queue(o["P_LOOM"].max(1))[nz].astype(float) - queue(o["P_PreVal"].max(1))[nz].astype(float)
    _, lo, hi = boot_mean(dl)
    rows.append(dict(study="E20", unit="record (seed 0 CI)", n=int(nz.sum()), diff=float(np.mean(per_seed)),
                     diff_seed0=float(dl.mean()), ci_low=lo, ci_high=hi, per_seed=per_seed, n_seeds=len(per_seed)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = {"experiment": "E22 (post hoc, D27)", "by_type": by_type(), "effects": effects(), "lead_coverage": lead_coverage()}
    json.dump(s, open(OUT / "summary.json", "w"), indent=1)
    for c in ptbxl.NOISE_COLS:
        a = s["by_type"]["mean_over_seeds"]["clean"][c]
        print(c, a["n"], {m: round(a[m]["auroc"], 3) for m in METHODS})
    print("lead coverage", s["lead_coverage"])
    for r in s["effects"]:
        print(r["study"], r["n"], round(r["diff"], 3), round(r["ci_low"], 3), round(r["ci_high"], 3))


if __name__ == "__main__":
    main()
