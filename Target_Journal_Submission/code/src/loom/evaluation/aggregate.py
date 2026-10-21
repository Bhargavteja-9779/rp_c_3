"""Collect cached job JSONs into tidy tables."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_rows(raw_dir) -> pd.DataFrame:
    recs = []
    for f in sorted(glob.glob(str(Path(raw_dir) / "*.json"))):
        r = json.load(open(f))
        base = dict(dataset=r["dataset"], resample=r["resample"], transform=r["transform"], job=r["methods"],
                    train_frac=r.get("train_frac", 1.0), n_train=r["n_train"], n_test=r["n_test"], n_classes=r["n_classes"])
        for row in r["rows"]:
            recs.append({**base, **{k: v for k, v in row.items() if k != "rel"}})
    return pd.DataFrame(recs)


def load_reliability(raw_dir, transform=("minirocket", "raw")):
    """Pooled reliability bins: {(method, cond_kind): [count, sum_conf, sum_correct]} where
    cond_kind is 'clean' or the corruption family at severity 3."""
    pooled = {}
    for f in sorted(glob.glob(str(Path(raw_dir) / "*.json"))):
        r = json.load(open(f))
        if r["transform"] not in transform or r.get("train_frac", 1.0) < 1:
            continue
        for row in r["rows"]:
            if "rel" not in row:
                continue
            key = (row["method"], "clean" if row["cond"] == "clean" else ("shifted-sev3" if row["sev"] == 3 else None))
            if key[1] is None:
                continue
            arr = np.array(row["rel"], float)
            pooled[key] = pooled.get(key, 0) + arr
    return pooled


def load_timing(raw_dir) -> pd.DataFrame:
    recs = []
    for f in sorted(glob.glob(str(Path(raw_dir) / "*.json"))):
        r = json.load(open(f))
        for m, t in r["timing"].items():
            recs.append(dict(dataset=r["dataset"], resample=r["resample"], transform=r["transform"], job=r["methods"],
                             train_frac=r.get("train_frac", 1.0), n_train=r["n_train"], n_test=r["n_test"],
                             method=m, fit=t["fit"], predict_all=t["predict_all"],
                             feature_time=r["extra"].get("feature_time_train", np.nan),
                             svd_time=r["extra"].get("svd_time", np.nan)))
    return pd.DataFrame(recs)


def load_extra(raw_dir) -> pd.DataFrame:
    recs = []
    for f in sorted(glob.glob(str(Path(raw_dir) / "*.json"))):
        r = json.load(open(f))
        lo = r["extra"].get("loom")
        if lo:
            rec = dict(dataset=r["dataset"], resample=r["resample"], transform=r["transform"], job=r["methods"],
                       train_frac=r.get("train_frac", 1.0), lam=lo["lam"], a=lo["a"], h_ref=lo["h_ref"])
            for k, v in lo["h_test_median"].items():
                rec[f"h_{k}"] = v
            recs.append(rec)
        for m, e in (r["extra"].get("errors") or {}).items():
            recs.append(dict(dataset=r["dataset"], resample=r["resample"], transform=r["transform"], job=r["methods"], error_method=m, error=e[-300:]))
    return pd.DataFrame(recs)


def dataset_table(df, metric, cond="clean", sev=None, methods=None, agg_shift="mean"):
    """datasets x methods table of `metric`, averaged over resamples.
    cond='shifted' averages over all 18 corruption conditions first."""
    d = df.copy()
    if cond == "shifted":
        d = d[d.cond != "clean"]
    else:
        d = d[d.cond == cond]
        if sev is not None:
            d = d[d.sev == sev]
    if methods is not None:
        d = d[d.method.isin(methods)]
    t = d.groupby(["dataset", "method", "resample"])[metric].mean().groupby(["dataset", "method"]).mean().unstack("method")
    if methods is not None:
        t = t[[m for m in methods if m in t.columns]]
    return t
