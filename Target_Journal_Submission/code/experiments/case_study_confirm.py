"""E13: confirmatory replication of the triage study on 38 further sensor datasets (Online Resource 2, 11.2).

Same protocol as experiments/case_study.py; datasets and fault sets fixed in the pre-registration.
Results: results/case_study_confirm.json and results/case_study/<dataset>__r<r>.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_study as cs                                                   # noqa: E402

PHYSIO = ["ECGFiveDays", "PigAirwayPressure", "PigArtPressure", "PigCVP", "EOGHorizontalSignal",
          "EOGVerticalSignal", "InsectEPGRegularTrain"]
MOTION = ["Haptics", "InlineSkate", "ToeSegmentation1", "ToeSegmentation2", "Worms", "WormsTwoClass"]
INDUSTRIAL = ["Car", "ChlorineConcentration", "CinCECGTorso", "Earthquakes", "FordA", "FordB", "InsectWingbeatSound",
              "ItalyPowerDemand", "Lightning2", "Lightning7", "MoteStrain", "Phoneme", "Plane",
              "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Trace",
              "ACSF1", "Computers", "HouseTwenty", "LargeKitchenAppliances", "ScreenType", "SmallKitchenAppliances",
              "Chinatown", "MelbournePedestrian"]
DOMAINS = {"physiological": (PHYSIO, cs.DOMAINS["cardiac"][1]),
           "motion": (MOTION, cs.DOMAINS["wearable"][1]),
           "industrial": (INDUSTRIAL, cs.DOMAINS["industrial"][1])}
assert len(PHYSIO + MOTION + INDUSTRIAL) == 38


def main():
    rows = []
    for dom, (dsets, faults) in DOMAINS.items():
        for d in dsets:
            for r in cs.RESAMPLES:
                o = cs.run(d, r, faults)
                y, is_f = o["y"], o["is_fault"].astype(bool)
                for m in cs.READOUTS + ["QUANT"]:
                    for stream in ("clean", "stream"):
                        key = f"P_{stream}_{m}"
                        if key not in o:
                            continue
                        P = o[key]; conf = P.max(1); corr = P.argmax(1) == y
                        rec = dict(domain=dom, dataset=d, resample=r, method=m, stream=stream,
                                   nll=cs.nll(P, y), acc=float(corr.mean()))
                        f = is_f if stream == "stream" else np.zeros(len(y), bool)
                        for b in cs.BUDGETS:
                            e, c = cs.triage(conf, corr, f, b)
                            rec[f"auto_err@{int(b * 100)}"] = e; rec[f"fault_caught@{int(b * 100)}"] = c
                        rows.append(rec)
                print(d, r, flush=True)
    json.dump(rows, open(cs.ROOT / "results" / "case_study_confirm.json", "w"), indent=0)


if __name__ == "__main__":
    main()
