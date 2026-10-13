"""Phase-8 leakage audit: exact duplicate series within/across official UCR splits."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from loom.data.ucr import dataset_names, duplicate_audit
rows = [duplicate_audit(n) for n in dataset_names()]
df = pd.DataFrame(rows)
out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(exist_ok=True)
df.to_csv(out / "leakage_audit.csv", index=False)
flag = df[df.test_in_train > 0]
print(f"{len(df)} datasets; {len(flag)} with test series identical to a training series")
print(flag.to_string(index=False))
