#!/usr/bin/env bash
# PTB-XL v1.0.3 (Wagner et al. 2020; PhysioNet, CC BY 4.0): metadata and the 100 Hz records used in E20
# (the 16,307 records with exactly one diagnostic superclass; about 450 MB). Idempotent: existing files are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/ptbxl && cd data/raw/ptbxl
BASE=https://physionet.org/files/ptb-xl/1.0.3
for f in ptbxl_database.csv scp_statements.csv LICENSE.txt; do [ -s "$f" ] || curl -sSfL --retry 5 -o "$f" "$BASE/$f"; done
python3 - <<'PY'
import sys; sys.path.insert(0, "../../../experiments")
import ptbxl
db = ptbxl.metadata()
open("files_needed.txt", "w").write("".join(f"{fn}.hea\n{fn}.dat\n" for fn in db["filename_lr"]))
print(len(db), "records")
PY
xargs -P 16 -I{} sh -c 'mkdir -p "$(dirname {})"; [ -s "{}" ] || curl -sSfL -m 120 --retry 5 -o "{}" "'"$BASE"'/{}" || echo "FAIL {}"' < files_needed.txt > dl.log 2>&1
echo "failed downloads: $(grep -c FAIL dl.log || true) (re-run this script to retry)"
