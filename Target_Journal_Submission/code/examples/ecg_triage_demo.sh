#!/usr/bin/env bash
# End-to-end demo of loom-triage on ECG recordings with real electrode noise (MIT-BIH NSTDB).
# Requires: pip install loom-tsc (or this repository), bash scripts/download_data.sh, bash scripts/download_nstdb.sh
set -e
cd "$(dirname "$0")/.."
out=examples/out && mkdir -p "$out"
python3 examples/make_demo_stream.py --dataset ECG5000 --out "$out"      # train.csv, stream.csv, truth.csv
loom-triage fit --train "$out/train.csv" --model "$out/model.pkl"
loom-triage run --model "$out/model.pkl" --input "$out/stream.csv" --budget 0.2 --alpha 0.1 \
                --out "$out/queue.csv" --report "$out/report.html"
python3 examples/make_demo_stream.py --score "$out"                       # how many noisy recordings were caught
