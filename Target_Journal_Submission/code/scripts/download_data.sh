#!/usr/bin/env bash
# Download and extract the UCR 2018 univariate archive (~550 MB zip, ~1.2 GB extracted).
set -euo pipefail
cd "$(dirname "$0")/../data"
mkdir -p raw && cd raw
URL=https://timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_arff.zip
for i in 1 2 3 4 5; do curl -L --retry 3 -C - -o Univariate2018_arff.zip "$URL" && break || sleep $((i*3)); done
unzip -q -o Univariate2018_arff.zip "*_TRAIN.txt" "*_TEST.txt" "*.csv" "*.xlsx"
cd .. && python3 ../scripts/build_dataset_list.py
