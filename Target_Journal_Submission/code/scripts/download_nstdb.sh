#!/usr/bin/env bash
# MIT-BIH Noise Stress Test Database (Moody, Muldrow & Mark 1984; PhysioNet, Goldberger et al. 2000).
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw/nstdb && cd data/raw/nstdb
for r in bw em ma; do for e in hea dat; do curl -sSfL -O "https://physionet.org/files/nstdb/1.0.0/$r.$e"; done; done
echo "NSTDB noise records in $(pwd)"
