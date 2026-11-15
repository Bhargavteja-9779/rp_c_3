#!/usr/bin/env bash
# PhysioNet/Computing in Cardiology Challenge 2011, "Improving the quality of ECGs collected using mobile phones"
# (Silva, Moody & Celi 2011; PhysioNet, Open Data Commons Attribution License v1.0). Set A: 12-lead, 500 Hz, 10 s
# records with the reference quality groups (acceptable / unacceptable / indeterminate). About 108 MB. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/cinc2011 && cd data/raw/cinc2011
BASE=https://physionet.org/files/challenge-2011/1.0.0
[ -s set-a.tar.gz ] || curl -sSfL --retry 5 -o set-a.tar.gz "$BASE/set-a.tar.gz"
sha256sum set-a.tar.gz > set-a.sha256
[ -d set-a ] || tar -xzf set-a.tar.gz
ls set-a | head -3; ls set-a | wc -l
