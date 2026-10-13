#!/usr/bin/env bash
# Idempotent: (re)start the full benchmark unless it is already running; finished jobs are cached.
cd "$(dirname "$0")/.."
if pgrep -f "^python3 run_all.py --mode full" >/dev/null; then echo "already running"; exit 0; fi
setsid nohup python3 run_all.py --mode full --workers 4 --stages main,raw,extract,trainsize >> results/logs/full_stdout.log 2>&1 < /dev/null &
echo "started $(date -u)"
