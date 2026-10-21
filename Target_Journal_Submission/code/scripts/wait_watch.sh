#!/usr/bin/env bash
# Wait up to $1 seconds; kill the side-runner if memory nears the cgroup limit.
cd "$(dirname "$0")/.."
end=$((SECONDS+${1:-560})); peak=0
while [ $SECONDS -lt $end ] && pgrep -f "^python3 run_all.py --mode full" >/dev/null; do
  m=$(cat /sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes 2>/dev/null | head -1)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt 13600000000 ] && pgrep -f run_subset.py >/dev/null; then pkill -f run_subset.py; echo "watchdog: killed side runner at $m bytes"; fi
  sleep 15
done
echo "$(date -u +%H:%M) jobs=$(ls results/raw | wc -l) peakGB=$((peak/1000000000)) failed=$(grep -c FAILED results/logs/full_stdout.log)"
grep "^\[" results/logs/full_stdout.log | tail -1; grep -v -i warn results/logs/subset_ford.log | tail -1
