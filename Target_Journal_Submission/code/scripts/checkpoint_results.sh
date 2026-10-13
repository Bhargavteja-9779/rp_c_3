#!/usr/bin/env bash
# Archive cached job results (compressed) and push them, as insurance against container loss.
cd "$(dirname "$0")/.."
tar -czf results/raw_jobs.tar.gz -C results raw logs 2>/dev/null
cd /home/user/rp_c_3 && git add Target_Journal_Submission/code/results/raw_jobs.tar.gz && \
  git -c user.email=bhargavteja5098171@gmail.com -c user.name="bhargavteja-9779" commit -q -m "Checkpoint benchmark results ($(ls Target_Journal_Submission/code/results/raw | wc -l) jobs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N37jqkuXFqBzMgdEb5bNnp" && git push -q origin claude/great-dijkstra-zth7ja && echo "checkpoint pushed"
