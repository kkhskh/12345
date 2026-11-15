#!/bin/bash
# XIII-C recovery after a container restart: release the claims of jobs that logged "start" but no "end" (their process died),
# so that lanes can pick them up again.  Resumable jobs (ft_factorial) continue from their last checkpoint; idempotent jobs skip.
cd "$(dirname "$0")/.."
for k in $(grep -o "start \[[0-9a-f]*\]" logs/COMPUTE_LOG.txt | grep -o "[0-9a-f]\{12\}" | sort -u); do
  if ! grep -q "end \[$k\]" logs/COMPUTE_LOG.txt; then
    if ! pgrep -f "lane.sh" > /dev/null || [ "$1" == "force" ]; then rmdir logs/claims/$k 2>/dev/null && echo "$(date -u +%FT%TZ) RECOVER: released claim $k" >> logs/COMPUTE_LOG.txt; fi
  fi
done
