#!/bin/bash
# Sequential queue for the training-objective site test (seeds 5, 6, 7). Run from the package directory.
cd "$(dirname "$0")/.."
for s in 5 6 7; do
  [ -f checkpoints/ft_pcsrc_s$s.json ] && continue
  echo "$(date -u) start pcsrc s$s" >> logs/COMPUTE_LOG.txt
  python3 scripts/site_train.py $s >> logs/site_lane.out 2>&1
  echo "$(date -u) end pcsrc s$s" >> logs/COMPUTE_LOG.txt
done
