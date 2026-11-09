#!/bin/bash
cd "$(dirname "$0")/.."
run() { [ -f checkpoints/$1.json ] || python3 scripts/run_job3.py $1 > logs/train_$1.out 2>&1; echo "$(date +%T) $1 done" >> logs/progress.log; }
for j in trans_full_s0 gate_s1 trans_holdout_s1 adapt_s0 trans_aux_s2 gate_s2; do run $j; done
echo "$(date +%T) LANE1 DONE" >> logs/progress.log
