#!/bin/bash
cd "$(dirname "$0")/.."
for s in 0 1 2; do python3 scripts/train_curriculum.py $s 3000 checkpoints/posthoc_curriculum > logs/train/curr_$s.out 2>&1; done
for a in mlp_grouphist; do for t in T1 T4; do for s in 0 1 2; do python3 scripts/train_run.py $t M $s 6000 checkpoints/main 3 $a > logs/train/${t}_${a}_$s.out 2>&1; done; done; done
