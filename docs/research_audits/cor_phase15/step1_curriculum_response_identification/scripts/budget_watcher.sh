#!/bin/bash
# Quality-manipulation variants for the cohort (FROZEN_PREDICTIONS.md P2b/P6): when ft_T1_s{S}.npz exists, run the
# 250- and 500-step transfers from pre_T1_s{S}.npz (seed S).  Single thread.  Exits when all eight parents are done.
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
while true; do
  left=0
  for s in 60 61 62 63 64 65 66 67; do
    for b in 250 500; do
      [ -f checkpoints/budget/cT${s}_b$b.json ] && continue; left=1
      [ -f checkpoints/ft_T1_s$s.json ] || continue
      echo "$(date -u) start budget cT${s}_b$b" >> logs/COMPUTE_LOG.txt
      python3 scripts/transfer_budget.py cT${s}_b$b checkpoints/pre_T1_s$s.npz $s $b > logs/budget_cT${s}_b$b.out 2>&1
      echo "$(date -u) end budget cT${s}_b$b" >> logs/COMPUTE_LOG.txt
    done
  done
  [ $left -eq 0 ] && break; sleep 120
done
