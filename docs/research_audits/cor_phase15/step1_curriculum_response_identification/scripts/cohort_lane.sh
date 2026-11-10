#!/bin/bash
# Shared job queue for the independent cohort; run two copies (two lanes). Each job is claimed atomically with mkdir.
cd "$(dirname "$0")/.."; mkdir -p logs/claims
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for s in 60 61 62 63 64 65 66 67; do for a in T1 P; do
  mkdir logs/claims/${a}_s$s 2>/dev/null || continue
  echo "$(date -u) start $a s$s lane $$" >> logs/COMPUTE_LOG.txt
  python3 scripts/cohort_train.py $a $s > logs/train_${a}_s$s.out 2>&1
  echo "$(date -u) end $a s$s exit $?" >> logs/COMPUTE_LOG.txt
done; done
