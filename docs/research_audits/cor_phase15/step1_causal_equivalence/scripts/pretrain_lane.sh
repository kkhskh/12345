#!/bin/bash
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; mkdir -p logs/claims
for s in 140 141 142 143; do mkdir logs/claims/preP$s 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) lane $1 start pretrain P$s" >> logs/COMPUTE_LOG.txt; python3 scripts/cohort10_pretrain.py P $s > logs/pretrain_P$s.out 2>&1
  echo "$(date -u +%FT%TZ) lane $1 end pretrain P$s rc=$?" >> logs/COMPUTE_LOG.txt; done
