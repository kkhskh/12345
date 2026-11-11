#!/bin/bash
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; mkdir -p logs/claims
for s in 90 91 92 93 94 95 96 97; do mkdir logs/claims/pre$s 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) lane $1 start pretrain P$s" >> logs/COMPUTE_LOG.txt
  python3 scripts/cohort7_pretrain.py $s > logs/pretrain_P$s.out 2>&1; echo "$(date -u +%FT%TZ) lane $1 end pretrain P$s rc=$?" >> logs/COMPUTE_LOG.txt; done
