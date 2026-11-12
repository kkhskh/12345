#!/bin/bash
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; mkdir -p logs/claims
for job in P:100 P:101 P:102 P:103 P:104 P:105 P:106 P:107 T1:110 T1:111 T1:112 T1:113; do a=${job%%:*}; s=${job##*:}; mkdir logs/claims/pre$a$s 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) lane $1 start pretrain $a$s" >> logs/COMPUTE_LOG.txt; python3 scripts/cohort8_pretrain.py $a $s > logs/pretrain_$a$s.out 2>&1
  echo "$(date -u +%FT%TZ) lane $1 end pretrain $a$s rc=$?" >> logs/COMPUTE_LOG.txt; done
