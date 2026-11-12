#!/bin/bash
# Confirmation lane (frozen): claims jobs from logs/confqueue.txt.
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" PHASE=conf; mkdir -p logs/claims logs/conf
while read -r line; do [ -z "$line" ] && continue; key=$(echo $line | tr ' ' '_'); mkdir logs/claims/conf_$key 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) lane $1 start CONF $line" >> logs/COMPUTE_LOG.txt
  python3 scripts/downstream.py $line > logs/conf/$key.out 2>&1; echo "$(date -u +%FT%TZ) lane $1 end CONF $line rc=$?" >> logs/COMPUTE_LOG.txt
done < logs/confqueue.txt
