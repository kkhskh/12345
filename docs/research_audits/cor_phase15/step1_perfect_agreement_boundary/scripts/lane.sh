#!/bin/bash
# Training lane: claims the next unclaimed parent from the frozen queue (mkdir lock), trains it with cohort6_train.py, repeats.
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p logs/claims logs/train
for job in T:70 P:80 T:71 P:81 T:72 P:82 T:73 P:83 T:74 P:84 T:75 P:85 T:76 P:86 T:77 P:87; do
  arm=${job%%:*}; s=${job##*:}; mkdir logs/claims/$arm$s 2>/dev/null || continue
  [ -f checkpoints/x${arm/T/T1}$s/step4000.json ] && continue
  echo "$(date -u +%FT%TZ) lane $1 start $arm $s" >> logs/COMPUTE_LOG.txt
  python3 scripts/cohort6_train.py ${arm/T/T1} $s > logs/train/$arm$s.log 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) lane $1 end $arm $s rc=$rc" >> logs/COMPUTE_LOG.txt
done
