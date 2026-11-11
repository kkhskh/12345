#!/bin/bash
# Development job lane: claims jobs from logs/devqueue.txt (mkdir lock), runs scripts/downstream.py.  Arg1 = lane name; Arg2 = 'wait' to
# start only after the Stage-0 pretraining lane has finished.
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; mkdir -p logs/claims logs/dev
if [ "$2" = "wait" ]; then while pgrep -f pretrain_lane.sh > /dev/null; do sleep 60; done; fi
while read -r line; do [ -z "$line" ] && continue; key=$(echo $line | tr ' ' '_'); mkdir logs/claims/dev_$key 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) lane $1 start $line" >> logs/COMPUTE_LOG.txt
  python3 scripts/downstream.py $line > logs/dev/$key.out 2>&1; echo "$(date -u +%FT%TZ) lane $1 end $line rc=$?" >> logs/COMPUTE_LOG.txt
done < ${QUEUE:-logs/devqueue.txt}
