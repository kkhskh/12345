#!/bin/bash
# Generic lane (XIII-B): each queue line is a command run from the package root; claims by mkdir on a hash of the line.
# XIII-B fix of XIII review 2: the job's own exit status is recorded (XIII's lane.sh logged the status of $(date)).
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; mkdir -p logs/claims logs/run
Q=${QUEUE:-logs/queue.txt}
while read -r line; do [ -z "$line" ] && continue; key=$(echo "$line" | md5sum | cut -c1-12); mkdir logs/claims/$key 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) lane $1 start [$key] $line" >> logs/COMPUTE_LOG.txt
  bash -c "$line" > logs/run/$key.out 2>&1; rc=$?; echo "$(date -u +%FT%TZ) lane $1 end [$key] rc=$rc" >> logs/COMPUTE_LOG.txt
done < $Q
