#!/bin/bash
# Wait for process PID $1 to exit, then run lane $2 over queues $3 ...
cd "$(dirname "$0")/.."; pid=$1; name=$2; shift 2
while kill -0 $pid 2>/dev/null; do sleep 30; done
for q in "$@"; do QUEUE=$q bash scripts/lane.sh $name; done
