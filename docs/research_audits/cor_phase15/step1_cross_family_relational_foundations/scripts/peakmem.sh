#!/bin/bash
# XIII-E: run a command and record its peak RSS (sum over the process tree, sampled every 2 s) to $PEAKFILE
"$@" & P=$!; PK=0
while kill -0 $P 2>/dev/null; do R=$(ps -o rss= --ppid $P -p $P 2>/dev/null | awk '{s+=$1} END {print s+0}'); [ "$R" -gt "$PK" ] && PK=$R; sleep 2; done
wait $P; rc=$?; echo "$PK" > ${PEAKFILE:-/dev/stdout}; exit $rc
