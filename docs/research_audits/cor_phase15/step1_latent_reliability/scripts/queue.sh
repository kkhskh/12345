#!/bin/bash
# usage: queue.sh JOBFILE ; each line: TASK SIZE SEED STEPS OUTDIR [K]; runs sequentially, skips finished jobs
cd "$(dirname "$0")/.."
while read -r t s sd st out k a; do
  [ -z "$t" ] && continue
  tag="${t}_${s}_K${k:-3}_s${sd}"; [ -n "$a" ] && [ "$a" != tf ] && tag="${tag}_$a"
  if [ -f "$out/$tag.json" ]; then continue; fi
  echo "$(date +%T) start $tag" >> logs/train/queue.log
  python3 scripts/train_run.py $t $s $sd $st $out ${k:-3} ${a:-tf} > logs/train/$tag.out 2>&1
  echo "$(date +%T) done $tag" >> logs/train/queue.log
done < "$1"
