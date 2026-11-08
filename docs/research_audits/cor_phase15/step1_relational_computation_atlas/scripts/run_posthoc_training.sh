#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_training.py" > /dev/null; do sleep 30; done
for j in pp3ph_T1_resetlast_freeze01 pp3ph_T1_reset1; do
  [ -f checkpoints/$j.json ] || python3 scripts/run_training.py $j > logs/train_$j.out 2>&1
  echo "$(date +%T) $j done" >> logs/training_progress.log
done
