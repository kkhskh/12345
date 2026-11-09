#!/bin/bash
cd "$(dirname "$0")/.."
for j in pp3_T2 pp3ph_T1_resetlast_freeze01 pp3ph_T1_reset1; do
  [ -f checkpoints/$j.json ] || python3 scripts/run_training.py $j > logs/train_$j.out 2>&1
  echo "$(date +%T) $j done" >> logs/training_progress.log
done
