#!/bin/bash
cd "$(dirname "$0")/.."
for j in pp4_T1perm pp3_T0 pp3_T1_resetlast pp3_T1_reset01 pp3_T5 pp4_T4perm pp3_scratch pp3_T2; do
  [ -f checkpoints/$j.json ] || python3 scripts/run_training.py $j > logs/train_$j.out 2>&1
  echo "$(date +%T) $j done" >> logs/training_progress.log
done
