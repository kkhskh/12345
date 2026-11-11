#!/bin/bash
# Runs the frozen per-model pipeline (cri_run.py conf) on each cohort model as soon as its checkpoint exists.
# Output goes to logs/conf/ and logs/conf_watcher.out; nothing is read until the confirmatory analysis.
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
while true; do
  left=0
  for s in 60 61 62 63 64 65 66 67; do
    for m in cT$s cP$s bcT${s}_b250 bcT${s}_b500; do
      [ -f logs/conf/model_$m.json ] && continue; left=1
      case $m in cT*) f=checkpoints/ft_T1_s$s.json;; cP*) f=checkpoints/ft_P_s$s.json;; b*) f=checkpoints/budget/${m#b}.json;; esac
      [ -f $f ] || continue
      python3 scripts/cri_run.py conf $m >> logs/conf_watcher.out 2>&1
    done
  done
  [ $left -eq 0 ] && break; sleep 120
done
