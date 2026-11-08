#!/bin/bash
# Frozen replication training (REPLICATION_PROTOCOL_T4.md §1), then affordable controls.
cd "$(dirname "$0")/.."
O=checkpoints/replication; mkdir -p $O logs/replication
for s in 10 11 12 13; do
  [ -f $O/T1_M_K3_s$s.json ] || python3 scripts/train_run.py T1 M $s 6000 $O 3 tf > logs/replication/train_T1_s$s.out 2>&1
  [ -f $O/T4_M_K3_s${s}_fromT1.json ] || python3 scripts/train_curriculum.py $s 3000 $O $O/T1_M_K3_s$s.npz > logs/replication/train_T4_s$s.out 2>&1
  echo "$(date +%T) seed $s trained" >> logs/replication/progress.log
done
for s in 10 11 12 13; do [ -f $O/mlp_grouphist_curriculum_s$s.json ] || python3 scripts/curriculum_base.py $s $O > logs/replication/base_s$s.out 2>&1; done
echo "$(date +%T) replication training done" >> logs/replication/progress.log
# affordable controls (main grid items 5 and 3), only after the replication training
for j in "T0 M 0" "T5 M 0" "T3 M 1"; do set -- $j; [ -f checkpoints/main/${1}_M_K3_s$3.json ] || python3 scripts/train_run.py $1 M $3 6000 checkpoints/main 3 tf > logs/train/$1_M_s$3.out 2>&1; done
echo "$(date +%T) controls done" >> logs/replication/progress.log
