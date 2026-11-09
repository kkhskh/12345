#!/bin/bash
# Background training queue (sequential).  Order chosen by dependency: PP3 feature tasks, beyond-reliability tasks,
# then held-out seeds for the PP2 structural prediction (trained before, but EVALUATED only after, the predictor freeze).
cd "$(dirname "$0")/.."; LRS=../step1_latent_reliability/scripts; O=checkpoints
run() { [ -f $O/$1.json ] || python3 scripts/run_job.py $1 > logs/train_$1.out 2>&1; echo "$(date +%T) $1 done" >> logs/progress.log; }
for j in pre_plocal pre_pcount pre_pgroup ft_plocal ft_pcount ft_pgroup gate_s0 trans_s0; do run $j; done
mkdir -p $O/heldout
for s in 14 15 16; do
  [ -f $O/heldout/T1_M_K3_s$s.json ] || python3 $LRS/train_run.py T1 M $s 6000 $O/heldout 3 tf > logs/train_heldout_T1_s$s.out 2>&1
  [ -f $O/heldout/T4_M_K3_s${s}_fromT1.json ] || python3 $LRS/train_curriculum.py $s 3000 $O/heldout $O/heldout/T1_M_K3_s$s.npz > logs/train_heldout_T4_s$s.out 2>&1
  echo "$(date +%T) heldout tf $s done" >> logs/progress.log
done
for s in 3 4 5; do [ -f $O/heldout/T4_M_K3_s${s}_deepsets.json ] || python3 $LRS/train_run.py T4 M $s 6000 $O/heldout 3 deepsets > logs/train_heldout_ds$s.out 2>&1; done
echo "$(date +%T) ALL DONE" >> logs/progress.log
