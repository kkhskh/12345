#!/bin/bash
# Replacement queue (compute-budget cut, 2 CPUs). Waits for the in-flight pre_plocal (pid passed as $1).
cd "$(dirname "$0")/.."; LRS=../step1_latent_reliability/scripts; O=checkpoints
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "$(date +%T) pre_plocal done" >> logs/progress.log
run() { [ -f $O/$1.json ] || python3 scripts/run_job.py $1 > logs/train_$1.out 2>&1; echo "$(date +%T) $1 done" >> logs/progress.log; }
run ft_plocal
mkdir -p $O/heldout
for s in 14 15; do
  [ -f $O/heldout/T1_M_K3_s$s.json ] || python3 $LRS/train_run.py T1 M $s 6000 $O/heldout 3 tf > logs/train_heldout_T1_s$s.out 2>&1
  [ -f $O/heldout/T4_M_K3_s${s}_fromT1.json ] || python3 $LRS/train_curriculum.py $s 3000 $O/heldout $O/heldout/T1_M_K3_s$s.npz > logs/train_heldout_T4_s$s.out 2>&1
  echo "$(date +%T) heldout tf $s done" >> logs/progress.log
done
for s in 3 4; do [ -f $O/heldout/T4_M_K3_s${s}_deepsets.json ] || python3 $LRS/train_run.py T4 M $s 6000 $O/heldout 3 deepsets > logs/train_heldout_ds$s.out 2>&1; echo "$(date +%T) heldout ds $s done" >> logs/progress.log; done
for j in pre_pcount ft_pcount pre_pgroup ft_pgroup gate_s0 trans_s0; do run $j; done
echo "$(date +%T) ALL DONE" >> logs/progress.log
