#!/bin/bash
cd "$(dirname "$0")/.."
LR=../step1_latent_reliability; CONT=../step1_foundational_atlas_continuation/checkpoints; H=checkpoints/heldout3; mkdir -p $H
run() { [ -f checkpoints/$1.json ] || python3 scripts/run_job3.py $1 > logs/train_$1.out 2>&1; echo "$(date +%T) $1 done" >> logs/progress.log; }
for j in ref_T1p1 ref_T1p2 ref_T1p10 ref_T1p11 ref_rand_s1 ref_rand_s2; do run $j; done
for k in 1 2 3 4; do run pre_pcount_s$k; run ft_pcount_s$k; done
for k in 1 2; do run pre_pgroup_s$k; run ft_pgroup_s$k; done
for k in 1 2; do run pre_plocal_s$k; run ft_plocal_s$k; done
for j in ftlong_pcount_s0 ftlong_T1_s0 ftlong_plocal_s0; do run $j; done
[ -f logs/pp3_dyn3.json ] && grep -q '"6000"' logs/pp3_dyn3.json || python3 scripts/pp3_dyn3.py > logs/pp3_dyn3.out 2>&1; echo "$(date +%T) dyn3 done" >> logs/progress.log
# held-out models for Workstream D (trained now; evaluated only after the D predictor is frozen)
i=0; for P in 0 1 2 10 11 12; do s=$((40+i)); i=$((i+1)); sub=main; [ $P -ge 10 ] && sub=replication
  [ -f $H/T4_M_K3_s${s}_fromT1.json ] || python3 $LR/scripts/train_curriculum.py $s 3000 $H $LR/checkpoints/$sub/T1_M_K3_s$P.npz > logs/train_heldout3_tf$s.out 2>&1
  echo "$(date +%T) heldout3 tf $s (parent T1 s$P) done" >> logs/progress.log; done
for s in 6 7 8 9 10 11 12 13 14 15; do [ -f $H/T4_M_K3_s${s}_deepsets.json ] || python3 $LR/scripts/train_run.py T4 M $s 6000 $H 3 deepsets > logs/train_heldout3_ds$s.out 2>&1; echo "$(date +%T) heldout3 ds $s done" >> logs/progress.log; done
echo "$(date +%T) LANE2 DONE" >> logs/progress.log
