#!/bin/bash
cd "$(dirname "$0")/../.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
nice -n 5 python3 scripts/xvii_slot_train.py 3:0.9:0.9:0.3333333333333333 checkpoints/replay/C_s202/step1000.npz checkpoints/devtrain/eq_slot1_long 1500 eq 1 all 1 100 600 > logs/expl/devtrain_eq_slot1_long.out 2>&1 &
nice -n 5 python3 scripts/xvii_slot_train.py 3:0.9:0.9:0.3333333333333333 checkpoints/replay/C_s202/step1000.npz checkpoints/devtrain/all_loss1 1000 all - 1 1 100 600 > logs/expl/devtrain_all_loss1.out 2>&1
nice -n 5 python3 scripts/xvii_slot_train.py 3:0.9:0.9:0.3333333333333333 checkpoints/replay/C_s202/step1000.npz checkpoints/devtrain/all_lossall 1000 all - all 1 100 600 > logs/expl/devtrain_all_lossall.out 2>&1
wait; touch logs/expl/DEVTRAIN_DONE
