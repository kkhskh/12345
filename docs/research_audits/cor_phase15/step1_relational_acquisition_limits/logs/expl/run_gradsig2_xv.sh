#!/bin/bash
# exploratory: rebuilt learning-signal estimator on XV checkpoints (not XVI confirmation data), low priority, after the XV offline diagnostics
cd "$(dirname "$0")/../.."; echo 8 > /proc/self/autogroup 2>/dev/null; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
until [ -f logs/expl/XV_OFFLINE_DONE ]; do sleep 60; done
XVO=../step1_relational_learning_order_laws/confirm_outputs; C3=3:0.9:0.9:0.3333333333333333; D3=3:0.9:0.9:0.05; A2=2:0.9:0.9:0.05; B2=2:0.9:0.9:0.3333333333333333
run(){ o=logs/expl/gs2_$4.json; [ -f $o ] || nice -n 5 python3 scripts/xvi_gradsig2.py $1 $2 $o 48 $3 > /dev/null 2>&1; }
run $C3 init:201 0 C_s201_init; run $C3 init:201 1 C_s201_init_r1; run $C3 init:201 2 C_s201_init_r2
for k in 1000 2000 4000; do run $C3 $XVO/C_s201/step$k.npz 0 C_s201_$k; done; run $C3 $XVO/C_s201/step4000.npz 1 C_s201_4000_r1; run $C3 $XVO/C_s201/step4000.npz 2 C_s201_4000_r2
for k in 1000 2000 2250 2500; do run $C3 $XVO/C_s204/step$k.npz 0 C_s204_$k; done
for k in 1000 2000 4000; do run $D3 $XVO/D_s201/step$k.npz 0 D_s201_$k; done
for k in 1000 1750 2000 2250; do run $D3 $XVO/D_s203/step$k.npz 0 D_s203_$k; done
for k in 300 450 500; do run $B2 $XVO/B_s201/step$k.npz 0 B_s201_$k; done
for k in 625 775 825; do run $A2 $XVO/A_s201/step$k.npz 0 A_s201_$k; done
touch logs/expl/GS2_XV_DONE
