#!/bin/bash
# exploratory (post-scoring): rebuilt learning-signal estimator on XVI confirmation checkpoints: plateau and ~500 steps before acquisition
cd "$(dirname "$0")/../.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
until [ -f logs/expl/GS2_XV_DONE ]; do sleep 30; done
C3=3:0.9:0.9:0.3333333333333333; D3=3:0.9:0.9:0.05
run(){ o=logs/expl/gs2_$4.json; [ -f $o ] || python3 scripts/xvi_gradsig2.py $1 $2 $o 48 $3 > /dev/null 2>&1; }
run $C3 confirm_outputs/WA_const_s403/step2500.npz 0 XVI_const_s403_2500; run $C3 confirm_outputs/WA_const_s404/step3000.npz 0 XVI_const_s404_3000
run $C3 confirm_outputs/WA_const_s404/step1000.npz 0 XVI_const_s404_1000; run $D3 confirm_outputs/WA_cont_D_s201/step4000.npz 0 XVI_cont_D_s201_4000
run $D3 confirm_outputs/WA_cont_D_s202/step4000.npz 0 XVI_cont_D_s202_4000; run $C3 confirm_outputs/WA_cont_C_s201/step500.npz 0 XVI_cont_C_s201_500
touch logs/expl/GS2_XVI_DONE
