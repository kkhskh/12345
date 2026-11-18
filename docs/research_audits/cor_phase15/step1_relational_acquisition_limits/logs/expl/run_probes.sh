#!/bin/bash
# exploratory (post-scoring): cue-conditioned probes around three confirmed transitions (replayed checkpoints) and on the two censored continuations;
# plus a bitwise check that a run resumed after the container restart equals uninterrupted training (WB_low_s503: resumed at 1500; compare step 2000)
cd "$(dirname "$0")/../.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
until [ -f logs/expl/REPLAYS_DONE ]; do sleep 30; done
python3 scripts/xvi_replay.py 2:0.9:0.55:0.05 503 /tmp/replay_wb503 cosine 4000 2000 2000 > /dev/null 2>&1 &
C3=3:0.9:0.9:0.3333333333333333; D3=3:0.9:0.9:0.05; H=2:0.9:0.9:0.05
for k in 800 950 1000 1050 1100 1200; do python3 scripts/xvi_probe.py $C3 checkpoints/replay/WA_const_s402/step$k.npz logs/expl/probe_XVI_const_s402_$k.json WA_const_s402@$k; done
for k in 500 600 650 700 725 800 900; do python3 scripts/xvi_probe.py $H checkpoints/replay/WB_high_s501/step$k.npz logs/expl/probe_XVI_WBhigh_s501_$k.json WB_high_s501@$k; done
for k in 500 1000 1100 1150 1200 1300; do python3 scripts/xvi_probe.py $C3 checkpoints/replay/WA_cont_C_s201/step$k.npz logs/expl/probe_XVI_cont_C_s201_$k.json WA_cont_C_s201@$k; done
for r in D_s201 D_s202; do python3 scripts/xvi_probe.py $D3 confirm_outputs/WA_cont_$r/step4000.npz logs/expl/probe_XVI_cont_${r}_4000.json WA_cont_$r@4000; done
wait; python3 -c "
import numpy as np; a=np.load('/tmp/replay_wb503/step2000.npz'); b=np.load('confirm_outputs/WB_low_s503/step2000.npz'); print('WB_low_s503 (resumed after restart at 1500) uninterrupted replay vs archive, step 2000: max|diff|', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))" > logs/expl/resume_determinism_check.txt
touch logs/expl/PROBES_DONE
