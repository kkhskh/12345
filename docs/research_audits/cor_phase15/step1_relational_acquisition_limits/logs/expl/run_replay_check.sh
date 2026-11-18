#!/bin/bash
# exploratory/reproducibility: replay dev run C_const_s150 to step 500 and compare with its archived snapshot (bitwise)
cd "$(dirname "$0")/../.."; echo 19 > /proc/self/autogroup 2>/dev/null; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
nice -n 19 python3 scripts/xvi_replay.py 3:0.9:0.9:0.3333333333333333 150 /tmp/replay_dev const 6000 500 500
python3 -c "
import numpy as np; a=np.load('/tmp/replay_dev/step500.npz'); b=np.load('checkpoints/dev/C_const_s150/step500.npz'); print('dev replay step500 max|diff|', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))" > logs/expl/replay_check_dev.txt
