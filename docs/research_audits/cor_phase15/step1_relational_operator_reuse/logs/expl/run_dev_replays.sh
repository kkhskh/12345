#!/bin/bash
# XVII development: dense deterministic replays of existing XVI runs around (partial) acquisition windows
cd "$(dirname "$0")/../.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
X=../step1_relational_learning_order_laws/confirm_outputs
S1=$(python3 -c "print(','.join(str(k) for k in list(range(100,2401,50))))")
python3 scripts/xvi_replay.py 3:0.9:0.9:0.3333333333333333 202 checkpoints/replay/C_s202 cont 4000 2400 $S1 $X/C_s202/step4000.npz > logs/expl/replay_C_s202.out 2>&1 &
S2=$(python3 -c "print(','.join(str(k) for k in list(range(1500,1901,10))))")
python3 scripts/xvi_replay.py 2:0.9:0.55:0.05 503 checkpoints/replay/WB_low_s503 cosine 4000 1900 $S2 > logs/expl/replay_WB_low_s503.out 2>&1
S3=$(python3 -c "print(','.join(str(k) for k in list(range(950,1251,10))))")
python3 scripts/xvi_replay.py 3:0.9:0.9:0.3333333333333333 402 checkpoints/replay/WA_const_s402 const 15000 1250 $S3 > logs/expl/replay_s402.out 2>&1
wait; touch logs/expl/DEV_REPLAYS_DONE
