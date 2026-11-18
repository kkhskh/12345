#!/bin/bash
# exploratory: deterministic replays to obtain exact checkpoints around three confirmed transitions (for the cue-conditioned probes)
cd "$(dirname "$0")/../.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; mkdir -p checkpoints/replay
python3 scripts/xvi_replay.py 3:0.9:0.9:0.3333333333333333 402 checkpoints/replay/WA_const_s402 const 15000 1200 800,950,1000,1050,1100,1200 > logs/expl/replay_s402.out 2>&1 &
python3 scripts/xvi_replay.py 2:0.9:0.9:0.05 501 checkpoints/replay/WB_high_s501 cosine 4000 900 500,600,650,700,725,800,900 > logs/expl/replay_wb501.out 2>&1; wait
python3 scripts/xvi_replay.py 3:0.9:0.9:0.3333333333333333 201 checkpoints/replay/WA_cont_C_s201 cont 4000 1300 500,1000,1100,1150,1200,1300 ../step1_relational_learning_order_laws/confirm_outputs/C_s201/step4000.npz > logs/expl/replay_cont201.out 2>&1
touch logs/expl/REPLAYS_DONE
