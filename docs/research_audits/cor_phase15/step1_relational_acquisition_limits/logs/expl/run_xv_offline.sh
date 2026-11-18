#!/bin/bash
# exploratory: XVI diagnostics on XV's existing snapshots (arity-3 C, D and arity-2 A, B; 16 runs), low priority
cd "$(dirname "$0")/../.."; echo 8 > /proc/self/autogroup 2>/dev/null; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
XVO=../step1_relational_learning_order_laws/confirm_outputs
for r in C_s201 C_s202 C_s203 C_s204 D_s201 D_s202 D_s203 D_s204 A_s201 A_s202 A_s203 A_s204 B_s201 B_s202 B_s203 B_s204; do
  case ${r:0:1} in C) c=3:0.9:0.9:0.3333333333333333;; D) c=3:0.9:0.9:0.05;; A) c=2:0.9:0.9:0.05;; B) c=2:0.9:0.9:0.3333333333333333;; esac
  [ -f logs/expl/xvdiag_XV_$r.json ] || nice -n 5 python3 scripts/xvi_traj_offline.py $c $XVO/$r logs/expl/xvdiag_XV_$r.json
done; touch logs/expl/XV_OFFLINE_DONE
