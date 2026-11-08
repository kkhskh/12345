#!/bin/bash
cd "$(dirname "$0")/.."
O=checkpoints/replication
for s in 10 11 12 13; do
  while [ ! -f $O/T4_M_K3_s${s}_fromT1.json ]; do sleep 60; done
  [ -f logs/replication/eval_s$s.json ] || python3 scripts/replicate_t4.py $O/T4_M_K3_s${s}_fromT1.npz $O/T1_M_K3_s$s.npz logs/replication/eval_s$s.json > logs/replication/eval_s$s.out 2>&1
done
