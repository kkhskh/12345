#!/bin/bash
cd "$(dirname "$0")/.."; LR=../step1_latent_reliability/checkpoints
for s in 11 12 13; do [ -f logs/mech/s$s.json ] || python3 scripts/mechanism_atlas.py $LR/replication/T4_M_K3_s${s}_fromT1.npz logs/mech/s$s.json > logs/mech/s$s.out 2>&1; done
for s in 0 1 2; do [ -f logs/mech/s$s.json ] || python3 scripts/mechanism_atlas.py $LR/posthoc_curriculum/T4_M_K3_s${s}_fromT1.npz logs/mech/s$s.json > logs/mech/s$s.out 2>&1
  [ -f logs/mech/ds$s.json ] || python3 scripts/mechanism_atlas.py $LR/main/T4_M_K3_s${s}_deepsets.npz logs/mech/ds$s.json ds > logs/mech/ds$s.out 2>&1; done
echo done > logs/mech/DONE
