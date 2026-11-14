#!/bin/bash
# DAS + factorial + circuit for the seed-11 replicates (needed for same-parent transfer and component comparison).
cd "$(dirname "$0")/.."; [ -f logs/replicates.txt ] || python3 scripts/pick_replicates.py
S=_eval_conf12; RS=_ev70500000
for P in $(cat logs/replicates.txt); do
  F=checkpoints/native_train/v${P}_full_c1_s4000_seed11${S}/final.json; while [ ! -f $F ]; do sleep 30; done
  DAS_EVAL_STREAM=70500000 python3 scripts/das.py v$P full_c1 11 $S E 2 q7 6
  XII_RDIR=logs/das FA_EVAL=stream:70000000 FA_TAG=_conf_${P}_s11 python3 scripts/factorial.py $RS v$P:full_c1:11:$S
done
