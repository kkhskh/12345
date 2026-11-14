#!/bin/bash
# Same-parent (seed10 <-> seed11, both directions), cross-parent (all ordered pairs of complete parents, seed 10), and C1 -> full transfer.
cd "$(dirname "$0")/.."; [ -f logs/replicates.txt ] || python3 scripts/pick_replicates.py
S=_eval_conf12; RS=_ev70500000; export XII_RDIR=logs/das XI_OUT=cross_conf
COMP=$(python3 -c "
import json,os
print(' '.join(p for p in ['P150','P151','P152','P153','P154','P155','P156'] if os.path.exists(f'checkpoints/z{p}/pre.json') and json.load(open(f'checkpoints/z{p}/pre.json'))['eval']['aux_acc']>=0.99))")
R() { echo logs/das/v$1_$2_s4000_seed$3${S}_E_L2_q7_k6${RS}_R.npy; }
for P in $(cat logs/replicates.txt); do
  while [ ! -f $(R $P full_c1 11) ] || [ ! -f $(R $P full_c1 10) ]; do sleep 30; done
  python3 scripts/cross_interchange.py v$P:full_c1:10:$S v$P:full_c1:11:$S $RS
  python3 scripts/cross_interchange.py v$P:full_c1:11:$S v$P:full_c1:10:$S $RS
done
for A in $COMP; do for B in $COMP; do [ $A = $B ] && continue
  while [ ! -f $(R $B full_c1 10) ]; do sleep 30; done; python3 scripts/cross_interchange.py v$A:full_c1:10:$S v$B:full_c1:10:$S $RS; done; done
for P in $COMP; do while [ ! -f $(R $P full_c1 10) ]; do sleep 30; done; python3 scripts/cross_interchange.py v$P:c1:10:$S v$P:full_c1:10:$S $RS; done
