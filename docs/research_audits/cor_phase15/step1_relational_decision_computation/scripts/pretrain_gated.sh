#!/bin/bash
# Investigation XIII Stage 0: readiness-gated P-count pretraining of parent P<seed> (frozen protocol, STAGE0_PROTOCOL_FREEZE.md).
#   block 0 : scripts/cohort13_pretrain.py P <seed>   (original 3000-step protocol, unchanged)             -> checkpoints/zP<seed>_b0/
#   block k : if aux_acc(block k-1) < 0.99 and k <= 2:  scripts/continue_pretrain.py ... <seed> k 3000      -> checkpoints/zP<seed>_b<k>/
#   final   : checkpoints/zP<seed>/ = the last block run (pre.npz, pre.json, and gate.json recording every block's aux_acc)
# Each block is resumable: a block whose pre.json exists is not rerun.
set -e
cd "$(dirname "$0")/.."; SEED=$1; B0=checkpoints/zP${SEED}_b0
if [ ! -f $B0/pre.json ]; then python3 scripts/cohort13_pretrain.py P $SEED; mv checkpoints/zP$SEED $B0; fi
LAST=$B0
for K in 1 2; do
  ACC=$(python3 -c "import json; print(json.load(open('$LAST/pre.json'))['eval']['aux_acc'])")
  python3 -c "import sys; sys.exit(0 if $ACC < 0.99 else 1)" || break
  OUT=checkpoints/zP${SEED}_b$K; [ -f $OUT/pre.json ] || python3 scripts/continue_pretrain.py $LAST/pre.npz $OUT $SEED $K 3000
  LAST=$OUT
done
mkdir -p checkpoints/zP$SEED; cp $LAST/pre.npz $LAST/pre.json checkpoints/zP$SEED/
python3 - <<PY
import json, glob
blocks = sorted(glob.glob("checkpoints/zP${SEED}_b*/pre.json"))
g = [dict(block=b.split("_b")[-1].split("/")[0], aux_acc=json.load(open(b))["eval"]["aux_acc"]) for b in blocks]
json.dump(dict(seed=$SEED, blocks=g, final=g[-1], complete=g[-1]["aux_acc"] >= 0.99), open("checkpoints/zP${SEED}/gate.json", "w"), indent=1); print(g)
PY
