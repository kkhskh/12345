#!/bin/bash
# Fine-tune the k-th (0/1) same-parent replicate (seed 11) chosen by pick_replicates.py.
cd "$(dirname "$0")/.."; [ -f logs/replicates.txt ] || python3 scripts/pick_replicates.py
P=$(awk -v k=$(( $1 + 1 )) '{print $k}' logs/replicates.txt); [ -z "$P" ] && { echo "no replicate parent $1"; exit 0; }
python3 scripts/native_train.py v$P full_c1 4000 11 label logs/eval_conf12.npz
