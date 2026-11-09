#!/bin/bash
# Clean-extraction check: unzip into an empty dir, re-run the exact-Bayes boundary checks and the post hoc
# reference-rule check, compare key numbers with the shipped logs, and check that every shipped checkpoint loads.
set -e
Z=$1; D=$(mktemp -d); unzip -q "$Z" -d "$D"; cd "$D"
cd step1_relational_computation_atlas
cp logs/bayes_boundaries.json /tmp/bb_shipped.json; cp logs/pp1_reference_rules_posthoc.json /tmp/rr_shipped.json
python3 scripts/bayes_boundaries.py > /dev/null 2>&1
python3 scripts/pp1_reference_rules.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open('/tmp/bb_shipped.json')), json.load(open('logs/bayes_boundaries.json'))
print("B1 unimodality identical:", a["B1_unimodality"] == b["B1_unimodality"])
print("B4 identifiability identical:", a["B4_identifiability"] == b["B4_identifiability"])
print("B2 phase rows identical:", a["B2_phase"] == b["B2_phase"])
r1, r2 = json.load(open('/tmp/rr_shipped.json')), json.load(open('logs/pp1_reference_rules_posthoc.json'))
print("reference rules identical:", r1 == r2)
PY
python3 -c "
import sys, glob; sys.path.insert(0,'../step1_latent_reliability/src'); import ckpt
n=0
for f in glob.glob('checkpoints/*.npz'): ckpt.load(f); n+=1
print('atlas checkpoints loadable:', n)"
echo "extracted to $D"
