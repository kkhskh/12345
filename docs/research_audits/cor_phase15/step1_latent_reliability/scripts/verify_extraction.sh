#!/bin/bash
# Clean-extraction check: unzip the package into an empty dir, re-run the math checks and the aggregator there,
# and compare with the shipped outputs.
set -e
Z=$1; D=$(mktemp -d)
unzip -q "$Z" -d "$D"; cd "$D"/step1_latent_reliability
python3 -I src/math_checks.py > /dev/null 2>&1 || python3 src/math_checks.py > /dev/null
cp logs/verdicts.json /tmp/verdicts_shipped.json; cp VERDICTS.md /tmp/VERDICTS_shipped.md
python3 scripts/aggregate.py > /dev/null
diff -q /tmp/VERDICTS_shipped.md VERDICTS.md && echo "VERDICTS reproduced"
python3 - <<'PY'
import json; a=json.load(open('/tmp/verdicts_shipped.json')); b=json.load(open('logs/verdicts.json'))
print("verdicts.json identical:", a["verdicts"] == b["verdicts"] and a["primary_answer_frozen"] == b["primary_answer_frozen"])
PY
grep -q "0.42523 0.42523 0.14954" logs/math_checks.txt && echo "E1 example reproduced"
python3 -c "import sys; sys.path.insert(0,'src'); import ckpt, glob; n=0
for f in glob.glob('checkpoints/main/*.npz'):
    (ckpt.load_base if ('mlp' in f or 'deepsets' in f) else ckpt.load)(f); n+=1
print('checkpoints loadable:', n)"
echo "extracted to $D"
