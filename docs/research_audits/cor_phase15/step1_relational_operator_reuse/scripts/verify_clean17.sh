#!/bin/bash
# Clean-extraction verification of cor_inv17_relational_operator_reuse.zip (mandate section 17, 18 steps; step 18 = multipart reconstruction,
# verified separately by scripts/verify_parts17.sh).   Usage: verify_clean17.sh ZIP COR_PHASE15_ROOT LOG
set -u; ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_relational_operator_reuse
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XVII clean extraction check $(date -u +%FT%TZ)"
echo "1. copy the sibling packages required by logs/DEPENDENCIES.json (not standalone) and extract the ZIP"
mkdir -p $T/cor_phase15; for b in $(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['sibling_packages_required']))" $SRC/$NAME/logs/DEPENDENCIES.json); do cp -r $SRC/$b $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"; P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. partial-acquisition trajectories reconstructed (C_s202 replay checkpoints, per-slot F_G)"
mkdir -p /tmp/x17_c && for k in 100 200 300 350 400 500 700 1000; do ln -sf $P/checkpoints/replay/C_s202/step$k.npz /tmp/x17_c/; done
python3 scripts/xvii_slot_traj.py 3:0.9:0.9:0.3333333333333333 /tmp/x17_c /tmp/x17_slot.json > /dev/null 2>&1
python3 -c "import json; a=json.load(open('logs/expl/dev_slot_C_s202_early.json')); b=json.load(open('/tmp/x17_slot.json')); print('   identical:', [x['F_G'] for x in a['traj']]==[x['F_G'] for x in b['traj']])"
echo "4. O/D derivation: source-level discrepancy check recomputed"
cp logs/xvii_od_check.json /tmp/x17_od.json; python3 scripts/xvii_od_check.py > /dev/null 2>&1; python3 -c "import json; print('   identical:', json.load(open('/tmp/x17_od.json'))==json.load(open('logs/xvii_od_check.json')))"
echo "5. GATE ideal-reference decisions recomputed (reference caches against fresh exact fits)"
python3 - <<'PY'
import sys, os, glob, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); import xvitask as T, xvidiag as XD, xvi_math_core as M
mx = 0.0
for f in glob.glob("logs/refs_*.npz"):
    c = os.path.basename(f)[5:-4].replace("_", ":", 3); a, b1, b0, nu = XD.parse(c); px, PB = T.joint(a, b1, b0, nu); z = np.load(f); XA = T.xall(6)
    mx = max(mx, float(abs(z["bayes"] - PB).max()), float(abs(z["ordera"] - M.fit(a, px, PB, XA)[0]).max()), float(abs(z["order1"] - M.fit(1, px, PB, XA)[0]).max()))
print("   caches:", len(glob.glob("logs/refs_*.npz")), "max diff:", mx)
PY
echo "6-9. Theorem P; embedding swap, source-matched controls, causal transfer and shared-component necessity recomputed on A3_s803 (frozen battery)"
python3 - <<'PY'
import sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "../step1_cross_family_relational_foundations/src"); import gatelib as GL, xviislot as XS
a = json.load(open("logs/analysis/A3_s803.json")); p = GL.load(f"confirm_outputs/A3_s803/step{a['selected']}.npz"); D = XS.SlotDiag(a["cond"], n=600)
X = np.random.default_rng(3).integers(0, 3, (64, 6, 6)); print("   Theorem P max diff:", max(XS.theorem_P_check(p, X, pm) for pm in ([1, 0, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])))
mx = 0.0
for n, (i, j) in enumerate(a["pairs"]):
    for w in ("0-3", "0-2", "2-3"):
        lo, hi = map(int, w.split("-")); s = D.score(p, (i, j, lo, hi))["F_G"]
        mx = max(mx, abs(s[j] - a["battery"][w]["recipient"][n]), abs(s[i] - a["battery"][w]["donor"][n]))
    mx = max(mx, abs(D.score(p, pat=(i, j, (0, 1), (0, 1, 2, 3), "all"))["F_G"][j] - a["battery"]["route_L01"]["recipient"][n]))
print("   battery entries (swap [0,3), transfer [0,2), necessity [2,3) donor, routing) max |recomputed - archived|:", mx)
PY
echo "10-11. confirmation statistics and frozen verdicts recomputed by the fail-closed scorer"
cp logs/score_XVII.json /tmp/x17_s.json; python3 scripts/score_XVII.py > /dev/null 2>&1; python3 -c "import json; a=json.load(open('/tmp/x17_s.json')); b=json.load(open('logs/score_XVII.json')); print('   identical:', a==b, {k: v['call'] for k, v in b['verdicts']['pooled'].items()})"
echo "12. fail-closed scoring: synthetic preflight and a refusal on the real data"
cp logs/preflight_XVII.json /tmp/x17_pf.json; python3 scripts/preflight_XVII.py > /dev/null 2>&1; python3 -c "import json; print('   preflight all_pass:', json.load(open('logs/preflight_XVII.json'))['all_pass'])"; cp /tmp/x17_pf.json logs/preflight_XVII.json
X=$(mktemp -d); mkdir -p $X/pkg; cp -r logs src scripts $X/pkg/; ln -s $P/confirm_outputs $X/pkg/confirm_outputs; ln -s $P/learning $X/pkg/learning; for sib in ../step1_*; do ln -s $(readlink -f $sib) $X/$(basename $sib) 2>/dev/null; done
rm $X/pkg/logs/analysis/A2_s705.json; python3 $X/pkg/scripts/score_XVII.py $X/pkg $X/o.json > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/o.json ] && echo "   real data with one analysis removed: refused (rc=$rc)" || echo "   NOT REFUSED (rc=$rc)"; rm -rf $X
echo "13. every included checkpoint loads with finite outputs"
python3 - <<'PY'
import sys, glob, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "../step1_cross_family_relational_foundations/src"); import gatelib as GL, xviislot as XS
X = np.random.default_rng(1).integers(0, 3, (4, 6, 6)); n = bad = 0
for f in sorted(glob.glob("confirm_outputs/*/step*.npz") + glob.glob("checkpoints/*/*/step*.npz") + glob.glob("learning/*/step*.npz")):
    n += 1
    try: bad += not np.isfinite(XS.probs(GL.load(f), X)).all()
    except Exception: bad += 1
print(f"   checkpoints: {n} loaded, {bad} failed")
PY
echo "14. representative intervention replayed: learning arm A3_s803 targeted (loss on min U), first 100 of 400 steps"
read SEL LOSS < <(python3 -c "import json; a=json.load(open('logs/analysis/A3_s803.json')); print(a['selected'], min(a['U']))")
python3 scripts/xvii_slot_train.py 3:0.9:0.9:0.3333333333333333 confirm_outputs/A3_s803/step$SEL.npz /tmp/x17_la 100 all - $LOSS 1 100 600 > /dev/null 2>&1
python3 -c "import json; a=json.load(open('learning/A3_s803_targeted/slot_traj.json'))['traj']; b=json.load(open('/tmp/x17_la/slot_traj.json'))['traj']; print('   F_G at steps 0 and 100 identical:', a[0]['F_G']==b[0]['F_G'] and a[1]['F_G']==b[1]['F_G'])"
echo "15. deterministic training reproduced bitwise (frozen trainer, A2_s701, stopped after its step-120 snapshot; step 100 compared)"
rm -rf /tmp/x17_rp; BLIND=1 SNAP=20 python3 scripts/xvii_train.py 2:0.9:0.9:0.05 701 /tmp/x17_rp cosine 4000 25 - > /dev/null 2>&1 & TP=$!
while [ ! -f /tmp/x17_rp/step120.npz ] && kill -0 $TP 2>/dev/null; do sleep 2; done; kill $TP 2>/dev/null; wait $TP 2>/dev/null
python3 -c "import numpy as np; a=np.load('/tmp/x17_rp/step100.npz'); b=np.load('confirm_outputs/A2_s701/step100.npz'); print('   max |diff| over', len(a.files), 'arrays:', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))"
echo "16. external dependencies by exact hash"; cp logs/DEPENDENCIES.json /tmp/x17_d.json; python3 scripts/deps_inventory17.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x17_d.json')); b=json.load(open('logs/DEPENDENCIES.json')); print('   modules:', len(b['external_python_modules']), '| identical:', a==b)"; cp /tmp/x17_d.json logs/DEPENDENCIES.json
echo "17. reviewer corrections and primary results checked against the reports"; python3 scripts/check_reports17.py
echo "   scripts/src with absolute local paths: $(grep -l '/home/claude' scripts/*.py src/*.py 2>/dev/null | wc -l)"
echo "18. multipart reconstruction: see logs/VERIFY_PARTS.txt (scripts/verify_parts17.sh)"
echo "== done"
} > $LOG 2>&1
