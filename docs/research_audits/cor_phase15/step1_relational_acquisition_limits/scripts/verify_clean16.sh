#!/bin/bash
# Clean-extraction verification of cor_inv16_relational_acquisition_limits.zip (the 16 steps of mandate section 13).
# Usage: verify_clean16.sh ZIP COR_PHASE15_ROOT LOG
set -u; ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_relational_acquisition_limits
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XVI clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME (siblings are required: the archive is not standalone) and extract the ZIP"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"; P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. revised GATE mathematics (xvi_math_verify.py, 7 checks) reproduced"
cp logs/xvi_math_verified.json /tmp/x16_v.json; python3 scripts/xvi_math_verify.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x16_v.json')); b=json.load(open('logs/xvi_math_verified.json')); print('   reproduced:', a==b, '| all_pass:', b['all_pass'], {k: v['pass_'] for k, v in b['checks'].items()})"
echo "4. decision-value and cue-matching grid recomputed"
cp logs/xvi_value_grid.json /tmp/x16_g.json; python3 scripts/xvi_value_search.py > /dev/null 2>&1; python3 -c "import json; print('   value grid identical:', json.load(open('/tmp/x16_g.json'))==json.load(open('logs/xvi_value_grid.json')))"
echo "5. arity-4 construction recomputed (orders 1-5, 7 positions)"
python3 scripts/xvi_math.py /tmp/x16_a4.json 2:0.9:0.9:0.3333333333333333:7 3:0.9:0.9:0.3333333333333333:7 4:0.9:0.9:0.3333333333333333:7 4:0.9:0.9:0.05:7 3:0.9:0.9:0.05:7 > /dev/null 2>&1
python3 - <<'PY'
import json; a = json.load(open("logs/xvi_math_arity4.json")); b = json.load(open("/tmp/x16_a4.json")); bad = []
for k in b:
    for f in ("gain_gate", "gain_intermediate", "min_decision_order", "MI_g_proper_gate_subsets_max"):
        x, y = a[k][f], b[k][f]
        if (abs(x - y) > 1e-9 if isinstance(x, float) else x != y): bad.append((k, f))
print("   conditions:", len(b), "| mismatches:", bad, "| min orders:", {k: b[k]["min_decision_order"] for k in b})
PY
echo "6. every new checkpoint loads with finite outputs"
python3 - <<'PY'
import sys, glob, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "../step1_cross_family_relational_foundations/src")
import gatelib as GL, xvitask as XT
X = XT.sample(8, np.random.default_rng(1), 2, 0.9, 0.9, 0.05)["X"]; n = bad = 0
for f in sorted(glob.glob("confirm_outputs/*/step*.npz") + glob.glob("checkpoints/*/*/step*.npz")):
    n += 1
    try: bad += not np.isfinite(GL.probs(GL.load(f), X)).all()
    except Exception: bad += 1
print(f"   checkpoints: {n} loaded, {bad} failed")
PY
echo "7. extended-training measurements reproduced: inline trajectory points recomputed offline from archived snapshots"
python3 - <<'PY'
import sys, os, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "../step1_cross_family_relational_foundations/src")
import gatelib as GL, xvidiag as XD
for rd, cond, steps in (("confirm_outputs/WA_const_s401", "3:0.9:0.9:0.3333333333333333", (500, 7500, 15000)), ("confirm_outputs/WA_cont_D_s201", "3:0.9:0.9:0.05", (500, 4000)), ("confirm_outputs/WB_low_s501", "2:0.9:0.55:0.05", (1000, 4000))):
    D = XD.Diag(cond, 99_700_000, 1500, "logs"); L = {json.loads(l)["step"]: json.loads(l) for l in open(os.path.join(rd, "traj.jsonl"))}; mx = 0.0
    for k in steps:
        p = GL.load(os.path.join(rd, f"step{k}.npz")); r = D.score(GL.probs(p, D.X), GL.probs(p, D.Xf), GL.probs(p, D.Xw))
        mx = max(mx, max(abs(r[f] - L[k][f]) for f in ("KL", "F_G", "G", "A_p", "acc")))
    print(f"   {rd}: steps {steps} max |offline - inline| = {mx:.2e}")
PY
echo "8+10+12. acquisition times, censoring, survival statistics, floor-aware stages and frozen verdicts recomputed by the fail-closed scorer"
cp logs/score_XVI.json /tmp/x16_s.json; python3 scripts/score_XVI.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x16_s.json')); b=json.load(open('logs/score_XVI.json')); print('   score_XVI identical:', a==b, b['verdicts'])"
echo "9. cue-conditioned probe baselines and a representative probe recomputed"
python3 scripts/xvi_probe.py 3:0.9:0.9:0.3333333333333333 ../step1_relational_learning_order_laws/confirm_outputs/C_s201/step4000.npz /tmp/x16_p.json XV_C_s201@4000 > /dev/null 2>&1
python3 -c "import json; a=json.load(open('logs/expl/probe_XV_C_s201_4000.json')); b=json.load(open('/tmp/x16_p.json')); print('   probe identical:', a==b, '| baselines', b['baselines'])"
echo "11. optimization diagnostic (rebuilt learning-signal estimator) recomputed on one checkpoint"
python3 scripts/xvi_gradsig2.py 3:0.9:0.9:0.3333333333333333 init:201 /tmp/x16_gs.json 48 0 > /dev/null 2>&1
python3 -c "import json; a=json.load(open('logs/expl/gs2_C_s201_init.json')); b=json.load(open('/tmp/x16_gs.json')); a.pop('seconds'); b.pop('seconds'); print('   gradient signal identical:', a==b)"
echo "13. fail-closed scorer behaviour: synthetic preflight (planted outcomes, refusals) and a refusal on the real data"
cp logs/preflight_XVI.json /tmp/x16_pf.json; python3 scripts/preflight_XVI.py > /dev/null 2>&1; python3 -c "import json; b=json.load(open('logs/preflight_XVI.json')); print('   preflight all_pass:', b['all_pass'])"; cp /tmp/x16_pf.json logs/preflight_XVI.json
X=$(mktemp -d); cp -r . $X/pkg; for sib in ../step1_*; do ln -s $(readlink -f $sib) $X/$(basename $sib) 2>/dev/null; done
head -n -1 $X/pkg/confirm_outputs/WB_high_s502/traj.jsonl > $X/t && mv $X/t $X/pkg/confirm_outputs/WB_high_s502/traj.jsonl
python3 $X/pkg/scripts/score_XVI.py $X/pkg $X/o.json > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/o.json ] && echo "   real data with one trajectory line removed: refused (rc=$rc)" || echo "   NOT REFUSED (rc=$rc)"; rm -rf $X
echo "14. deterministic continuation and fresh training reproduced bitwise (replay to step 500)"
python3 scripts/xvi_replay.py 3:0.9:0.9:0.05 201 /tmp/x16_rc cont 4000 500 500 ../step1_relational_learning_order_laws/confirm_outputs/D_s201/step4000.npz > /dev/null 2>&1
python3 scripts/xvi_replay.py 2:0.9:0.55:0.05 501 /tmp/x16_rf cosine 4000 500 500 > /dev/null 2>&1
python3 -c "
import numpy as np
for a, b in (('/tmp/x16_rc/step500.npz', 'confirm_outputs/WA_cont_D_s201/step500.npz'), ('/tmp/x16_rf/step500.npz', 'confirm_outputs/WB_low_s501/step500.npz')):
    x, y = np.load(a), np.load(b); print('   ', b, 'max |diff|:', max(float(np.abs(x[k] - y[k]).max()) for k in x.files))"
echo "15. external dependencies by exact hash"; cp logs/DEPENDENCIES.json /tmp/x16_d.json; python3 scripts/deps_inventory16.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x16_d.json')); b=json.load(open('logs/DEPENDENCIES.json')); print('   modules:', len(b['external_python_modules']), '| identical:', a==b, '| siblings:', b['sibling_packages_required'])"; cp /tmp/x16_d.json logs/DEPENDENCIES.json
echo "16. reports checked against the primary results"; python3 scripts/check_reports16.py
echo "   scripts/src with absolute local paths: $(grep -l '/home/claude' scripts/*.py src/*.py 2>/dev/null | wc -l)"
echo "== done"
} > $LOG 2>&1
