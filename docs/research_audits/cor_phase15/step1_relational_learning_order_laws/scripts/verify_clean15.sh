#!/bin/bash
# Clean-extraction verification of cor_inv15_relational_learning_order_laws.zip (the 16 mandated steps).  Usage: verify_clean15.sh ZIP COR_PHASE15_ROOT LOG
set -u; ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_relational_learning_order_laws
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XV clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME and extract the ZIP"; mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"; P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3+6. generator definitions, exact posteriors, gate-free formula, source invisibility, minimal orders and benefit matching (xv_math_verify.py)"
cp logs/xv_math_verified.json /tmp/x15_v.json; python3 scripts/xv_math_verify.py > /dev/null 2>&1; python3 -c "import json; a=json.load(open('/tmp/x15_v.json')); b=json.load(open('logs/xv_math_verified.json')); print('   reproduced:', a==b, '| all_pass:', b['all_pass'])"
echo "4+5. exact decision rules and interaction-order limits recomputed (xv_math.py on the four conditions)"
python3 scripts/xv_math.py /tmp/x15_m.json 2:0.9 3:0.9 2:0.9:0.05 3:0.9:0.05 > /dev/null 2>&1
python3 - <<'PY'
import json
a = json.load(open("logs/xv_math_dev.json")); b = json.load(open("/tmp/x15_m.json")); bad = []
for k in b:
    for f in ("gain_gate", "gain_intermediate", "min_decision_order", "order_a_has_zero_G"):
        x, y = a[k][f], b[k][f]
        if (abs(x - y) > 1e-6 if isinstance(x, float) else x != y): bad.append((k, f))
    if abs(a[k]["bayes"]["NLL"] - b[k]["bayes"]["NLL"]) > 1e-9: bad.append((k, "bayes"))
print("   conditions:", len(b), "| mismatches:", bad, "| min orders:", {k: b[k]["min_decision_order"] for k in b})
PY
echo "7. every included checkpoint loads with finite outputs"
python3 - <<'PY'
import sys, os, glob, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "src"); sys.path.insert(0, "../step1_cross_family_relational_foundations/src"); sys.path.insert(0, "../step1_latent_reliability/src")
import gatelib as GL, xvtask as XT, nets as N, bayes as B
X = XT.sample(8, np.random.default_rng(1), 2, 0.9)["X"]; n = bad = 0
for f in sorted(glob.glob("confirm_outputs/[ABCD]_s*/step*.npz") + glob.glob("checkpoints/dev*/*/step*.npz")):
    if "T4ds" in f: continue
    n += 1
    try: bad += not np.isfinite(GL.probs(GL.load(f), X)).all()
    except Exception: bad += 1
env = B.Env("T4", K=3, Q=6); tree = jax.tree_util.tree_structure(N.init_base("deepsets", "M", env, 0)); fwd = N.make_forward("deepsets", 3); X4 = B.sample_fast(env, 8, np.random.default_rng(1))["X"]; m = bad4 = 0
for f in sorted(glob.glob("confirm_outputs/T4ds_s*/step*.npz") + glob.glob("checkpoints/dev/T4ds*/step*.npz")):
    m += 1; z = np.load(f)
    try: bad4 += not np.isfinite(np.asarray(fwd(jax.tree_util.tree_unflatten(tree, [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))]), jnp.asarray(X4))["answer"])).all()
    except Exception: bad4 += 1
print(f"   GATE-variant checkpoints: {n} loaded, {bad} failed; DeepSets T4: {m} loaded, {bad4} failed")
PY
echo "8. learning-stage measurements reproduced from checkpoints (2 GATE-variant runs + 1 DeepSets run)"
for r in "2:0.9:0.3333333333333333 B_s201" "3:0.9:0.3333333333333333 C_s204"; do set -- $r; python3 scripts/xv_traj.py $1 confirm_outputs/$2 98700000 1500 /tmp/x15_t_$2.json > /dev/null 2>&1; python3 -c "import json; a=json.load(open('confirm_outputs/traj_$2.json')); b=json.load(open('/tmp/x15_t_$2.json')); print('   $2 trajectory identical:', a['traj']==b['traj'])"; done
python3 scripts/t4_ds_traj.py confirm_outputs/T4ds_s301 301 98800000 3000 /tmp/x15_t4.json > /dev/null 2>&1; python3 -c "import json; a=json.load(open('confirm_outputs/traj_T4ds_s301.json')); b=json.load(open('/tmp/x15_t4.json')); print('   T4ds_s301 trajectory identical:', a['traj']==b['traj'])"
echo "9+10+12. transition times, abruptness, competence gates and frozen verdicts recomputed by the fail-closed scorers"
for s in XV T4; do cp logs/score_$s.json /tmp/x15_s$s.json; python3 scripts/score_$s.py > /dev/null 2>&1; python3 -c "import json; a=json.load(open('/tmp/x15_s$s.json')); b=json.load(open('logs/score_$s.json')); print('   score_$s identical:', a==b, b['verdicts'], 'competent:', b['n_competent'])"; done
echo "11. principal statistics recomputed (threshold sensitivity; the withdrawn gradient-signal output is also reproduced, as a record)"
cp logs/xv_sensitivity.json /tmp/x15_sens.json; python3 scripts/xv_sensitivity.py > /dev/null 2>&1; python3 -c "import json; print('   sensitivity identical:', json.load(open('/tmp/x15_sens.json'))==json.load(open('logs/xv_sensitivity.json')))"
python3 scripts/xv_gradsig.py 2:0.9:0.3333333333333333 confirm_outputs/B_s201 0,175 201 /tmp/x15_g.json > /dev/null 2>&1; python3 -c "import json; a=json.load(open('logs/expl/gradsig_B_s201.json')); b=json.load(open('/tmp/x15_g.json')); print('   gradient signal B_s201 identical:', a['points']==b['points'])"
echo "13. representative relational counterfactual replay (x_mech.py on B_s201: gate flips, residual patches, probes)"
python3 scripts/x_mech.py 2:0.9:0.3333333333333333 confirm_outputs/B_s201 500 /tmp/x15_mech.json > /dev/null 2>&1; python3 -c "import json; a=json.load(open('logs/expl/mech_B_s201.json')); b=json.load(open('/tmp/x15_mech.json')); print('   identical:', a['points']==b['points'])"
echo "14. incomplete frozen inputs are refused"
X=$(mktemp -d); cp -r . $X/pkg 2>/dev/null; for sib in ../step1_*; do ln -s $(readlink -f $sib) $X/$(basename $sib) 2>/dev/null; done
rm $X/pkg/confirm_outputs/traj_C_s202.json; python3 scripts/score_XV.py $X/pkg $X/o1.json > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/o1.json ] && echo "   score_XV without one trajectory: refused (rc=$rc)" || echo "   score_XV NOT REFUSED (rc=$rc)"
rm $X/pkg/confirm_outputs/T4ds_s303/step1000.npz; python3 scripts/score_T4.py $X/pkg $X/o2.json > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/o2.json ] && echo "   score_T4 without one snapshot: refused (rc=$rc)" || echo "   score_T4 NOT REFUSED (rc=$rc)"; rm -rf $X
echo "15. external dependencies by exact hash"; cp logs/DEPENDENCIES.json /tmp/x15_d.json; python3 scripts/deps_inventory15.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x15_d.json')); b=json.load(open('logs/DEPENDENCIES.json')); print('   modules:', len(b['external_python_modules']), '| mismatches:', [k for k in a['external_python_modules'] if a['external_python_modules'][k]!=b['external_python_modules'].get(k)], '| siblings:', b['sibling_packages_required'])"; cp /tmp/x15_d.json logs/DEPENDENCIES.json
echo "16. representative training reproduced bitwise (B_s201 retrained from scratch to step 300 and compared with its archived snapshot)"
BLIND=1 python3 scripts/x_train.py 2:0.9:0.3333333333333333 201 /tmp/x15_tr 300 > /dev/null 2>&1; python3 -c "
import numpy as np; a=np.load('/tmp/x15_tr/step300.npz'); b=np.load('confirm_outputs/B_s201/step300.npz'); print('   B_s201 step 300 max |diff|:', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))"
n=$(grep -rl --exclude=verify_clean15.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src with absolute local paths: $n"
echo "== done"; rm -rf $T /tmp/x15_tr
} 2>&1 | tee $LOG
