#!/bin/bash
# Clean-extraction verification of cor_inv13b_unresolved_mechanisms_and_validation.zip (10 steps, REPRODUCTION.md).
# Usage: verify_clean13b.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_relational_decision_computation_followup
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XIII-B clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME to a clean directory and extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. mathematics: censored_bayes.py (beta=1 local sufficiency, beta=0.6 gap) rerun and compared"
cp logs/censored_bayes.json /tmp/x13b_cb.json; nice python3 scripts/censored_bayes.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x13b_cb.json')); b=json.load(open('logs/censored_bayes.json')); print('   identical:', a==b, '| beta1 LOCALQ KL', b['beta1.0']['LOCALQ_KL_to_full'], '| beta0.6 LOCALQ acc', round(b['beta0.6']['LOCALQ_acc'],4))"
echo "4. frozen verdicts re-derived (score_13b.py) and reporting layer (report_13b.py)"
cp logs/score_13b_conf13b.json /tmp/x13b_s.json; cp logs/report_13b.json /tmp/x13b_r.json; python3 scripts/score_13b.py 2>&1 | sed 's/^/   /'
python3 -c "import json; a=json.load(open('/tmp/x13b_s.json')); b=json.load(open('logs/score_13b_conf13b.json')); print('   verdicts identical:', {k:v['verdict'] for k,v in a['predictions'].items()}=={k:v['verdict'] for k,v in b['predictions'].items()}, '| checks identical:', a['checks']==b['checks'])"
echo "5. principal statistics recomputed (report_13b.py: strict diagnostics, Q4 sensitivity, AQ1 correlation)"
python3 scripts/report_13b.py > /dev/null 2>&1; python3 -c "import json; a=json.load(open('/tmp/x13b_r.json')); b=json.load(open('logs/report_13b.json')); print('   report identical:', a==b, '| AQ1', b['AQ1']['verdict'], round(b['AQ1']['corr'],3))"
echo "6. replay of a pre-assembled (early-readout) causal intervention: class_features.py on uP160 seed 21 (confirmation stream)"
CF_STREAM=79000000 CF_TAG=_verify python3 scripts/class_features.py uP160:full_c1:21:_eval_conf13 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/class_features_conf13b_P160.json')); b=json.load(open('logs/class_features_verify.json')); k='uP160:full_c1:21:_eval_conf13'
print('   max |diff| lik_RO1/route_Q/route_R/transfer:', max(abs(a[k][x]-b[k][x]) for x in ('lik_RO1','route_Q','route_R','transfer')))"
rm -f logs/class_features_verify.json
echo "7. replay of the mixture-head control / restoration test: l2_components.py on uP165 seed 21"
L2C_STREAM=79200000 L2C_TAG=_verify python3 scripts/l2_components.py uP165:full_c1:21:_eval_conf13 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/l2_components_conf13b_P165.json')); b=json.load(open('logs/l2_components_verify.json')); k='uP165:full_c1:21:_eval_conf13'
A,B=a[k]['EA'],b[k]['EA']; print('   max |diff| decoy proj:', max(abs(A['decoys'][d]['proj']-B['decoys'][d]['proj']) for d in A['decoys']), '| restoration:', max(abs(A['restoration'][d]['proj']-B['restoration'][d]['proj']) for d in A['restoration'] if isinstance(A['restoration'][d],dict)))"
rm -f logs/l2_components_verify.json
echo "8. representative two-colluder adaptation result: adapt2.py beta=0.6 early01 on uP161 seed 21 (deterministic seeds)"
AD_BETA=0.6 AD_CONDS=early01 AD_STREAM=79500000 AD_OUT=_verify python3 scripts/adapt2.py uP161:full_c1:21:_eval_conf13 > /dev/null 2>&1
python3 -c "
import json; t='uP161_full_c1_s4000_seed21_eval_conf13_b0.6.json'; a=json.load(open('logs/adapt2_conf13b/'+t)); b=json.load(open('logs/adapt2_verify/'+t))
print('   early01 acc archived/rerun', round(a['early01']['acc'],4), round(b['early01']['acc'],4), '| J slope', round(a['early01']['Jpair']['slope'],3), round(b['early01']['Jpair']['slope'],3))"
rm -rf logs/adapt2_verify checkpoints/adapt2/*verify* 2>/dev/null
echo "9. training checkpoint integrity: every seed-21 fine-tune loads and produces finite logits; final.json present"
python3 - <<'PY'
import sys, os, numpy as np, jax.numpy as jnp; sys.path.insert(0, "src"); import ftlib as F, native as NV
X = NV.load_eval("logs/eval_13b.npz")["X"][:64]
for p in ("P160", "P161", "P163", "P164", "P165"):
    d = f"checkpoints/native_train/u{p}_full_c1_s4000_seed21_eval_conf13"
    try: M = F.load_model("u" + p, "full_c1", seed=21, suffix="_eval_conf13"); ok = bool(np.isfinite(np.asarray(F.forward(M, jnp.asarray(X)))).all())
    except Exception as ex: ok = f"FAILED {ex}"
    print(f"   u{p} seed21: final.json {os.path.exists(d + '/final.json')} | {len([f for f in os.listdir(d) if f.startswith('step')])} step checkpoints | loadable+finite {ok}")
PY
echo "10. dependencies on earlier cor_phase15 packages (deps_inventory13b.py) + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/x13b_d.json; python3 scripts/deps_inventory13b.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/x13b_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
print("   software:", b["software"]); print("   external files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| missing:", b["missing"], "| sibling packages:", len(b["sibling_packages_required"]))
PY
cp /tmp/x13b_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean13b.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files with absolute local paths: $n"
echo "== done (any FAILED / DIFFERENT / False / mismatches > 0 is a failure)"; rm -rf $T
} 2>&1 | tee $LOG
