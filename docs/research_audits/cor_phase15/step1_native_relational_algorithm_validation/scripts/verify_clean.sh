#!/bin/bash
# Clean-extraction verification of cor_inv12_native_relational_algorithm_validation.zip (8 steps, REPRODUCTION.md).
# Usage: verify_clean.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_native_relational_algorithm_validation
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XII clean extraction check $(date -u +%FT%TZ)"
echo "1. copy the cor_phase15 tree WITHOUT $NAME to $T/cor_phase15, extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. mathematical verification (math_xii.py; XI audit scripts unchanged in XI package)"
cp logs/math_xii.json /tmp/xii_m.json; nice python3 scripts/math_xii.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/xii_m.json')); b=json.load(open('logs/math_xii.json')); print('   math_xii.json', 'identical' if a==b else 'DIFFERENT', '| M1 max err', max(b['M1_independent_copy_esym_theorem_max_abs_err'].values()))"
echo "4. frozen-prediction verdicts re-derived from archived raw results (score_xii.py)"
cp logs/score_xii.json /tmp/xii_s.json; python3 scripts/score_xii.py 2>&1 | sed 's/^/   /'
echo "5. statistics / confidence intervals recomputed (inside score_xii.py, fixed bootstrap seeds)"
python3 -c "import json; a=json.load(open('/tmp/xii_s.json')); b=json.load(open('logs/score_xii.json')); print('   verdicts identical:', a['verdicts']==b['verdicts'], '| effects+CIs identical:', a['effects']==b['effects'], '| checks identical:', a['checks']==b['checks'])"
echo "6. replay of a causal intervention (alt_ablation on the first complete fresh parent, confirmation stream)"
PP=$(python3 -c "import json; print(json.load(open('logs/score_xii.json'))['complete'][0])")
XII_RDIR=logs/das FA_EVAL=stream:70000000 FA_TAG=_verify python3 scripts/alt_ablation.py _ev70500000 v$PP:full_c1:10:_eval_conf12 2>&1 | grep -v Warn | head -1 | cut -c1-160 | sed 's/^/   /'
python3 - <<PY
import json
a = json.load(open("logs/alt_ablation_conf.json"))["v$PP:full_c1:10:_eval_conf12"]; b = json.load(open("logs/alt_ablation_verify.json"))["v$PP:full_c1:10:_eval_conf12"]
d = max(abs(a[k][m] - b[k][m]) for k in a if k != "single_head_path_inA" for m in ("KL_bayes", "KL_e0", "pref_e0", "E_interaction_remaining")); print("   max |diff| vs archived:", d)
PY
rm -f logs/alt_ablation_verify.json
echo "7. no undocumented local files: external dependency inventory + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/xii_d.json; python3 scripts/deps_inventory.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/xii_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
print("   external data files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| missing:", b["missing"], "| modules identical:", a["external_python_modules"] == b["external_python_modules"])
PY
cp /tmp/xii_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files containing absolute local paths: $n"
echo "8. failures recorded above (any line marked FAILED / DIFFERENT / mismatches > 0)"
echo "== done"; rm -rf $T
} 2>&1 | tee $LOG
