#!/bin/bash
# Clean-extraction verification of cor_inv11_native_causal_construction.zip (steps 1-6 of REPRODUCTION.md).
# Usage: verify_clean.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_native_causal_construction
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 tree (without $NAME) to $T/cor_phase15 and extract ZIP"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. mathematics"; cp logs/audit_x1.json /tmp/a1.json; cp logs/audit_x1_extra.json /tmp/a2.json
python3 scripts/audit_x1.py > /dev/null 2>&1; python3 scripts/audit_x1_extra.py > /dev/null 2>&1
python3 - <<'E'
import json
for a, b in (("/tmp/a1.json", "logs/audit_x1.json"), ("/tmp/a2.json", "logs/audit_x1_extra.json")):
    x, y = json.load(open(a)), json.load(open(b)); print("  ", b, "identical" if x == y else "DIFFERENT")
E
echo "4. verdicts"; cp logs/score_conf.json /tmp/s.json; python3 scripts/score_conf.py 2>&1 | sed 's/^/   /'
python3 -c "import json; print('   score_conf.json', 'identical' if json.load(open('/tmp/s.json')) == json.load(open('logs/score_conf.json')) else 'DIFFERENT')"
echo "5. forward-pass causal check (factorial_ablate wP140 full_c1, confirmation stream)"
FA_EVAL=stream:66000000 FA_TAG=_verify python3 scripts/factorial_ablate.py _ev62500000 wP140:full_c1:0:_eval_conf10 2>&1 | grep -v Warn | sed 's/^/   /'
python3 - <<'E'
import json
a = json.load(open("logs/factorial_ablate_conf.json"))["wP140:full_c1:0:_eval_conf10"]; b = json.load(open("logs/factorial_ablate_verify.json"))["wP140:full_c1:0:_eval_conf10"]
d = max(abs(a[k][m] - b[k][m]) for k in a if isinstance(a[k], dict) for m in a[k]); print("   max |diff| vs archived:", d)
E
rm -f logs/factorial_ablate_verify.json
echo "6. dependency inventory"; cp logs/DEPENDENCIES.json /tmp/d.json; python3 scripts/deps_inventory.py > /dev/null 2>&1
python3 - <<'E'
import json
a, b = json.load(open("/tmp/d.json")), json.load(open("logs/DEPENDENCIES.json"))
diff = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
print("   external data files:", len(b["external_data_files"]), "| hash mismatches:", len(diff), "| missing:", b["missing"])
print("   modules identical:", a["external_python_modules"] == b["external_python_modules"], "| software:", b["software"])
E
cp /tmp/d.json logs/DEPENDENCIES.json
echo "== done"; rm -rf $T
} 2>&1 | tee $LOG
