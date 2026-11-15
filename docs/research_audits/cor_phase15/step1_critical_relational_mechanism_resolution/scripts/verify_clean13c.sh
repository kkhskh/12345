#!/bin/bash
# Clean-extraction verification of cor_inv13c_critical_relational_mechanism_resolution.zip (12 mandated steps).
# Usage: verify_clean13c.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_critical_relational_mechanism_resolution
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XIII-C clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME into a clean directory and extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. every new checkpoint loads and gives finite logits (ft_factorial runs: all saved steps of every run; adapt3 parameter files)"
python3 - <<'PY'
import sys, os, glob, numpy as np, jax.numpy as jnp; sys.path.insert(0, "src"); import xc, ftlib as F, native as NV
X = NV.Bm.sample_fast(NV.ENV, 32, np.random.default_rng(1))["X"]; bad = 0; n = 0
for d in sorted(glob.glob("checkpoints/ft/*")):
    t = os.path.basename(d); p, _, L, W, D = t.split("_")[:5]; suf = "_".join(t.split("_")[5:]); spec = f"{p}:fx:{L[1:]}:{W[1:]}:{D[1:]}" + (f":{suf}" if suf else "")
    for s in xc.steps_available(spec):
        n += 1
        try: ok = bool(np.isfinite(np.asarray(F.forward(xc.load(spec, s), jnp.asarray(X)))).all())
        except Exception as ex: ok = False
        bad += (not ok)
    print(f"   {t}: final.json {os.path.exists(d + '/final.json')} | {len(xc.steps_available(spec))} checkpoints")
na = len(glob.glob("checkpoints/adapt3/*.npz")); badz = sum(1 for f in glob.glob("checkpoints/adapt3/*.npz") if not all(np.isfinite(v).all() for v in np.load(f).values()))
print(f"   ft checkpoints loaded: {n}, non-finite/failed: {bad} | adapt3 parameter files: {na}, non-finite: {badz}")
PY
echo "4. factorial-training comparisons recomputed (analyze_A.py) and compared with the archived table"
cp logs/analyze_A_final.json /tmp/x13c_A.json; python3 scripts/analyze_A.py _final > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x13c_A.json')); b=json.load(open('logs/analyze_A_final.json')); print('   factorial effects identical:', a['factorial']==b['factorial'], '| finals identical:', a['finals']==b['finals'])"
echo "5. representative decision-model refit (fair_fits.py on uP160 seed 23 and the ro0 confirmation run, confirmation stream)"
FF_STREAM=88300000 FF_TAG=_verify FF_SHAPLEY=0 nice python3 scripts/fair_fits.py uP160:s:23 uP160:fx:21:21:21:ro0T600 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/fair_fits_conf.json')); b=json.load(open('logs/fair_fits_verify.json'))
for k in ('uP160:s:23','uP160:fx:21:21:21:ro0T600'): print('   ', k, 'rho_PM archived/refit', round(a[k]['fams']['PM']['params']['rho'],4), round(b[k]['fams']['PM']['params']['rho'],4), '| KL_test', round(a[k]['fams']['PML']['KL_test'],6), round(b[k]['fams']['PML']['KL_test'],6))"
rm -f logs/fair_fits_verify.json
echo "6. mathematical verification rerun (math_identifiability.py, c_math.py, review-1 exact LOCALQ) and compared"
cp logs/math_identifiability.json /tmp/x13c_m.json; cp logs/c_math.json /tmp/x13c_c.json
nice python3 scripts/math_identifiability.py > /dev/null 2>&1; nice python3 scripts/c_math.py > /dev/null 2>&1
python3 -c "import json; print('   math_identifiability identical:', json.load(open('/tmp/x13c_m.json'))==json.load(open('logs/math_identifiability.json')), '| c_math identical:', json.load(open('/tmp/x13c_c.json'))==json.load(open('logs/c_math.json')))"
(cd reviews/review1_scripts && nice python3 r1_localq_exact.py 2>/dev/null | grep -i "gap_exact" | sed 's/^/   exact LOCALQ (3000 eps): /')
echo "7. frozen verdicts recomputed (collect_13c.py + score_13c.py) and compared; preflight rerun"
cp logs/score_13c.json /tmp/x13c_s.json; python3 scripts/collect_13c.py > /dev/null; python3 scripts/score_13c.py logs/bundle_13c.json > /dev/null
python3 -c "import json; a=json.load(open('/tmp/x13c_s.json')); b=json.load(open('logs/score_13c.json')); print('   verdicts identical:', a['summary']==b['summary'], b['summary'], '| arms identical:', a['scored']==b['scored'])"
python3 scripts/preflight_13c.py > /dev/null 2>&1 && echo "   preflight OK" || echo "   PREFLIGHT FAILED"
echo "8. genuine no-op and matched causal control replayed (d_rows.py on uP161 L20/D23, confirmation stream)"
DR_STREAM=88600000 DR_TAG=_verify DR_NSTREAMS=1 python3 scripts/d_rows.py uP161:fx:20:20:23 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/d_rows_conf.json'))['uP161:fx:20:20:23']['arms']; b=json.load(open('logs/d_rows_verify.json'))['uP161:fx:20:20:23']['arms']
print('   NOOP max|dlogit|', b['NOOP']['max_abs_dlogit_vs_unmodified'], '| max |diff| over arms (R2, slope):', max(max(abs(a[k]['R2']-b[k]['R2']), abs(a[k]['slope']-b[k]['slope'])) for k in a))"
rm -f logs/d_rows_verify.json
echo "9. representative joint-relation intervention replayed (c_causal.py, uP161 L20/D23 early01, confirmation stream)"
CC_STREAM=88500000 CC_TAG=_verify nice python3 scripts/c_causal.py uP161:fx:20:20:23 early01 > /dev/null 2>&1
python3 -c "
import json; k='uP161:fx:20:20:23|early01'; a=json.load(open('logs/c_causal_conf.json'))[k]; b=json.load(open('logs/c_causal_verify.json'))[k]
print('   DAS k3 archived/replay', round(a['C3']['das_QSRO']['k3']['test_fid']['slope'],4), round(b['C3']['das_QSRO']['k3']['test_fid']['slope'],4), '| m-probe QS', round(a['C3']['probe_QS']['interchange_m']['slope'],4), round(b['C3']['probe_QS']['interchange_m']['slope'],4), '| rows Shapley', round(a['C4']['shapley']['A'],4), round(b['C4']['shapley']['A'],4))"
rm -f logs/c_causal_verify.json
echo "10. representative adaptation comparison replayed (adapt3.py arms L2 and supG on uP161 L20/D23, confirmation stream)"
AD_STREAM=88000000 AD_OUT=_verify AD_ARMS=L2,supG nice python3 scripts/adapt3.py uP161:fx:20:20:23 > /dev/null 2>&1
python3 -c "
import json; t='uP161_fx_20_20_23_b0.6_lr0.001.json'; a=json.load(open('logs/adapt3_conf/'+t)); b=json.load(open('logs/adapt3_verify/'+t))
print('   L2 acc/J archived', round(a['L2']['acc'],4), round(a['L2']['J_DM']['slope'],4), 'replay', round(b['L2']['acc'],4), round(b['L2']['J_DM']['slope'],4), '| supG archived', round(a['supG']['acc'],4), round(a['supG']['J_DM']['slope'],4), 'replay', round(b['supG']['acc'],4), round(b['supG']['J_DM']['slope'],4))"
rm -rf logs/adapt3_verify; rm -f checkpoints/adapt3/*verify* 2>/dev/null
echo "11. principal uncertainty estimates recomputed (uncertainty_13c.py) and compared"
cp logs/uncertainty_13c.json /tmp/x13c_u.json; nice python3 scripts/uncertainty_13c.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x13c_u.json')); b=json.load(open('logs/uncertainty_13c.json')); print('   bootstrap identical:', a['boot']==b['boot'], '| R1 distance intervals identical:', a['R1_distances']==b['R1_distances'])"
echo "12. external dependencies (deps_inventory13c.py) + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/x13c_d.json; python3 scripts/deps_inventory13c.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/x13c_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
print("   software:", b["software"]); print("   external files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| missing:", b["missing"], "| sibling packages:", b["sibling_packages_required"])
PY
cp /tmp/x13c_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean13c.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files with absolute local paths: $n"
echo "== done (any FAILED / False / mismatches > 0 / non-finite > 0 is a failure)"; rm -rf $T
} 2>&1 | tee $LOG
