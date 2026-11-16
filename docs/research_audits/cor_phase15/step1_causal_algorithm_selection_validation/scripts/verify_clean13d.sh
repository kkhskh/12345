#!/bin/bash
# Clean-extraction verification of cor_inv13d_causal_algorithm_selection_validation.zip (the 13 mandated steps).
# Usage: verify_clean13d.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_causal_algorithm_selection_validation
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XIII-D clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME into a clean directory and extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. every new checkpoint loads and gives finite logits (all saved steps of every ft run; every adapt4 parameter file)"
python3 - <<'PY'
import sys, os, glob, numpy as np, jax.numpy as jnp; sys.path.insert(0, "src"); import xc, ftlib as F, native as NV
X = NV.Bm.sample_fast(NV.ENV, 32, np.random.default_rng(1))["X"]; bad = n = 0
for d in sorted(glob.glob("checkpoints/ft/*")):
    t = os.path.basename(d); p, _, L, W, D = t.split("_")[:5]; suf = "_".join(t.split("_")[5:]); spec = f"{p}:fx:{L[1:]}:{W[1:]}:{D[1:]}" + (f":{suf}" if suf else "")
    st = xc.steps_available(spec)
    for s in st:
        n += 1
        try: ok = bool(np.isfinite(np.asarray(F.forward(xc.load(spec, s), jnp.asarray(X)))).all())
        except Exception: ok = False
        bad += (not ok)
    print(f"   {t}: final.json {os.path.exists(d + '/final.json')} | {len(st)} checkpoints")
za = glob.glob("checkpoints/adapt4*/*.npz"); badz = sum(1 for f in za if not all(np.isfinite(v).all() for v in np.load(f).values()))
print(f"   ft checkpoints loaded: {n}, non-finite/failed: {bad} | adapt4 parameter files: {len(za)}, non-finite: {badz}")
PY
echo "4. frozen verdicts recomputed from the archived raw outputs and compared; all preflights rerun"
for f in f1 f2 f3a f3b; do cp logs/score_$f.json /tmp/x13d_s_$f.json; python3 scripts/score_$f.py logs > /dev/null 2>&1
  python3 -c "import json; a=json.load(open('/tmp/x13d_s_$f.json')); b=json.load(open('logs/score_$f.json')); print('   $f verdicts identical:', a['summary']==b['summary'], b['summary'], '| arms identical:', a['scored']==b['scored'])"; done
for f in f1 f2 f3a f3b; do python3 scripts/preflight_$f.py > /dev/null 2>&1 && echo "   preflight $f OK" || echo "   PREFLIGHT $f FAILED"; done
echo "5. principal statistics recomputed (uncertainty_13d.py) and compared"
cp logs/uncertainty_13d.json /tmp/x13d_u.json; nice python3 scripts/uncertainty_13d.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x13d_u.json')); b=json.load(open('logs/uncertainty_13d.json')); print('   effects identical:', a['effects']==b['effects'], '| sensitivity identical:', a['sensitivity']==b['sensitivity'], '| factorial identical:', a['factorial']==b['factorial'])"
echo "6. mathematical results: XIII-C posterior/sufficiency verification rerun from the sibling package and compared"
(cd ../step1_critical_relational_mechanism_resolution && cp logs/c_math.json /tmp/x13d_cm.json && nice python3 scripts/c_math.py > /dev/null 2>&1 && python3 -c "import json; print('   c_math identical:', json.load(open('/tmp/x13d_cm.json'))==json.load(open('logs/c_math.json')))"; cp /tmp/x13d_cm.json logs/c_math.json)
echo "7. pathway-selection replay: F1c (uP160 23/23/23, l2q mask 0-100) retrained to step 246 and compared bit-for-bit with the archived checkpoints"
FX_STOP=246 FX_INTERV=l2q:100 FX_TAGSUF=_verify nice python3 scripts/ft_factorial.py uP160 23 23 23 > /dev/null 2>&1
python3 -c "
import numpy as np, glob
for s in (25, 50, 100, 111, 246):
    a=np.load('checkpoints/ft/uP160_fx_L23_W23_D23_l2qT100_stop246_verify/step%d.npz'%s); b=np.load('checkpoints/ft/uP160_fx_L23_W23_D23_l2qT100/step%d.npz'%s)
    print('   step', s, 'max |diff|', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))"
rm -rf checkpoints/ft/uP160_fx_L23_W23_D23_l2qT100_stop246_verify
echo "8+9. repair-carrier intervention, natural-donor necessity and null-pair test replayed (c_carrier.py on C-C1, confirmation stream)"
CA_TAG=_verify CA_STREAM=92200000 CA_CKDIR=adapt4_carrier nice python3 scripts/c_carrier.py uP160:s:23 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/c_carrier_f2.json'))['uP160:s:23']['sets']; b=json.load(open('logs/c_carrier_verify.json'))['uP160:s:23']['sets']
for k in ('DAS','ENC_m','DEC_m'): print('   ', k, 'SUFF archived/replay', round(a[k]['SUFF']['slope'],5), round(b[k]['SUFF']['slope'],5))
print('    DAS reverse archived/replay', round(a['DAS']['NEC']['reverse']['remaining_slope'],5), round(b['DAS']['NEC']['reverse']['remaining_slope'],5), '| null D0', round(a['DAS']['NULL_JD0']['rel_to_JDM_patch'],5), round(b['DAS']['NULL_JD0']['rel_to_JDM_patch'],5))"
rm -f logs/c_carrier_verify.json
echo "10. supplied-information adaptation replayed (adapt4 arm supRO, lr 3e-3, uP165 s20)"
AD_OUT=_verify AD_LR=0.003 AD_ARMS=supRO nice python3 scripts/adapt4.py uP165:s:20 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/adapt4_f3a/uP165_s_20_b0.6_lr0.003.json'))['supRO']; b=json.load(open('logs/adapt4_verify/uP165_s_20_b0.6_lr0.003.json'))['supRO']
print('   supRO acc/J archived', round(a['acc'],5), round(a['J_DM']['slope'],5), 'replay', round(b['acc'],5), round(b['J_DM']['slope'],5))"
rm -rf logs/adapt4_verify checkpoints/adapt4_verify
echo "11. corrected no-op and attention-row controls replayed (d_rows.py on uP165 s23, confirmation stream)"
DR_TAG=_verify DR_STREAM=92600000 DR_NSTREAMS=1 nice python3 scripts/d_rows.py uP165:s:23 > /dev/null 2>&1
python3 -c "
import json; a=json.load(open('logs/d_rows_f2.json'))['uP165:s:23']['arms']; b=json.load(open('logs/d_rows_verify.json'))['uP165:s:23']['arms']
print('   NOOP max|dlogit|', b['NOOP']['max_abs_dlogit_vs_unmodified'], 'within tol', b['NOOP']['within_tolerance'], '| SHUFD_vs_borrowed R2 archived/replay', round(a['SHUFD_vs_borrowed']['R2'],5), round(b['SHUFD_vs_borrowed']['R2'],5), '| max |diff| over arms', max(abs(a[k]['R2']-b[k]['R2']) for k in a))"
rm -f logs/d_rows_verify.json
echo "12. full-length determinism test validated: archived result + direct re-comparison of every common checkpoint of the uP163 replay"
python3 - <<'PY'
import json, os, sys, numpy as np; sys.path.insert(0, "src"); import native as NV
d = json.load(open("logs/determinism_check.json")); print("   archived:", {k: (v["bitwise_all_steps"], len(v["common_steps"])) for k, v in d.items()})
r = "checkpoints/ft/uP163_fx_L21_W22_D21_rep"; o = os.path.join(NV.X13C, "checkpoints", "ft", "uP163_fx_L21_W22_D21")
st = sorted(set(f for f in os.listdir(r) if f.startswith("step")) & set(f for f in os.listdir(o) if f.startswith("step")))
mx = max(max(float(np.abs(np.load(os.path.join(r, f))[k] - np.load(os.path.join(o, f))[k]).max()) for k in np.load(os.path.join(r, f)).files) for f in st)
print(f"   recomputed: {len(st)} common checkpoints, max |param diff| = {mx}")
PY
echo "13. external dependencies (deps_inventory13d.py) by exact hashes + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/x13d_d.json; python3 scripts/deps_inventory13d.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/x13d_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
mp = [k for k in a["external_python_modules"] if b["external_python_modules"].get(k) != a["external_python_modules"][k]]
print("   software:", b["software"]); print("   external files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| module mismatches:", len(mp), "| missing:", b["missing"])
print("   sibling packages required:", b["sibling_packages_required"])
PY
cp /tmp/x13d_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean13d.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files with absolute local paths: $n"
echo "== done (any FAILED / False / mismatches > 0 / non-finite > 0 / nonzero diff is a failure)"; rm -rf $T
} 2>&1 | tee $LOG
