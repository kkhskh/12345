#!/bin/bash
# Clean-extraction verification of cor_inv13e_reroll_critical_period_and_causal_audit.zip (the 14 mandated steps).
# Usage: verify_clean13e.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_reroll_and_critical_period_validation
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XIII-E clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME into a clean directory and extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. full 64-run re-roll inventory + every new checkpoint loads with finite logits"
python3 - <<'PY'
import sys, os, glob, json, numpy as np, jax.numpy as jnp; sys.path.insert(0, "src"); import xc, ftlib as F, native as NV
D = json.load(open("logs/A_design.json")); miss = [r["tag"] for r in D["runs"] if not all(os.path.exists(f"checkpoints/ft/{r['tag']}/{f}") for f in ("stop_done.json", "step1211.npz", "opt_step1211.pkl"))]
print(f"   re-roll runs in design: {len(D['runs'])}, complete: {len(D['runs']) - len(miss)}, missing: {miss}")
X = NV.Bm.sample_fast(NV.ENV, 16, np.random.default_rng(1))["X"]; n = bad = 0
for d in sorted(glob.glob("checkpoints/ft/*")):
    t = os.path.basename(d); p, _, L, W, Dd = t.split("_")[:5]; suf = "_".join(t.split("_")[5:]); spec = f"{p}:fx:{L[1:]}:{W[1:]}:{Dd[1:]}" + (f":{suf}" if suf else "")
    for s in xc.steps_available(spec):
        n += 1
        try: ok = bool(np.isfinite(np.asarray(F.forward(xc.load(spec, s), jnp.asarray(X)))).all())
        except Exception: ok = False
        bad += not ok
print(f"   run directories: {len(glob.glob('checkpoints/ft/*'))}; checkpoints loaded: {n}; non-finite/failed: {bad}")
PY
echo "4+5. re-roll probabilities and principal statistical comparisons recomputed by the frozen fail-closed scorer"
cp logs/score_A.json /tmp/x13e_A.json; python3 scripts/score_A.py logs > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x13e_A.json')); b=json.load(open('logs/score_A.json')); print('   cells identical:', a['cells']==b['cells'], '| comparisons identical:', a['comparisons']==b['comparisons'], '| R1 identical:', a['R1']==b['R1'], b['summary'])"
echo "6. step-1211 classifications recomputed from checkpoints (unit_metrics on 4 re-roll runs, frozen stream) and compared"
python3 - <<'PY'
import json, subprocess, sys, os
D = json.load(open("logs/A_design.json")); specs = [D["runs"][i]["spec"] for i in (0, 17, 38, 63)]
env = dict(os.environ, UM_RHO="0", UM_STREAM="95100000", UM_TAG="_verifyA", UM_STEPS="1211"); subprocess.run([sys.executable, "scripts/unit_metrics.py"] + specs, env=env, capture_output=True)
a = {(json.loads(l)["unit"]): json.loads(l) for l in open("logs/unit_metrics_A.jsonl")}; b = {json.loads(l)["unit"]: json.loads(l) for l in open("logs/unit_metrics_verifyA.jsonl")}
print("   max |diff| lik/mix over 4 runs:", max(max(abs(a[s]["lik_RO1"] - b[s]["lik_RO1"]), abs(a[s]["mix_mass"] - b[s]["mix_mass"])) for s in specs)); os.remove("logs/unit_metrics_verifyA.jsonl")
PY
echo "7. 4000-step continuations: all present; 2 re-measured at 4000 and compared; continuation set re-derived by the frozen rule"
python3 - <<'PY'
import json, subprocess, sys, os
C = json.load(open("logs/manifest_A_cont.json"))["continuation_final"]; ok = all(os.path.exists(f"checkpoints/ft/{t.replace('_stop1211', '')}_br1211/final.json") for t in C); print("   continuations finished:", ok, len(C))
cp = "logs/manifest_A_cont.json"; old = json.load(open(cp)); subprocess.run([sys.executable, "scripts/select_continuation.py"], capture_output=True); print("   frozen rule reproduces the set:", json.load(open(cp))["continuation_final"] == old["continuation_final"])
D = json.load(open("logs/A_design.json")); specs = [next(r["spec"] for r in D["runs"] if r["tag"] == t).replace(":stop1211", ":br1211") for t in C[:2]]
env = dict(os.environ, UM_RHO="0", UM_STREAM="95100000", UM_TAG="_verifyC", UM_STEPS="4000"); subprocess.run([sys.executable, "scripts/unit_metrics.py"] + specs, env=env, capture_output=True)
a = {json.loads(l)["unit"]: json.loads(l) for l in open("logs/unit_metrics_Acont.jsonl")}; b = {json.loads(l)["unit"]: json.loads(l) for l in open("logs/unit_metrics_verifyC.jsonl")}
print("   max |diff| lik/mix:", max(max(abs(a[s]["lik_RO1"] - b[s]["lik_RO1"]), abs(a[s]["mix_mass"] - b[s]["mix_mass"])) for s in specs)); os.remove("logs/unit_metrics_verifyC.jsonl")
PY
echo "8. representative real-mask and sham runs replayed to step 150 and compared bit-for-bit with the archived checkpoints"
FX_STOP=150 FX_INTERV=ro0:100 FX_TAGSUF=_verify_ro0 nice python3 scripts/ft_factorial_b.py uP160 21 21 801 > /dev/null 2>&1
FX_STOP=150 FX_INTERV=sham:100 FX_TAGSUF=_verify_sham nice python3 scripts/ft_factorial_b.py uP160 23 23 801 > /dev/null 2>&1
python3 - <<'PY'
import numpy as np
for new, old in (("uP160_fx_L21_W21_D801_ro0T100_stop150_verify_ro0", "uP160_fx_L21_W21_D801_ro0T100_stop1211_B_ro0100"), ("uP160_fx_L23_W23_D801_shamT100_stop150_verify_sham", "uP160_fx_L23_W23_D801_shamT100_stop1211_B_sham100"), ("uP160_fx_L23_W23_D801_shamT100_stop150_verify_sham", "uP160_fx_L23_W23_D801_stop1211_B_base")):
    a, b = np.load(f"checkpoints/ft/{new}/step150.npz"), np.load(f"checkpoints/ft/{old}/step150.npz"); print("   ", new, "vs", old, "max |diff|", max(float(np.abs(a[k] - b[k]).max()) for k in a.files))
PY
rm -rf checkpoints/ft/*_verify_*
echo "9. ideal-model metric references recomputed (c1_ideal_refs.py) and compared"
cp logs/c1_ideal_refs.json /tmp/x13e_c1.json; nice python3 scripts/c1_ideal_refs.py > /dev/null 2>&1
python3 -c "import json; print('   identical:', json.load(open('/tmp/x13e_c1.json'))==json.load(open('logs/c1_ideal_refs.json')))"
echo "10+11. decoder, principal-angle and k-sweep analyses recomputed on C-C2 (c_audit_v.py, k = 1, 2, 3) and compared"
CA_CKDIR=adapt4_carrier CA_STREAM=96200000 CA_KS=1,2,3 nice python3 scripts/c_audit_v.py uP165:s:21 > /dev/null 2>&1
python3 - <<'PY'
import json
a = json.load(open("logs/c_audit.json"))["uP165:s:21"]; b = json.load(open("logs/c_audit_verify.json"))["uP165:s:21"]
print("   C2 decoding identical:", a["C2"]["decoding"] == b["C2"]["decoding"], "| C2 geometry identical:", a["C2"]["geometry"] == b["C2"]["geometry"], "| C3 identical:", a["C3"] == b["C3"])
print("   C4 k=1..3 identical:", all(a["C4"][k] == b["C4"][k] for k in ("1", "2", "3")))
PY
rm -f logs/c_audit_verify.json
echo "12. frozen prediction verdicts reproduced (score_B, score_C; score_A in step 4) and all preflights rerun"
for f in B C; do cp logs/score_$f.json /tmp/x13e_$f.json; python3 scripts/score_$f.py logs > /dev/null 2>&1; python3 -c "import json; a=json.load(open('/tmp/x13e_$f.json')); b=json.load(open('logs/score_$f.json')); print('   score_$f identical:', a==b, b['summary'])"; done
for f in A B C; do python3 scripts/preflight_$f.py > /dev/null 2>&1 && echo "   preflight $f OK" || echo "   PREFLIGHT $f FAILED"; done
echo "13. incomplete scoring is refused (each scorer on a copy of logs/ with one required file removed)"
for pair in "A unit_metrics_Acont.jsonl" "B unit_metrics_Bfinal.jsonl" "C c_audit.json"; do set -- $pair; X=$(mktemp -d); cp -r logs/* $X/; rm $X/$2; python3 scripts/score_$1.py $X > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/score_$1.json.new ] && echo "   score_$1 without $2: refused (rc=$rc)" || echo "   score_$1 without $2: NOT REFUSED (rc=$rc)"; rm -rf $X; done
echo "14. external dependencies by exact hash (deps_inventory13e.py) + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/x13e_d.json; python3 scripts/deps_inventory13e.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/x13e_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
mp = [k for k in a["external_python_modules"] if b["external_python_modules"].get(k) != a["external_python_modules"][k]]
print("   external files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| module mismatches:", len(mp), "| missing:", b["missing"]); print("   sibling packages required:", b["sibling_packages_required"])
PY
cp /tmp/x13e_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean13e.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files with absolute local paths: $n"
echo "== done (any FAILED / False / NOT REFUSED / mismatches > 0 / non-finite > 0 / nonzero diff is a failure)"; rm -rf $T
} 2>&1 | tee $LOG
