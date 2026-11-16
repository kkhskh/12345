#!/bin/bash
# Clean-extraction verification of cor_inv14_cross_family_relational_foundations.zip (the 13 mandated steps).
# Usage: verify_clean14.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_cross_family_relational_foundations
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XIV clean extraction check $(date -u +%FT%TZ)"
echo "1. copy cor_phase15 WITHOUT $NAME into a clean directory and extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3+4. T4 controls and early-vs-late persistence recomputed by the frozen fail-closed scorer; 3 units re-measured from checkpoints"
cp logs/score_I.json /tmp/x14_I.json; python3 scripts/score_I.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x14_I.json')); b=json.load(open('logs/score_I.json')); print('   score_I identical:', a==b, b['verdicts'])"
python3 - <<'PY'
import json, subprocess, sys, os
D = json.load(open("logs/I_design.json")); pick = [m for m in D["measurements"] if "msk" in m["unit"]][:1] + [m for m in D["measurements"] if "lnz" in m["unit"]][:1] + [m for m in D["measurements"] if "w546-646" in m["unit"] and m["step"] == 1757][:1]
a = {(json.loads(l)["unit"], json.loads(l)["step"]): json.loads(l) for l in open("logs/unit_metrics_I.jsonl")}
for m in pick:
    env = dict(os.environ, UM_RHO="0", UM_STREAM="95300000", UM_TAG="_verifyI", UM_STEPS=str(m["step"])); subprocess.run([sys.executable, "scripts/unit_metrics.py", m["unit"]], env=env, capture_output=True)
b = {(json.loads(l)["unit"], json.loads(l)["step"]): json.loads(l) for l in open("logs/unit_metrics_verifyI.jsonl")}; os.remove("logs/unit_metrics_verifyI.jsonl")
print("   re-measured units:", len(b), "| max |diff| mix/lik/KL:", max(max(abs(a[k][f] - b[k][f]) for f in ("mix_mass", "lik_RO1", "KL")) for k in b))
PY
echo "5. evidence ledger: every package pointer exists and every quoted anchor number occurs in its cited source"
python3 scripts/ledger_check.py
echo "6. key mathematical derivations recomputed (exact rules, interaction-order ceilings, invariances)"
for f in g_rules_exact g_order_ceilings math_verify; do cp logs/$f.json /tmp/x14_$f.json; done
python3 scripts/g_rules_exact.py > /dev/null 2>&1; python3 scripts/g_order_ceilings.py > /dev/null 2>&1; python3 scripts/math_verify.py > /dev/null 2>&1
python3 - <<'PY'
import json
def close(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float): return abs(a - b) <= 1e-9 * max(1, abs(a)) or (a != a and b != b)
    return a == b
for f in ("g_rules_exact", "g_order_ceilings", "math_verify"): print(f"   {f} reproduced:", close(json.load(open(f"/tmp/x14_{f}.json")), json.load(open(f"logs/{f}.json"))))
PY
echo "7+9. representative GATE relational intervention replayed (g_mech.py on s1: patching, presence, mediation, null and decoy controls)"
cp logs/gate_mech/s1.json /tmp/x14_m1.json; python3 scripts/g_mech.py s1 ../step1_relational_mechanism_deepening/checkpoints/gate_s1.npz 98200000 logs/gate_mech/s1.json > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x14_m1.json')); b=json.load(open('logs/gate_mech/s1.json')); a.pop('ckpt'); b.pop('ckpt'); print('   g_mech s1 identical:', a==b, '| null donor', b['mediation']['null_donor_decision_change'], '| decoy dir', b['mediation']['decoy_direction'], '| random', b['mediation']['random_vm_max'])"
echo "8+10. frozen cross-family prediction X1 and its statistics recomputed (score_X.py)"
cp logs/score_X.json /tmp/x14_X.json; python3 scripts/score_X.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x14_X.json')); b=json.load(open('logs/score_X.json')); print('   score_X identical:', a==b, b['verdicts'])"
echo "11. every new checkpoint loads with finite outputs; required source files present"
python3 - <<'PY'
import sys, os, glob, numpy as np, jax.numpy as jnp
sys.path.insert(0, "src"); import xc, ftlib as F, native as NV, gatelib as GL
X = NV.Bm.sample_fast(NV.ENV, 16, np.random.default_rng(1))["X"]; n = bad = 0
for d in sorted(glob.glob("checkpoints/ft/*")):
    t = os.path.basename(d); p = t.split("_"); spec = f"{p[0]}:fx:{p[2][1:]}:{p[3][1:]}:{p[4][1:]}" + (":" + "_".join(p[5:]) if len(p) > 5 else "")
    for s in xc.steps_available(spec):
        n += 1
        try: bad += not bool(np.isfinite(np.asarray(F.forward(xc.load(spec, s), jnp.asarray(X)))).all())
        except Exception: bad += 1
Xg = GL.TX.sample_gate(16, np.random.default_rng(1))["X"]; ng = badg = 0
for f in sorted(glob.glob("checkpoints/gate/*/step*.npz")):
    ng += 1
    try: badg += not bool(np.isfinite(GL.probs(GL.load(f), Xg)).all())
    except Exception: badg += 1
req = ["src/gatelib.py", "src/maskfwd.py", "scripts/ft_factorial_c.py", "scripts/g_train.py", "scripts/score_I.py", "scripts/score_X.py", "logs/FREEZE_I.md", "logs/FREEZE_X.md", "T4_FOUNDATIONAL_EVIDENCE_LEDGER.md", "MAIN_RESEARCH_SYNTHESIS.md"]
print(f"   T4 checkpoints loaded: {n}, failed/non-finite: {bad}; GATE snapshots loaded: {ng}, failed/non-finite: {badg}; required files missing: {[r for r in req if not os.path.exists(r)]}")
PY
echo "12. fail-closed scoring: each scorer on an input with one required item removed"
X=$(mktemp -d); cp -r logs/* $X/; rm $X/unit_metrics_I.jsonl; python3 scripts/score_I.py $X > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/score_I.json.new ] && echo "   score_I without unit_metrics_I.jsonl: refused (rc=$rc)" || echo "   score_I: NOT REFUSED (rc=$rc)"; rm -rf $X
X=$(mktemp -d); cp -r checkpoints/gate/s3 checkpoints/gate/s4 $X/; rm $X/s3/step1000.npz; python3 scripts/score_X.py $X $X/out.json > /dev/null 2>&1; rc=$?; [ $rc -eq 2 ] && [ ! -f $X/out.json ] && echo "   score_X without one snapshot: refused (rc=$rc)" || echo "   score_X: NOT REFUSED (rc=$rc)"; rm -rf $X
for f in I X; do python3 scripts/preflight_$f.py > /dev/null 2>&1 && echo "   preflight $f OK" || echo "   PREFLIGHT $f FAILED"; done
echo "13. external dependencies by exact hash (deps_inventory14.py) + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/x14_d.json; python3 scripts/deps_inventory14.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/x14_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
mp = [k for k in a["external_python_modules"] if b["external_python_modules"].get(k) != a["external_python_modules"][k]]
print("   external files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| module mismatches:", len(mp), "| missing:", b["missing"]); print("   sibling packages required:", b["sibling_packages_required"])
PY
cp /tmp/x14_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean14.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files with absolute local paths: $n"
echo "== done (any FAILED / False / NOT REFUSED / mismatches > 0 / non-finite > 0 / nonzero diff is a failure)"; rm -rf $T
} 2>&1 | tee $LOG
