#!/bin/bash
# Clean-extraction verification of cor_inv13_relational_decision_computation.zip (10 steps, REPRODUCTION.md).
# Usage: verify_clean13.sh ZIP COR_PHASE15_ROOT LOGFILE
set -u
ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_relational_decision_computation
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XIII clean extraction check $(date -u +%FT%TZ)"
echo "1. copy the cor_phase15 tree WITHOUT $NAME to $T/cor_phase15, extract the ZIP there"
mkdir -p $T/cor_phase15; for d in $SRC/*; do b=$(basename $d); [ "$b" = "$NAME" ] && continue; cp -r $d $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"
P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. integrity manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. mathematics rerun (verify_shared_posterior.py exhaustive + MC; sufficiency_boundary.py)"
for s in verify_shared_posterior sufficiency_boundary; do cp logs/$s.json /tmp/x13_$s.json; nice python3 scripts/$s.py > /dev/null 2>&1
  python3 -c "import json; a=json.load(open('/tmp/x13_$s.json')); b=json.load(open('logs/$s.json')); a.pop('seconds',None); b.pop('seconds',None); print('   $s.json', 'identical' if json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True) else 'DIFFERENT')"; done
python3 -c "import json; r=json.load(open('logs/verify_shared_posterior.json'))['exhaustive_Q2']; print('   max |posterior_shared - brute-force latent sum| over all regimes:', max(v['shared_max_abs_err'] for v in r.values()))"
echo "4. frozen-prediction verdicts re-derived from archived raw outputs (score_xiii.py, frozen)"
cp logs/score_xiii.json /tmp/x13_s.json; python3 scripts/score_xiii.py 2>&1 | sed 's/^/   /'
python3 -c "import json; a=json.load(open('/tmp/x13_s.json')); b=json.load(open('logs/score_xiii.json')); print('   verdicts identical:', {k:v['verdict'] for k,v in a['predictions'].items()}=={k:v['verdict'] for k,v in b['predictions'].items()}, '| checks identical:', a['checks']==b['checks'])"
echo "5. statistics recomputed (stats_xiii.py; report_xiii.py incl. two-sample DiD bootstrap, fixed seed)"
cp logs/stats_xiii_conf.json /tmp/x13_st.json; cp logs/report_xiii_conf.json /tmp/x13_r.json; python3 scripts/stats_xiii.py > /dev/null 2>&1; python3 scripts/report_xiii.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x13_st.json')); b=json.load(open('logs/stats_xiii_conf.json')); c=json.load(open('/tmp/x13_r.json')); d=json.load(open('logs/report_xiii_conf.json')); print('   stats identical:', a==b, '| report identical:', c==d)"
echo "6. replay of a causal intervention (decoy_disc on the first complete fresh parent, confirmation stream 74_200_000)"
PP=$(python3 -c "import json; print(json.load(open('logs/score_xiii.json'))['complete'][0])"); PN=$(echo $PP | cut -c2-5)
DD_STREAM=74200000 DD_TAG=_verify python3 scripts/decoy_disc.py logs/decoy_conf $PP > /dev/null 2>&1
python3 - <<PY
import json
a = json.load(open("logs/decoy_disc_conf.json"))["$PP"]; b = json.load(open("logs/decoy_disc_verify.json"))["$PP"]
d = max(abs(a[s][v]["R2"] - b[s][v]["R2"]) for s in a for v in ("E", "A1", "PERM")); print("   $PP decoy_disc max |R2 diff| vs archived:", d)
PY
rm -f logs/decoy_disc_verify.json
echo "7. Bayes references: posterior_shared (C=1) vs closed form on the confirmation evaluation set; C=2 shared vs independent differ"
python3 - <<'PY'
import sys, numpy as np; sys.path.insert(0, "src"); import genlib as GL, xii_bayes as XB
d = np.load("logs/eval_conf13.npz", allow_pickle=True); X, q = d["X"], d["q"]; env = GL.Env(3, 1)
print("   C=1 max |shared - closed|:", float(np.abs(XB.posterior_shared(env, X, q) - GL.posterior_closed(env, X, q)).max()))
e2 = GL.Env(3, 2); r = np.random.default_rng(1); dd = GL.sample(e2, 2000, r); qq = r.integers(0, 6, 2000)
print("   C=2 mean |shared - independent|:", float(np.abs(XB.posterior_shared(e2, dd["X"], qq) - GL.posterior_brute(e2, dd["X"], qq)).sum(-1).mean()))
PY
echo "8. checkpoints: every cohort parent and fine-tune present and loadable"
python3 - <<'PY'
import sys, os, json, numpy as np, jax.numpy as jnp; sys.path.insert(0, "src"); import ftlib as F, native as NV
for p in ("P160", "P161", "P162", "P163", "P164", "P165"):
    g = f"checkpoints/z{p}/gate.json"; ok_p = os.path.exists(f"checkpoints/z{p}/pre.npz")
    try: M = F.load_model("u" + p, "full_c1", seed=20, suffix="_eval_conf13"); L = np.asarray(F.forward(M, jnp.asarray(NV.load_eval("logs/eval_conf13.npz")["X"][:50]))); ok_m = bool(np.isfinite(L).all())
    except Exception as ex: ok_m = f"FAILED {ex}"
    print(f"   {p}: parent {ok_p} gate {json.load(open(g))['complete'] if os.path.exists(g) else 'MISSING'} fine-tune loadable {ok_m}")
PY
echo "9. adaptation result re-run (adapt_sel.py, readout+attn conditions, first complete parent; deterministic seeds)"
AD_CONDS=readout,attn AD_OUT=adapt_sel_verify AD_STREAM=74600000 python3 scripts/adapt_sel.py - $PP > /dev/null 2>&1
python3 - <<PY
import json, glob
t = "$PP".split(":"); tag = f"{t[0]}_{t[1]}_s4000_seed{t[2]}{t[3]}"
a = json.load(open(f"logs/adapt_sel_conf/{tag}.json")); b = json.load(open(f"logs/adapt_sel_verify/{tag}.json"))
print("   readout/attn acc archived vs rerun:", [(c, round(a[c]["acc"], 4), round(b[c]["acc"], 4)) for c in ("readout", "attn")], "| native identical:", a["native"]["acc"] == b["native"]["acc"])
PY
rm -rf logs/adapt_sel_verify
echo "10. dependency versions and hashes (deps_inventory13.py) + absolute-path scan"
cp logs/DEPENDENCIES.json /tmp/x13_d.json; python3 scripts/deps_inventory13.py > /dev/null 2>&1
python3 - <<'PY'
import json
a, b = json.load(open("/tmp/x13_d.json")), json.load(open("logs/DEPENDENCIES.json"))
mm = [k for k in a["external_data_files"] if b["external_data_files"].get(k) != a["external_data_files"][k]]
print("   software:", b["software"]); print("   external data files:", len(b["external_data_files"]), "| hash mismatches:", len(mm), "| missing:", b["missing"], "| modules identical:", a["external_python_modules"] == b["external_python_modules"])
PY
cp /tmp/x13_d.json logs/DEPENDENCIES.json
n=$(grep -rl --exclude=verify_clean13.sh "/home/claude\|/mnt/user-data" scripts src 2>/dev/null | wc -l); echo "   scripts/src files containing absolute local paths: $n"
echo "== done (any line marked FAILED / DIFFERENT / mismatches > 0 is a failure)"; rm -rf $T
} 2>&1 | tee $LOG
