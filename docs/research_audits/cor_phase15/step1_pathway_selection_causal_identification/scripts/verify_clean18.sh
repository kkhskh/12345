#!/bin/bash
# Clean-extraction verification of cor_inv18_pathway_selection_causal_identification.zip (mandate §14).
# Usage: verify_clean18.sh ZIP COR_PHASE15_ROOT LOG.   Not standalone: sibling packages listed in logs/DEPENDENCIES.json are copied first.
set -u; ZIP=$1; SRC=$2; LOG=$3; T=$(mktemp -d); NAME=step1_pathway_selection_causal_identification
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
{
echo "== Investigation XVIII clean extraction check $(date -u +%FT%TZ)"
echo "1. sibling packages (not standalone) + extraction"
mkdir -p $T/cor_phase15; for b in $(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['sibling_packages_required']))" $SRC/$NAME/logs/DEPENDENCIES.json); do cp -r $SRC/$b $T/cor_phase15/; done
(cd $T/cor_phase15 && unzip -q $ZIP) && echo "   extracted OK" || echo "   EXTRACT FAILED"; P=$T/cor_phase15/$NAME; cd $P || exit 1
echo "2. manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "   manifest OK ($(wc -l < MANIFEST.sha256) files)" || echo "   MANIFEST FAILED"
echo "3. frozen files unchanged since the freeze (logs/FREEZE_XVIII_HASHES.txt)"; sha256sum -c --quiet logs/FREEZE_XVIII_HASHES.txt && echo "   freeze hashes OK" || echo "   FREEZE HASH MISMATCH"
echo "4. dependency hashes"; cp logs/DEPENDENCIES.json /tmp/x18_d.json; python3 scripts/deps_inventory18.py > /dev/null 2>&1
python3 -c "import json; a=json.load(open('/tmp/x18_d.json')); b=json.load(open('logs/DEPENDENCIES.json')); print('   identical:', a==b)"; cp /tmp/x18_d.json logs/DEPENDENCIES.json
echo "5. exact intervention masks, tensor locations and mode semantics (unit checks re-run)"
for u in unit_modes unit_modes2 unit_modes3 unit_sha_far; do cp logs/$u.json /tmp/x18_$u.json; python3 scripts/$u.py > /dev/null 2>&1
  python3 -c "import json,sys; a=json.load(open('/tmp/x18_$u.json')); b=json.load(open('logs/$u.json')); print('   $u identical:', a==b)"; cp /tmp/x18_$u.json logs/$u.json; done
python3 - <<'PY'
import sys; sys.path.insert(0, "../step1_cross_family_relational_foundations/src"); import maskfwd as M, numpy as np
for k in ("own0", "own2", "ro0"):
    l, m = M.edges(k); print(f"   edges {k}: layer {l}, {int(m.sum())} edges, rows {sorted(set(np.nonzero(m)[0].tolist()))}, own-q only: {all(((j // 6) == (i - 36)) for i, j in zip(*np.nonzero(m))) if k != 'ro0' else 'n/a'}")
PY
echo "6. baseline replication: bitwise replay of a confirmation baseline (E, D=811) to step 150"
rm -rf checkpoints/ft/uP160_fx_L21_W21_D811_stop150_vrf; FX_SLOG=own0 FX_SLOGT=100 FX_STOP=150 FX_TAGSUF=_vrf python3 scripts/xviii_train.py uP160 21 21 811 > /dev/null 2>&1
python3 -c "import numpy as np; a=np.load('checkpoints/ft/uP160_fx_L21_W21_D811_stop150_vrf/step150.npz'); b=np.load('checkpoints/ft/uP160_fx_L21_W21_D811_stop1211_c18/step150.npz'); print('   max |diff|:', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))"
echo "   and of an intervention run (vrg, M, D=812) to step 150"
FX_INTERV=vrg:100 FX_EDGES=own2 FX_GLOG=1 FX_STOP=150 FX_TAGSUF=_vrf python3 scripts/xviii_train.py uP160 23 23 812 > /dev/null 2>&1
python3 -c "import numpy as np; a=np.load('checkpoints/ft/uP160_fx_L23_W23_D812_vrgown2T100_stop150_vrf/step150.npz'); b=np.load('checkpoints/ft/uP160_fx_L23_W23_D812_vrgown2T100_stop1211_c18/step150.npz'); print('   max |diff|:', max(float(np.abs(a[k]-b[k]).max()) for k in a.files))"
rm -rf checkpoints/ft/*_vrf
echo "7. training schedule"; python3 -c "
import optax; s=optax.warmup_cosine_decay_schedule(0.0,1e-3,200,4000); print('   lr at 0/50/100/200/1211/2546:', [round(float(s(i)),6) for i in (0,50,100,200,1211,2546)])"
echo "8. model independence: parents, seeds and streams"
python3 - <<'PY'
import json; d = json.load(open("logs/DEPENDENCIES.json"))["external_data_files"]["parents_and_eval"]
print("   distinct parent hashes:", len({v for k, v in d.items() if "pre.npz" in k}), "of", sum("pre.npz" in k for k in d))
print("   streams: confirmation 95400000, screening 95500000, parents 95600000, development 86100000; orders: confirmation 811-813, screening 811, parents 812")
PY
echo "9. organisational classification and competence recomputed for two confirmation runs"
python3 - <<'PY'
import json, sys; sys.path.insert(0, "src"); import xviiim as XM
R = {json.loads(l)["run"]: json.loads(l) for l in open("logs/measure_c18.jsonl")}
for t in ("uP160_fx_L21_W21_D811_stop1211_c18", "uP160_fx_L23_W23_D813_gsvown2T100_stop1211_c18"):
    r = XM.measure_v2(XM.load_dir("checkpoints/ft/" + t, "uP160", 1211), 95_400_000); a = R[t]["final"]["1211"]
    print("   ", t, "max |diff| over numeric fields:", max(abs(r[k] - a[k]) for k in a if isinstance(a[k], float)), "| classes equal:", all(r[k] == a[k] for k in a if isinstance(a[k], str)))
PY
echo "10. immediate H3 metric recomputed"; cp logs/h3_immediate_c18.json /tmp/x18_h3.json; python3 scripts/h3_immediate18.py logs/XVIII_design.json /tmp/x18_h3b.json > /dev/null 2>&1
python3 -c "import json; print('   identical:', json.load(open('/tmp/x18_h3.json'))==json.load(open('/tmp/x18_h3b.json')))"
echo "11. statistics and verdicts recomputed by the fail-closed scorers"
for s in score_XVIII score_cont18 score_parents18; do f=logs/$s.json; [ "$s" = score_parents18 ] || true; [ -f $f ] || { echo "   $s: no output archived"; continue; }; cp $f /tmp/x18_$s.json; python3 scripts/$s.py > /dev/null 2>&1
  python3 -c "import json; print('   $s identical:', json.load(open('/tmp/x18_$s.json'))==json.load(open('$f')))"; done
echo "12. fail-closed: preflight, and refusal on the real data with one measurement removed"
cp logs/preflight_XVIII.json /tmp/x18_pf.json; python3 scripts/preflight_XVIII.py > /dev/null 2>&1; python3 -c "import json; print('   preflight all_pass:', json.load(open('logs/preflight_XVIII.json'))['all_pass'])"; cp /tmp/x18_pf.json logs/preflight_XVIII.json
X=$(mktemp -d); tail -n +2 logs/measure_c18.jsonl > $X/m.jsonl; cp logs/h3_immediate_c18.json $X/; python3 scripts/score_XVIII.py logs/XVIII_design.json $X/m.jsonl $X/o.json > /dev/null 2>&1; rc=$?
[ $rc -eq 2 ] && [ ! -f $X/o.json ] && echo "   refused (rc=$rc)" || echo "   NOT REFUSED (rc=$rc)"; rm -rf $X
echo "13. negative and incomplete runs: every designed run is measured"
python3 - <<'PY'
import json; d = json.load(open("logs/XVIII_design.json")); m = {json.loads(l)["run"] for l in open("logs/measure_c18.jsonl")}
miss = [v["tag"] for v in d["runs"].values() if v["tag"] not in m]; print("   confirmation:", len(d["runs"]), "designed,", len(miss), "unmeasured", miss)
try:
    P = json.load(open("logs/XVIII_parents.json")); mp = {json.loads(l)["run"] for l in open("logs/measure_p18.jsonl")}
    runs = [t for p in P["parents"].values() for t in p.get("runs", {}).values()]; print("   parents:", len(runs), "designed,", sum(t not in mp for t in runs), "unmeasured;", {k: v["bistable"] for k, v in P["parents"].items()})
except FileNotFoundError as e: print("   parents:", e)
PY
echo "14. reviews present and reports consistent with the scores"; ls reviews/*.md | sed 's/^/   /'; python3 scripts/check_reports18.py
echo "== done"; rm -rf $T /tmp/x18_*
} > $LOG 2>&1
