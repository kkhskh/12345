#!/bin/bash
# Clean-extraction check for Investigation V.  Exports the committed tree of docs/research_audits/cor_phase15 (git archive
# HEAD; the companion packages step1_latent_reliability, step1_relational_computation_atlas, step1_foundational_atlas_continuation,
# step1_relational_mechanism_deepening and step1_cross_implementation_causal_equivalence are read-only dependencies), then
#  [1] checks MANIFEST.sha256;
#  [2] FORWARD-PASS reproduction: re-runs scripts/cri_run.py conf on cT61 and cP61 from their committed checkpoints in the extracted
#      copy and compares the regenerated model_*.json (quality and operator fits) with the committed ones (tol 1e-6);
#  [3] REPLAY (not an independent forward pass): re-runs scripts/cri_stats.py on the committed response arrays and compares the
#      primary statistics (tol 1e-9).
set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); REL=docs/research_audits/cor_phase15
OUT=${1:-/tmp/cor_inv5_clean}; rm -rf "$OUT"; mkdir -p "$OUT"; git -C "$REPO" archive HEAD $REL | tar -x -C "$OUT"
C="$OUT/$REL/step1_curriculum_response_identification"; cd "$C"; mkdir -p verify
echo "[1] manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "    manifest OK"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cmp() { python3 - "$1" "$2" "$3" <<'PY'
import json, sys, math
a, b, tol = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), float(sys.argv[3]); bad = []
def c(x, y, p):
    if isinstance(y, dict): [c(x.get(k), y[k], p + "/" + k) for k in y]
    elif isinstance(y, list): [c(u, v, f"{p}[{i}]") for i, (u, v) in enumerate(zip(x, y))]
    elif isinstance(y, (int, float)) and not isinstance(y, bool):
        if not (math.isclose(x, y, rel_tol=tol, abs_tol=tol) or (math.isnan(x) and math.isnan(y))): bad.append(f"{p}: {x} vs {y}")
    elif x != y: bad.append(f"{p}: {x} vs {y}")
c(a, b, ""); print(("    OK   " if not bad else "    DIFF ") + sys.argv[1] + ("" if not bad else " " + str(bad[:3])))
PY
}
echo "[2] forward-pass reproduction (cri_run.py conf cT61 cP61)"
for m in cT61 cP61; do cp logs/conf/model_$m.json verify/model_$m.committed.json; done
python3 scripts/cri_run.py conf cT61 cP61 > verify/cri_run.out 2>&1
for m in cT61 cP61; do cmp logs/conf/model_$m.json verify/model_$m.committed.json 1e-6; done
echo "[3] replay of the primary statistics from committed arrays"
cp logs/conf/primary.json verify/primary.committed.json
python3 scripts/cri_stats.py conf primary cT61 cT62 cT63 cT64 cT65 cT67 cP61 cP62 cP63 cP64 cP66 cP67 > verify/stats.out 2>&1
python3 - <<'PY'
import json, math
a = json.load(open("logs/conf/primary.json")); b = json.load(open("verify/primary.committed.json"))
for k in ("P1_primary_N_input_clogit_pearson", "P2_adjusted", "P3_R_error_coherence", "family_Nabs"):
    x, y = a[k], b[k]; keys = [kk for kk in ("DELTA", "perm_p_one_sided", "DELTA_adj_KL", "perm_p_KL") if kk in y]
    ok = all(math.isclose(x[kk], y[kk], abs_tol=1e-9) for kk in keys); print(("    OK   " if ok else "    DIFF ") + k, {kk: y[kk] for kk in keys})
PY
echo "done: $C/verify"
