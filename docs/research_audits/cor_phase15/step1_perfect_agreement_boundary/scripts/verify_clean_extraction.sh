#!/bin/bash
# Clean-extraction check for Investigation VI.  Exports the committed tree of docs/research_audits/cor_phase15 (git archive HEAD;
# earlier packages are read-only dependencies), then
#  [1] checks MANIFEST.sha256;
#  [2] FORWARD-PASS reproduction (exact replay, not independent confirmation): re-runs scripts/boundary_run.py conf on xT70@4000 and
#      xP80@4000 from committed checkpoints (stream 15_100_000) and compares regenerated model_*.json with the committed ones (tol 1e-6);
#  [3] REPLAY of the frozen confirmatory analysis from committed per-model logs (scripts/confirm_analysis.py) and comparison of
#      P1/P2/P4 statistics with committed logs/confirmatory.json (tol 1e-9);
#  [4] REPLAY of P6 analysis (scripts/p6_analysis.py) against committed logs/p6.json.
set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); REL=docs/research_audits/cor_phase15
OUT=${1:-/tmp/cor_inv6_clean}; rm -rf "$OUT"; mkdir -p "$OUT"; git -C "$REPO" archive HEAD $REL | tar -x -C "$OUT"
C="$OUT/$REL/step1_perfect_agreement_boundary"; cd "$C"; mkdir -p verify
echo "[1] manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "    manifest OK"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cmpj() { python3 - "$1" "$2" "$3" <<'PY'
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
echo "[2] forward-pass reproduction"
for m in xT70_4000 xP80_4000; do cp logs/conf/model_$m.json verify/model_$m.committed.json; done
python3 scripts/boundary_run.py conf xT70@4000 xP80@4000 > verify/boundary_run.out 2>&1
for m in xT70_4000 xP80_4000; do cmpj logs/conf/model_$m.json verify/model_$m.committed.json 1e-6; done
echo "[3] replay of frozen confirmatory analysis"; cp logs/confirmatory.json verify/confirmatory.committed.json
python3 scripts/confirm_analysis.py conf > verify/confirm.out 2>&1; cmpj logs/confirmatory.json verify/confirmatory.committed.json 1e-9
echo "[4] replay of P6 analysis"; cp logs/p6.json verify/p6.committed.json; python3 scripts/p6_analysis.py > verify/p6.out 2>&1; cmpj logs/p6.json verify/p6.committed.json 1e-9
echo done
