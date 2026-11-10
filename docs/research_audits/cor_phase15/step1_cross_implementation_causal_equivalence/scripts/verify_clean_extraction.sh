#!/bin/bash
# Clean-extraction check: export the committed tree (git archive HEAD) of cor_phase15 to a fresh directory, check
# MANIFEST.sha256, and re-run deterministic scripts whose JSON outputs must match the committed logs:
#   info_theory.py (exact generator numerics, tol 1e-9) and analyze.py conf (bootstrap with fixed seed, from the committed
#   responses.npz, tol 1e-9).  Usage: bash scripts/verify_clean_extraction.sh [OUTDIR]
set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); REL=docs/research_audits/cor_phase15
OUT=${1:-/tmp/cor_inv4_clean}; rm -rf "$OUT"; mkdir -p "$OUT"
git -C "$REPO" archive HEAD $REL | tar -x -C "$OUT"
C="$OUT/$REL/step1_cross_implementation_causal_equivalence"; cd "$C"
echo "[1] manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "    manifest OK"
mkdir -p verify
cmp_json() { python3 - "$1" "$2" "$3" <<'PY'
import json, sys, math
a, b, tol = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), float(sys.argv[3]); bad = []
def c(x, y, p):
    if isinstance(x, dict):
        for k in y: c(x.get(k), y[k], p + "/" + str(k))
    elif isinstance(x, list): [c(u, v, f"{p}[{i}]") for i, (u, v) in enumerate(zip(x, y))]
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if not (math.isclose(x, y, rel_tol=tol, abs_tol=tol) or (math.isnan(x) and math.isnan(y))): bad.append(f"{p}: {x} vs {y}")
    elif x != y: bad.append(f"{p}: {x} vs {y}")
c(a, b, ""); print(("    OK   " if not bad else "    DIFF ") + sys.argv[1] + ("" if not bad else f" {bad[:3]}"))
PY
}
echo "[2] exact generator numerics"
cp logs/info_theory.json verify/info_theory.committed.json; python3 scripts/info_theory.py > verify/info_theory.out
cmp_json logs/info_theory.json verify/info_theory.committed.json 1e-9
echo "[3] confirmation group statistics from committed responses"
cp logs/conf/analysis.json verify/analysis.committed.json; python3 scripts/analyze.py conf > verify/analyze.out
cmp_json logs/conf/analysis.json verify/analysis.committed.json 1e-9
echo "done: $C/verify"
