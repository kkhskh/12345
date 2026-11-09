#!/bin/bash
# Clean-extraction check.
# Exports the committed tree (git archive HEAD) to a fresh directory and re-runs a set of cheap, deterministic scripts.
# It then compares their JSON outputs with the committed logs.
#   - Exact-Bayes scripts must match to 1e-9.
#   - Learned-model scripts use committed checkpoints and must match to 1e-6.
# The script also checks MANIFEST.sha256.
# Usage (from anywhere): bash scripts/verify_clean_extraction.sh [OUTDIR]
set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
REL=docs/research_audits/cor_phase15
OUT=${1:-/tmp/cor_clean_extract}
rm -rf "$OUT"; mkdir -p "$OUT"
git -C "$REPO" archive HEAD $REL | tar -x -C "$OUT"
C="$OUT/$REL/step1_foundational_atlas_continuation"
echo "[1] manifest"; (cd "$C" && sha256sum -c --quiet MANIFEST.sha256) && echo "    manifest OK"
mkdir -p "$C/verify"; cd "$C"
run() {
  # Run one script in the extracted tree, then compare its JSON output with the committed log.
  # Arguments: script, path of its output (relative to $C), committed log, tolerance.
  local script=$1 produced=$2 committed=$3 tol=$4
  cp "$committed" "verify/$(basename "$committed").committed"
  python3 "scripts/$script" > "verify/$(basename "$script").out" 2>&1
  python3 - "$produced" "verify/$(basename "$committed").committed" "$tol" <<'EOF'
import json, sys, math
a, b, tol = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), float(sys.argv[3])
bad = []
def cmp(x, y, path):
    if isinstance(x, dict):
        for k in y:
            if k in x: cmp(x[k], y[k], path + "/" + str(k))
            else: bad.append(path + "/" + str(k) + " missing")
    elif isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y)): cmp(u, v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if not (math.isclose(x, y, rel_tol=tol, abs_tol=tol) or (math.isnan(x) and math.isnan(y))): bad.append(f"{path}: {x} vs {y}")
    elif x != y: bad.append(f"{path}: {x} vs {y}")
cmp(a, b, "")
print(("    OK   " if not bad else "    DIFF ") + sys.argv[1] + ("" if not bad else f"  ({len(bad)} diffs, first: {bad[:3]})"))
EOF
}
echo "[2] exact-Bayes reproductions"
run landscape_checks.py        logs/landscape/checks.json         logs/landscape/checks.json        1e-9
run review_checks_landscape.py logs/landscape/review_checks.json  logs/landscape/review_checks.json 1e-9
run od_bridge.py               logs/od_bridge.json                logs/od_bridge.json               1e-9
run unknown_groups.py          logs/landscape/unknown_groups.json logs/landscape/unknown_groups.json 1e-9
run pp2_probe_control.py       logs/pp2_probe_control.json        logs/pp2_probe_control.json       1e-9

echo "[3] constructed-model and learned-model reproductions (committed checkpoints)"
run pp1_cx_matched.py          logs/pp1_cx_matched.json           logs/pp1_cx_matched.json          1e-9
run pp1_learned_ece.py         logs/pp1_learned_ece.json          logs/pp1_learned_ece.json         1e-6
echo "done: outputs in $C/verify"
