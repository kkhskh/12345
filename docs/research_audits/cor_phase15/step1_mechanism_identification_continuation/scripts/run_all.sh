#!/bin/sh
# Reproduce every result in this package. From the package root:
#   sh scripts/run_all.sh          # analytic + engineered experiments + tests (~2 min), uses existing checkpoints if present
#   sh scripts/run_all.sh --train  # also retrain all learned models (~35 min on 2 CPU cores)
set -e
cd "$(dirname "$0")/.."
python3 src/cprime_regions.py
python3 src/mlp_norm.py
python3 src/matched.py
python3 src/program_a.py
python3 src/program_b.py
python3 src/program_c.py
python3 src/relation_free_equivalent.py
python3 src/additive_lp.py
python3 src/mediator_sign_test.py
if [ "$1" = "--train" ]; then
  python3 src/learned.py dev 0,1,2
  python3 src/learned_analysis.py dev 0,1,2
  python3 src/learned.py heldout 3,4
  python3 src/learned_analysis.py heldout 3,4
  python3 src/learned_posthoc.py dev 0,1,2
  python3 src/learned_posthoc.py heldout 3,4
  python3 src/reparam.py
  python3 src/transfer.py
  python3 src/learned_review_checks.py
fi
python3 -m pytest -q tests/
