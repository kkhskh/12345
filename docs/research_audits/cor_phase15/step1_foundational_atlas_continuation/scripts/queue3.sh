#!/bin/bash
# After queue2 (pid $1): PP3 probes (needs pre_pcount, pre_pgroup) and the PP3 learning-dynamics run.
cd "$(dirname "$0")/.."
while kill -0 $1 2>/dev/null; do sleep 30; done
python3 scripts/pp3_probes.py > logs/pp3_probes.out 2>&1; echo "$(date +%T) pp3_probes done" >> logs/progress.log
python3 scripts/pp3_dynamics.py > logs/pp3_dynamics.out 2>&1; echo "$(date +%T) pp3_dynamics done" >> logs/progress.log
