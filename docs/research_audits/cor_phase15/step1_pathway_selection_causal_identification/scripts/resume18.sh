#!/bin/bash
# XVIII: resumption after the container restart of 2026-11-19 ~00:4xZ (stages 1-2 complete). Runs the remaining frozen stages of
# run_all18.sh (3 screening, 4 parents) and then the deferred stage-2b stability continuations. Lane claims make it idempotent;
# interrupted runs resume bitwise from resume.pkl.
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
lanes() { QUEUE=$1 bash scripts/lane2.sh ${2}1 & QUEUE=$1 bash scripts/lane2.sh ${2}2 & wait; }
stamp() { echo "$(date -u +%FT%TZ) STAGE $*" >> logs/COMPUTE_LOG.txt; }
stamp "3 screening resume"; lanes logs/queue_scr.txt S; python3 scripts/xviii_parents.py > logs/xviii_parents.out 2>&1
if [ -s logs/queue_scr2.txt ]; then lanes logs/queue_scr2.txt R; python3 scripts/xviii_parents.py > logs/xviii_parents.out 2>&1; fi; stamp "3 screening end"
[ -s logs/queue_p18.txt ] && { stamp "4 parents start"; lanes logs/queue_p18.txt P; python3 scripts/score_parents18.py > logs/score_parents18.out 2>&1; stamp "4 parents end rc=$?"; }
stamp "2b continuations start"; lanes logs/queue_cont18.txt K; python3 scripts/score_cont18.py > logs/score_cont18.out 2>&1; stamp "2b continuations end"
stamp "ALL DONE"
