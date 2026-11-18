#!/bin/bash
# XVIII post-freeze orchestration (frozen).  Stages run in order, each on two lanes; every stage is resumable (lane claims).
#  1. within-parent confirmation (logs/queue_c18.txt: 48 runs, each followed by its frozen measurement)
#  2. immediate H3 metric, primary scoring (score_XVIII.py), stability continuations (continue18.py -> queue_cont18.txt) and their measurement
#  3. parent screening (queue_scr.txt), the frozen parent rule (xviii_parents.py; extra seeds 24/25 via queue_scr2.txt if required)
#  4. parent arms (queue_p18.txt) and the parent scorer (score_parents18.py)
cd "$(dirname "$0")/.."; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
lanes() { QUEUE=$1 bash scripts/lane2.sh ${2}1 & QUEUE=$1 bash scripts/lane2.sh ${2}2 & wait; }
stamp() { echo "$(date -u +%FT%TZ) STAGE $*" >> logs/COMPUTE_LOG.txt; }
stamp "1 confirmation start"; lanes logs/queue_c18.txt C; stamp "1 confirmation end"
python3 scripts/h3_immediate18.py logs/XVIII_design.json logs/h3_immediate_c18.json > logs/h3_immediate_c18.out 2>&1
python3 scripts/score_XVIII.py > logs/score_XVIII.out 2>&1; stamp "2 primary score rc=$?"
python3 scripts/continue18.py logs/score_XVIII.json logs/XVIII_design.json logs/measure_c18_cont.jsonl > logs/continue18.out 2>&1
[ -s logs/queue_cont18.txt ] && { stamp "2 continuations start"; lanes logs/queue_cont18.txt K; python3 scripts/score_cont18.py > logs/score_cont18.out 2>&1; stamp "2 continuations end"; }
stamp "3 screening start"; lanes logs/queue_scr.txt S; python3 scripts/xviii_parents.py > logs/xviii_parents.out 2>&1
if [ -s logs/queue_scr2.txt ]; then lanes logs/queue_scr2.txt R; python3 scripts/xviii_parents.py > logs/xviii_parents.out 2>&1; fi; stamp "3 screening end"
[ -s logs/queue_p18.txt ] && { stamp "4 parents start"; lanes logs/queue_p18.txt P; python3 scripts/score_parents18.py > logs/score_parents18.out 2>&1; stamp "4 parents end rc=$?"; }
stamp "ALL DONE"
