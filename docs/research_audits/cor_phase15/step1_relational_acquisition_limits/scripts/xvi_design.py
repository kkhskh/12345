"""XVI frozen design: conditions, cohorts, seeds, schedules, budgets, and the confirmation queue.
Writes logs/XVI_conditions.json, logs/XVI_design.json, logs/queue_train.txt.  Priority order of the queue follows the mandate:
Workstream A fresh constant-rate runs, then Workstream A continuations, then Workstream B."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.join(HERE, "..", "logs")
C3 = "3:0.9:0.9:0.3333333333333333"; D3 = "3:0.9:0.9:0.05"; LOW = "2:0.9:0.55:0.05"; HIGH = "2:0.9:0.9:0.05"
XV = "../step1_relational_learning_order_laws/confirm_outputs"
cond = dict(WA_fresh=C3, WA_cont_C=C3, WA_cont_D=D3, value_low=LOW, value_high=HIGH, all_neural_conditions=[C3, D3, LOW, HIGH],
            arity4=dict(spec="4:0.9:0.9:0.3333333333333333:7", matched_arity3_spec="3:0.9:0.9:0.3333333333333333:7"))
runs = []
for s in (401, 402, 403, 404):
    runs.append(dict(cohort="WA_fresh_const", cond=C3, seed=s, sched="const", steps=15000, diag_every=50, init=None, dir=f"confirm_outputs/WA_const_s{s}"))
for c, sp, seeds in (("C", C3, (201, 202, 203)), ("D", D3, (201, 202, 204))):
    for s in seeds:
        runs.append(dict(cohort="WA_continuation_restart", cond=sp, seed=s, sched="cont", steps=4000, diag_every=50, init=f"{XV}/{c}_s{s}/step4000.npz",
                         xv_run=f"{c}_s{s}", dir=f"confirm_outputs/WA_cont_{c}_s{s}"))
for lvl, sp in (("low", LOW), ("high", HIGH)):
    for s in (501, 502, 503, 504):
        runs.append(dict(cohort=f"WB_value_{lvl}", cond=sp, seed=s, sched="cosine", steps=4000, diag_every=25, init=None, dir=f"confirm_outputs/WB_{lvl}_s{s}"))
design = dict(runs=runs, diag_stream=99_700_000, diag_n=1500, snapshot_every=500,
              independence=dict(WA_fresh="4 independent fresh initialisations (seeds 401-404) and independent data streams",
                                WA_cont="6 continuations of 6 distinct XV models (the 6 censored XV arity-3 runs); NOT fresh replications; selected as the XV failures",
                                WB="4 seed pairs (501-504): within a pair the two value conditions share the initialisation and every random draw (verified V1), "
                                   "so the inferential unit for the value contrast is the pair (4 independent pairs); across pairs everything is independent"),
              xv_reference_runs=dict(arity3_cosine=[f"{c}_s{s}" for c in "CD" for s in (201, 202, 203, 204)]))
json.dump(cond, open(os.path.join(LOGS, "XVI_conditions.json"), "w"), indent=1); json.dump(design, open(os.path.join(LOGS, "XVI_design.json"), "w"), indent=1)
with open(os.path.join(LOGS, "queue_train.txt"), "w") as f:
    for r in runs:
        f.write(f"BLIND=1 python3 scripts/xvi_train.py {r['cond']} {r['seed']} {r['dir']} {r['sched']} {r['steps']} {r['diag_every']}" + (f" {r['init']}" if r["init"] else "") + "\n")
print(len(runs), "runs")
