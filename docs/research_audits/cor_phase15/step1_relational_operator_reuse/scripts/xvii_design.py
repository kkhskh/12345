"""XVII frozen design: fresh confirmation cohorts, the analysis jobs and the learning-arm jobs -> logs/XVII_design.json, logs/queue_xvii.txt.
Arity 3: condition C (3:0.9:0.9:1/3), XVI constant-rate recipe, 5000 steps, inline diagnostics every 50, seeds 801-804.
Arity 2: condition 2:0.9:0.9:0.05 (XV's A = XVI's WB-high), XV cosine recipe, 4000 steps, inline diagnostics every 25, seeds 701-708.
Every run: scripts/xvii_train.py with BLIND=1 and parameter snapshots every 20 steps.  All 12 models are independent fresh initialisations
with independent data streams (seed-specific).  Queue order interleaves arities; analysis jobs wait for their run; learning jobs wait for all
analyses."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.join(HERE, "..", "logs")
runs = []
for s in (801, 802, 803, 804): runs.append(dict(name=f"A3_s{s}", arity=3, cond="3:0.9:0.9:0.3333333333333333", seed=s, sched="const", steps=5000, diag_every=50, dir=f"confirm_outputs/A3_s{s}"))
for s in range(701, 709): runs.append(dict(name=f"A2_s{s}", arity=2, cond="2:0.9:0.9:0.05", seed=s, sched="cosine", steps=4000, diag_every=25, dir=f"confirm_outputs/A2_s{s}"))
order = [runs[0], runs[4], runs[1], runs[5], runs[2], runs[6], runs[3], runs[7], runs[8], runs[9], runs[10], runs[11]]
lines = [f"BLIND=1 SNAP=20 python3 scripts/xvii_train.py {r['cond']} {r['seed']} {r['dir']} {r['sched']} {r['steps']} {r['diag_every']} -" for r in order]
lines += [f"bash scripts/when_file.sh {r['dir']}/done.json -- python3 scripts/xvii_confirm_analysis.py {r['cond']} {r['dir']} logs/analysis/{r['name']}.json" for r in order]
allan = " ".join(f"logs/analysis/{r['name']}.json" for r in runs)
lines += [f"bash scripts/when_file.sh {allan} -- python3 scripts/xvii_learning_arm.py {r['name']} {arm}" for r in order for arm in ("targeted", "full")]
json.dump(dict(runs=runs, analysis_dir="logs/analysis", learning_dir="learning"), open(os.path.join(LOGS, "XVII_design.json"), "w"), indent=1)
open(os.path.join(LOGS, "queue_xvii.txt"), "w").write("\n".join(lines) + "\n"); print(len(lines), "queue lines")
