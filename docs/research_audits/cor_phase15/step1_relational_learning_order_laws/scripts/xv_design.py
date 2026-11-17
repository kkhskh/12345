"""XV frozen confirmation design -> logs/XV_design.json, logs/queue_train.txt, logs/queue_measure.txt.  Run once before FREEZE_XV."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
COND = dict(A="2:0.9:0.05", B="2:0.9:0.3333333333333333", C="3:0.9:0.3333333333333333", D="3:0.9:0.05")
SEEDS = (201, 202, 203, 204); STREAM, N = 98_700_000, 1500
runs = [dict(cond=c, spec=COND[c], seed=s, dir=f"confirm_outputs/{c}_s{s}", traj=f"confirm_outputs/traj_{c}_s{s}.json") for s in SEEDS for c in "ABCD"]
json.dump(dict(conditions=COND, seeds=SEEDS, stream=STREAM, n_episodes=N, runs=runs, steps=4000,
               snapshots="every 25 to 2000, every 250 to 4000"), open(os.path.join(L, "XV_design.json"), "w"), indent=1)
with open(os.path.join(L, "queue_train.txt"), "w") as f:
    for r in runs: f.write(f"BLIND=1 python3 scripts/x_train.py {r['spec']} {r['seed']} {r['dir']}\n")
with open(os.path.join(L, "queue_measure.txt"), "w") as f:
    for r in runs: f.write(f"bash scripts/when_file.sh {r['dir']}/done.json -- python3 scripts/xv_traj.py {r['spec']} {r['dir']} {STREAM} {N} {r['traj']}\n")
print(len(runs), "runs")
