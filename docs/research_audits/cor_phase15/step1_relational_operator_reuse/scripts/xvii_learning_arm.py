"""XVII frozen learning-intervention arm runner.  Usage: xvii_learning_arm.py NAME ARM   (ARM targeted | full)
Learning subset (frozen): every evaluable arity-3 model and the first two evaluable arity-2 models by seed (evaluable = analysis status
'evaluable', logs/analysis/<name>.json).  Models outside the subset: exits 0 without training.  Arm: scripts/xvii_slot_train.py from the
selected partial checkpoint, MODE all, 400 steps, diagnostics every 100 steps (600 episodes), data seed 1; targeted = loss on slot
j0 = min(U) only; full = loss on all six slots.  Output: learning/<NAME>_<ARM>/slot_traj.json."""
import json, os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, ".."); name, arm = sys.argv[1], sys.argv[2]
D = json.load(open(os.path.join(P, "logs", "XVII_design.json"))); R = {r["name"]: r for r in D["runs"]}
A = {n: json.load(open(os.path.join(P, "logs", "analysis", n + ".json"))) for n in R}
ev = lambda ar: sorted([n for n in R if A[n]["status"] == "evaluable" and R[n]["arity"] == ar], key=lambda n: R[n]["seed"])
if name not in ev(3) + ev(2)[:2]: print("not in learning subset:", name); sys.exit(0)
a = A[name]; ck = os.path.join(R[name]["dir"], f"step{a['selected']}.npz"); loss = str(min(a["U"])) if arm == "targeted" else "all"
sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "xvii_slot_train.py"), R[name]["cond"], ck, os.path.join("learning", f"{name}_{arm}"), "400", "all", "-", loss, "1", "100", "600"]))
