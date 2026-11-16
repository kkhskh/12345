"""Preflight for score_X_amend.py (no confirmation output read): P0 refuses on a root with an incomplete seed; P1 runs on the synthetic
development root (dev seed 1 snapshots 25..400 + dev final for later steps) and returns valid verdicts; P2 refuses with a missing snapshot.
Writes logs/preflight_X_amend.json."""
import json, os, sys, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, gatelib as GL
def run(ck):
    o = os.path.join(tempfile.mkdtemp(), "s.json"); p = subprocess.run([sys.executable, os.path.join(HERE, "score_X_amend.py"), ck, o], capture_output=True, text=True)
    return p.returncode, (json.load(open(o)) if os.path.exists(o) else None)
SN = sorted(set(range(25, 1501, 25)) | set(range(1600, 2001, 100)) | set(range(2500, 4001, 500)))
root = tempfile.mkdtemp(); fin = jax.tree_util.tree_leaves(GL.load(GL.DEV_FINALS["s1"]))
for s in (3, 4):
    d = os.path.join(root, f"s{s}"); os.makedirs(d)
    for k in SN:
        if k <= 400: shutil.copy(os.path.join(PKG, "checkpoints", "gate", "s1_stop400", f"step{k}.npz"), os.path.join(d, f"step{k}.npz"))
        else: np.savez(os.path.join(d, f"step{k}.npz"), *[np.asarray(a) for a in fin])
res = {}
rc, o = run(root); res["P0_refuse_incomplete"] = rc == 2 and o is None          # hist.json absent -> training 'not finished'
for s in (3, 4): open(os.path.join(root, f"s{s}", "hist.json"), "w").write("{}")
rc, o = run(root); res["P1_runs_on_synthetic_dev"] = rc == 0 and set(o["verdicts"].values()) <= {"CONFIRMED", "FALSIFIED", "INCONCLUSIVE"}; res["P1_detail"] = o["verdicts"] if o else None
res["P1_dev_s1_traj_A_p_max_le400"] = max(t["A_p"] for t in o["models"]["s3"]["trajectory"] if t["step"] <= 400) if o else None
os.remove(os.path.join(root, "s4", "step700.npz")); rc, o = run(root); res["P2_missing_refuse"] = rc == 2 and o is None
json.dump(res, open(os.path.join(PKG, "logs", "preflight_X_amend.json"), "w"), indent=1); print(res)
sys.exit(0 if res["P0_refuse_incomplete"] and res["P1_runs_on_synthetic_dev"] and res["P2_missing_refuse"] else 1)
