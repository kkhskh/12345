"""XIV preflight for score_X.py (no confirmation output read).  P0: refuses on the real state while training runs / before completion;
P1: on a synthetic root built from DEVELOPMENT seed 1 (snapshots 25..400 from g_train.py 1 400, the development final gate_s1 for every
later snapshot) the scorer runs and returns X1a CONFIRMED / X1b FALSIFIED (dev s1 shows an order-1 phase, and its agreement index stays
< 0.5 before 400; the final jumps to G = 1), exercising both branches;  P2: one snapshot removed -> refused.  Writes logs/preflight_X.json."""
import json, os, sys, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, gatelib as GL
def run(ck):
    o = os.path.join(tempfile.mkdtemp(), "s.json"); p = subprocess.run([sys.executable, os.path.join(HERE, "score_X.py"), ck, o], capture_output=True, text=True)
    return p.returncode, (json.load(open(o)) if os.path.exists(o) else None), p.stdout[-300:]
res = {}
rc, o, _ = run(os.path.join(PKG, "checkpoints", "gate")); res["P0_refuse_real_state"] = rc == 2 and o is None
SN = sorted(set(range(25, 1501, 25)) | set(range(1600, 2001, 100)) | set(range(2500, 4001, 500)))
root = tempfile.mkdtemp(); fin = jax.tree_util.tree_leaves(GL.load(GL.DEV_FINALS["s1"]))
for s in (3, 4):
    d = os.path.join(root, f"s{s}"); os.makedirs(d); open(os.path.join(d, "hist.json"), "w").write("{}")
    for k in SN:
        src = os.path.join(PKG, "checkpoints", "gate", "s1_stop400", f"step{k}.npz")
        if k <= 400: shutil.copy(src, os.path.join(d, f"step{k}.npz"))
        else: np.savez(os.path.join(d, f"step{k}.npz"), *[np.asarray(a) for a in fin])
rc, o, out = run(root); res["P1_synthetic_dev"] = rc == 0 and o["verdicts"] == {"X1a": "CONFIRMED", "X1b": "FALSIFIED"}; res["P1_detail"] = o["verdicts"] if o else out
os.remove(os.path.join(root, "s4", "step700.npz")); rc, o, _ = run(root); res["P2_missing_refuse"] = rc == 2 and o is None
json.dump(res, open(os.path.join(PKG, "logs", "preflight_X.json"), "w"), indent=1); print(res); sys.exit(0 if all(v for k, v in res.items() if k.startswith("P") and not k.endswith("detail")) else 1)
