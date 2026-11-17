"""XV trajectory diagnostics for one run directory (all snapshots).  Usage: xv_traj.py COND RUNDIR STREAM N OUT.json
Uses src/xvdiag.Diag (exact references) and the XIV GATE loader/forward (step1_cross_family_relational_foundations/src/gatelib.py)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, gatelib as GL, xvdiag as XD
cond, rd, stream, n, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
D = XD.Diag(cond, stream, n, os.path.join(HERE, "..", "logs")); res = dict(cond=cond, run=rd, stream=stream, n=n, refs=D.ref_scores(), traj=[])
for k in GL.snap_steps(rd):
    p = GL.load(os.path.join(rd, f"step{k}.npz")); r = D.score(GL.probs(p, D.X), GL.probs(p, D.Xf), GL.probs(p, D.Xw)); r["step"] = k; res["traj"].append(r)
json.dump(res, open(outp, "w"), indent=1)
