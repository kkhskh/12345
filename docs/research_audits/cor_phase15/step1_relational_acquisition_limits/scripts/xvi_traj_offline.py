"""XVI diagnostics (src/xvidiag.py) recomputed offline on every parameter snapshot of an existing run directory (XV runs, development
runs, replayed checkpoints).  Usage: xvi_traj_offline.py COND RUNDIR OUT.json   (COND a:b1:b0:nu; XV's a:b:nu is a:b:b:nu).
Same diagnostic stream (99_700_000) and size (1500 episodes) as the inline confirmation diagnostics."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import gatelib as GL, xvidiag as XD
cond, rd, outp = sys.argv[1], sys.argv[2], sys.argv[3]
D = XD.Diag(cond, 99_700_000, 1500, os.path.join(HERE, "..", "logs")); res = dict(cond=cond, run=rd, refs=D.ref_scores(), traj=[])
for k in GL.snap_steps(rd):
    p = GL.load(os.path.join(rd, f"step{k}.npz")); r = D.score(GL.probs(p, D.X), GL.probs(p, D.Xf), GL.probs(p, D.Xw)); r["step"] = k; res["traj"].append(r)
json.dump(res, open(outp, "w"), indent=1)
