"""XVII per-slot reconstruction: src/xviislot.SlotDiag on every snapshot in a run directory.  Usage: xvii_slot_traj.py COND RUNDIR OUT.json
Diagnostic stream 99_800_000 (distinct from XVI's inline 99_700_000), 1500 episodes, per-slot gate flips (one slot at a time)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import gatelib as GL, xviislot as XS
cond, rd, outp = sys.argv[1], sys.argv[2], sys.argv[3]; D = XS.SlotDiag(cond); res = dict(cond=cond, run=rd, balance=D.balance, traj=[])
for k in GL.snap_steps(rd):
    r = D.score(GL.load(os.path.join(rd, f"step{k}.npz"))); r["step"] = k; res["traj"].append(r)
    print(k, [round(x, 2) for x in r["F_G"]], flush=True)
json.dump(res, open(outp, "w"), indent=1)
