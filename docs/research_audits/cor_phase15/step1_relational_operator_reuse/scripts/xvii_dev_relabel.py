"""XVII DEVELOPMENT: identity-relabel windows at one partial checkpoint. Usage: xvii_dev_relabel.py COND CKPT ACQ UNACQ OUT.json
For every (i in ACQ, j in UNACQ) and window (lo, hi): per-slot F_G at the recipient j (which receives i's slot code inside the window) and at the
donor i (which receives j's code).  Baseline: no intervention."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, gatelib as GL, xviislot as XS
cond, ck, A, U, outp = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], [int(x) for x in sys.argv[4].split(",")], sys.argv[5]
D = XS.SlotDiag(cond); p = GL.load(ck); base = D.score(p); res = dict(cond=cond, ckpt=ck, A=A, U=U, base=base, windows={})
W = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
for lo, hi in W:
    rec, don = [], []
    for i, j in itertools.product(A, U):
        s = D.score(p, (i, j, lo, hi)); rec.append(s["F_G"][j]); don.append(s["F_G"][i])
    res["windows"][f"{lo}-{hi}"] = dict(recipient_FG=rec, donor_FG=don)
    print(f"window layers [{lo},{hi}): recipient F_G mean {np.mean(rec):.2f} {[round(x,2) for x in rec]} | donor F_G mean {np.mean(don):.2f}", flush=True)
json.dump(res, open(outp, "w"), indent=1)
