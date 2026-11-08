"""PP3 curriculum-control fine-tunes and PP4 unknown-group training. Usage: run_training.py JOB
JOBS: pp3_T0, pp3_T5, pp3_T1_resetlast, pp3_T1_reset01, pp3_T2, pp3_scratch, pp4_T1perm, pp4_T4perm"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train_ext as TE
import ckpt as C, numpy as np
job = sys.argv[1]; LR = os.path.join(os.path.dirname(__file__), "..", "..", "step1_latent_reliability", "checkpoints", "main")
out = os.path.join(os.path.dirname(__file__), "..", "checkpoints"); os.makedirs(out, exist_ok=True)
lf = open(os.path.join(out, job + ".log"), "w"); log = lambda s: (lf.write(s + "\n"), lf.flush())
spec = {"pp3_T0": ("T0_M_K3_s0", ()), "pp3_T5": ("T5_M_K3_s0", ()), "pp3_T1_resetlast": ("T1_M_K3_s0", (2,)),
        "pp3_T1_reset01": ("T1_M_K3_s0", (0, 1)), "pp3_T2": ("T2_M_K3_s0", ()), "pp3_scratch": (None, ())}
if job in spec:
    src, re = spec[job]
    p0 = C.load(os.path.join(LR, src + ".npz")) if src else None
    p, h, dt = TE.train("T4", 0, 3000, p0=p0, reinit_layers=re, log=log)
    ev = TE.evaluate(p, "T4", 20000, 6_300_000); ev.pop("X"); ev.pop("Y")
elif job == "pp4_T1perm":
    p, h, dt = TE.train("T1", 20, 6000, perm=True, log=log)
    ev = TE.evaluate(p, "T1", 20000, 6_400_000, perm=True); ev.pop("X"); ev.pop("Y")
elif job == "pp4_T4perm":
    p0 = C.load(os.path.join(out, "pp4_T1perm.npz"))
    p, h, dt = TE.train("T4", 20, 3000, p0=p0, perm=True, log=log)
    ev = TE.evaluate(p, "T4", 20000, 6_400_001, perm=True); ev.pop("X"); ev.pop("Y")
C.save(os.path.join(out, job + ".npz"), p)
ev.update(job=job, seconds=dt, hist=h); json.dump(ev, open(os.path.join(out, job + ".json"), "w"), indent=1); log(json.dumps(ev))
