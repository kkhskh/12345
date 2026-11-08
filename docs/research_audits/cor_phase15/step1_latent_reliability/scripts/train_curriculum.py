"""POST HOC curriculum: start from a trained T1 M checkpoint, train on T4. Usage: train_curriculum.py SEED STEPS OUTDIR"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, jax
import nets as N, ckpt as C, bayes as B
seed, steps, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
p0 = C.load(os.path.join(os.path.dirname(__file__), "..", f"checkpoints/main/T1_M_K3_s{seed}.npz"))
# T4 has 6 R values vs 3 in T1: the (unused, unsupervised) R head WR must be re-initialised
fresh = N.init("M", B.Env("T4"), seed)
p0["WR"] = np.asarray(fresh["WR"])
tag = f"T4_M_K3_s{seed}_fromT1"
lf = open(os.path.join(out, tag + ".log"), "w")
p, hist, dt = N.train("T4", "M", seed, steps, log=lambda s: (lf.write(s + "\n"), lf.flush()), p0=p0)
C.save(os.path.join(out, tag + ".npz"), p)
ev = N.evaluate(p, "T4", seed=999)
json.dump(dict(task="T4", size="M", seed=seed, steps=steps, init=f"T1_M_K3_s{seed}", train_seconds=dt, hist=hist, dev_eval=ev,
               note="POST HOC curriculum (not spontaneous)"), open(os.path.join(out, tag + ".json"), "w"), indent=1)
lf.write(json.dumps(ev) + "\n")
