"""Usage: run_job.py JOB.  Jobs:
  pre_{plocal,pgroup,pcount}  : pretrain (plocal 6000; pcount/pgroup 3000 after a compute-budget cut, recorded in COMPUTE_LOG)
  ft_{plocal,pgroup,pcount}   : T4 fine-tune 2000 steps (budget cut from 3000) from the pretrained model, layers 0-1 + embeddings FROZEN, last layer
                                re-initialised (the atlas post hoc (vii) protocol, under which frozen T1 reached 0.742)
  gate_s0, trans_s0           : beyond-reliability tasks, 4000 steps (budget cut)"""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_computation_atlas", "src"))
import numpy as np
import train_tasks as TT, train_ext as TE, ckpt as C, tasks_ext as T
job = sys.argv[1]; out = os.path.join(HERE, "..", "checkpoints"); os.makedirs(out, exist_ok=True)
lf = open(os.path.join(out, job + ".log"), "w"); log = lambda s: (lf.write(s + "\n"), lf.flush())
if job.startswith("pre_"):
    task = job[4:]; p, h, dt = TT.train(task, 0, 6000 if task == "plocal" else 3000, log=log); ev = TT.evaluate(p, task, seed=41_000_000)
elif job in ("ref_T1", "ref_randctrl"):
    # P-reliability reference (atlas T1 s0 features) and P-control (random frozen layers), same 2000-step frozen protocol
    p0 = C.load(os.path.join(HERE, "..", "..", "step1_latent_reliability", "checkpoints", "main", "T1_M_K3_s0.npz")) if job == "ref_T1" else None
    p, h, dt = TE.train("T4", 0, 2000, p0=p0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True, log=log)
    ev = TE.evaluate(p, "T4", 20000, 6_300_000); ev.pop("X"); ev.pop("Y")
elif job.startswith("ft_"):
    task = job[3:]; p0 = C.load(os.path.join(out, f"pre_{task}.npz")); p0.pop("WX", None)
    p, h, dt = TE.train("T4", 0, 2000, p0=p0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True, log=log)
    ev = TE.evaluate(p, "T4", 20000, 6_300_000); ev.pop("X"); ev.pop("Y")
elif job in ("gate_s0", "trans_s0"):
    task = job.split("_")[0]; p, h, dt = TT.train(task, 0, 4000, log=log); ev = TT.evaluate(p, task, seed=42_000_000)
C.save(os.path.join(out, job + ".npz"), p)
ev.update(job=job, seconds=dt, hist=h); json.dump(ev, open(os.path.join(out, job + ".json"), "w"), indent=1); log(json.dumps(ev))
