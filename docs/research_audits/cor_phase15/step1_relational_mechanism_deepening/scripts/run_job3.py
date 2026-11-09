"""Usage: run_job3.py JOB.  Outputs checkpoints/JOB.{npz,json,log} (snapshots in checkpoints/JOB_snaps/).
  pre_{plocal|pcount|pgroup}_s{K}  : continuation trainer train_tasks.train (identical protocol to the seed-0 arms):
                                     plocal 6000 steps, pcount/pgroup 3000 steps
  ft_{plocal|pcount|pgroup}_s{K}   : frozen-feature T4 transfer, 2000 steps (layers 0-1 + embeddings frozen, layer 2 reset),
                                     T4 RNG seed K, eval 20,000 episodes stream 6_300_000 (identical to the continuation)
  ftlong_{arm}_s{K}                : same with 6000 fine-tune steps (asymptotic usability); arm in plocal|pcount|pgroup|T1|rand
  ref_T1p{P}                       : frozen-feature transfer from an existing T1 parent checkpoint (P = 0,1,2 main; 10-13 replication; 14,15 heldout)
  ref_rand_s{K}                    : random frozen network
  trans_full_s{K} (12000 steps) / trans_holdout_s{K}, trans_aux_s{K} (8000 steps) : TRANS, snapshots every 2000
  gate_s{K}                        : continuation trainer, 4000 steps (identical to gate_s0)
  adapt_s{K}                       : 6000 steps"""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax
import train3 as T3, train_tasks as TT, train_ext as TE, ckpt as C, nets as N, bayes as B
job = sys.argv[1]; out = os.path.join(HERE, "..", "checkpoints"); os.makedirs(out, exist_ok=True)
CONT = os.path.join(HERE, "..", "..", "step1_foundational_atlas_continuation", "checkpoints")
LRC = os.path.join(HERE, "..", "..", "step1_latent_reliability", "checkpoints")
lf = open(os.path.join(out, job + ".log"), "w"); log = lambda s: (lf.write(s + "\n"), lf.flush())
parts = job.split("_"); seed = int(parts[-1][1:]) if parts[-1].startswith("s") and parts[-1][1:].isdigit() else 0
def pre_path(arm, k):
    if arm == "T1": return os.path.join(LRC, "main", f"T1_M_K3_s{k}.npz")
    if k == 0: return os.path.join(CONT, f"pre_{arm}.npz")
    return os.path.join(out, f"pre_{arm}_s{k}.npz")
def transfer(p0, k, steps):
    p, h, dt = TE.train("T4", k, steps, p0=p0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True, log=log)
    ev = TE.evaluate(p, "T4", 20000, 6_300_000); ev.pop("X"); ev.pop("Y"); return p, h, dt, ev
if parts[0] == "pre":
    arm = parts[1]; p, h, dt = TT.train(arm, seed, 6000 if arm == "plocal" else 3000, log=log); ev = TT.evaluate(p, arm, seed=41_000_000)
elif parts[0] in ("ft", "ftlong"):
    arm = parts[1]; steps = 2000 if parts[0] == "ft" else 6000
    p0 = None if arm == "rand" else C.load(pre_path(arm, seed));
    if p0 is not None: p0.pop("WX", None)
    p, h, dt, ev = transfer(p0, seed, steps)
elif parts[0] == "ref" and parts[1].startswith("T1p"):
    P = int(parts[1][3:]); sub = "main" if P <= 2 else "replication"
    path = os.path.join(LRC, sub, f"T1_M_K3_s{P}.npz") if P <= 13 else os.path.join(CONT, "heldout", f"T1_M_K3_s{P}.npz")
    p, h, dt, ev = transfer(C.load(path), P, 2000); ev["parent"] = path
elif parts[0] == "ref" and parts[1] == "rand":
    p, h, dt, ev = transfer(None, seed, 2000)
elif parts[0] == "trans":
    task = "trans" if parts[1] == "full" else ("trans_holdout" if parts[1] == "holdout" else "trans_aux")
    sd = os.path.join(out, job + "_snaps"); os.makedirs(sd, exist_ok=True)
    steps = 12000 if task == "trans" else 8000
    p, h, dt = T3.train(task, seed, steps, log=log, snaps=tuple(range(2000, steps + 1, 2000)), snapdir=sd); ev = {}
elif parts[0] == "gate":
    p, h, dt = TT.train("gate", seed, 4000, log=log); ev = TT.evaluate(p, "gate", seed=42_000_000)
elif parts[0] == "adapt":
    sd = os.path.join(out, job + "_snaps"); os.makedirs(sd, exist_ok=True)
    p, h, dt = T3.train("adapt", seed, 6000, log=log, snaps=(1500, 3000, 4500), snapdir=sd); ev = {}
else:
    raise SystemExit("unknown job " + job)
C.save(os.path.join(out, job + ".npz"), p)
ev.update(job=job, seconds=dt, hist=h); json.dump(ev, open(os.path.join(out, job + ".json"), "w"), indent=1, default=float); log(json.dumps(ev, default=float))
