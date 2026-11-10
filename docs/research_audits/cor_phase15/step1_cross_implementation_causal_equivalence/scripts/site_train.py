"""Training-objective site test (mandate §8).  Usage: site_train.py SEED

Arm "pcsrc": the P-count objective of the continuation (train_tasks.train, task "pcount", 3000 steps, identical optimiser,
schedule, sampler, loss and output head WX) with ONE change: the count targets of question q (leave-q agreement counts of
the three groups, 3 x 6-way CE) are read from the final-layer state of one SOURCE token of question q instead of the
readout token 36+q.  Supervised token for question q: slot 2*(q % 3) (groups 0,1,2,0,1,2 for q = 0..5), so the number of
supervised positions (6), the loss normalisation and the targets are identical to the readout arm; only the position moves.
The readout-site control arm is the existing P-count pretraining (Investigation III pre_pcount_s0..s4, same trainer).
Then the identical frozen-feature T4 transfer as Investigation III run_job3.transfer (2000 steps, layers 0-1 + embeddings
frozen, layer 2 reset, T4 RNG seed = SEED, eval 20,000 episodes stream 6_300_000).
Outputs checkpoints/pre_pcsrc_s{SEED}.{npz,json} and checkpoints/ft_pcsrc_s{SEED}.{npz,json,log}."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src"))
sys.path.insert(0, os.path.join(ROOT, "step1_relational_mechanism_deepening", "src"))
import numpy as np, jax
import train_tasks as TT, train_ext as TE, ckpt as C, fwd as F
seed = int(sys.argv[1]); out = os.path.join(HERE, "..", "checkpoints"); os.makedirs(out, exist_ok=True)
SUP = np.array([6 * q + 2 * (q % 3) for q in range(6)])


def src_feats(p, X):
    rec = {}
    def hk(n, a):
        if n == "final": rec[n] = a
        return a
    o = F.forward(p, X, hk); return o, rec["final"][:, SUP]


TT.readout_feats = src_feats                     # loss() and evaluate() look this name up at call time
lf = open(os.path.join(out, f"ft_pcsrc_s{seed}.log"), "w"); log = lambda s: (lf.write(s + "\n"), lf.flush())
p, h, dt = TT.train("pcount", seed, 3000, log=log); ev = TT.evaluate(p, "pcount", seed=41_000_000)
C.save(os.path.join(out, f"pre_pcsrc_s{seed}.npz"), p)
ev.update(job=f"pre_pcsrc_s{seed}", seconds=dt, hist=h, sup_positions=SUP.tolist())
json.dump(ev, open(os.path.join(out, f"pre_pcsrc_s{seed}.json"), "w"), indent=1, default=float); log(json.dumps(ev, default=float))
p0 = dict(p); p0.pop("WX", None)
q, h2, dt2 = TE.train("T4", seed, 2000, p0=p0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True, log=log)
ev2 = TE.evaluate(q, "T4", 20000, 6_300_000); ev2.pop("X"); ev2.pop("Y")
C.save(os.path.join(out, f"ft_pcsrc_s{seed}.npz"), q)
ev2.update(job=f"ft_pcsrc_s{seed}", seconds=dt2, hist=h2)
json.dump(ev2, open(os.path.join(out, f"ft_pcsrc_s{seed}.json"), "w"), indent=1, default=float); log(json.dumps(ev2, default=float))
