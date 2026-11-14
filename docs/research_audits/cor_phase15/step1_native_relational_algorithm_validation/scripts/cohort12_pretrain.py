"""Investigation XII fresh confirmation cohort, STAGE 0 (copied from Investigation X cohort10_pretrain.py; only docstring changed) (pretraining only).  Usage: cohort8_pretrain.py ARM SEED   ARM in {P, T1}
P-count: train_tasks.train("pcount", seed, 3000) (original protocol); T1: nets.train("T1", "M", seed, 6000, K=3, arch="tf") (original).
Seeds: P 150-154 (no overlap with Investigations IV-XI).  Writes checkpoints/z{ARM}{seed}/pre.npz, pre.json.  No T4 transfer
or evaluation happens here (STAGE0_COHORT_FREEZE.md)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"): sys.path.insert(0, os.path.join(ROOT, p_))
import ckpt as C, train_tasks as TT, nets as N
arm, seed = sys.argv[1], int(sys.argv[2]); out = os.path.join(HERE, "..", "checkpoints", f"z{arm}{seed}"); os.makedirs(out, exist_ok=True)
if not os.path.exists(os.path.join(out, "pre.json")):
    lf = open(os.path.join(out, "pretrain.log"), "a"); log = lambda s: (lf.write(s + "\n"), lf.flush())
    if arm == "P": p, h, dt = TT.train("pcount", seed, 3000, log=log); ev = TT.evaluate(p, "pcount", seed=41_000_000)
    else: p, h, dt = N.train("T1", "M", seed, 6000, K=3, log=log, arch="tf"); ev = N.evaluate(p, "T1", K=3, seed=999, arch="tf")
    C.save(os.path.join(out, "pre.npz"), p); json.dump(dict(arm=arm, seed=seed, seconds=dt, eval=ev, hist=h), open(os.path.join(out, "pre.json"), "w"), indent=1, default=float)
