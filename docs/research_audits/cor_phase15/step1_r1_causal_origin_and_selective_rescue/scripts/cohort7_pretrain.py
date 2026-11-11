"""Investigation VII confirmation cohort, STAGE 0 (pretraining only).  Usage: cohort7_pretrain.py SEED
P-count pretraining with the ORIGINAL protocol (train_tasks.train("pcount", seed, 3000)), seeds 90-97 (new lineages: no seed shared
with Investigations IV-VI, no parameter inherited from any development parent).  Writes checkpoints/yP{seed}/pre.npz and pre.json
(the pretraining auxiliary evaluation only).  NO T4 transfer and NO T4 evaluation is run here: transfer, repair and evaluation of these
parents happen only after FROZEN_CONFIRMATION_PROTOCOL.md is committed (see STAGE0_COHORT_FREEZE.md)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, ckpt as C, train_tasks as TT
seed = int(sys.argv[1]); out = os.path.join(HERE, "..", "checkpoints", f"yP{seed}"); os.makedirs(out, exist_ok=True)
if not os.path.exists(os.path.join(out, "pre.json")):
    lf = open(os.path.join(out, "pretrain.log"), "a"); log = lambda s: (lf.write(s + "\n"), lf.flush())
    p, h, dt = TT.train("pcount", seed, 3000, log=log); ev = TT.evaluate(p, "pcount", seed=41_000_000)
    C.save(os.path.join(out, "pre.npz"), p)
    json.dump(dict(arm="P", seed=seed, seconds=dt, eval=ev, hist=h), open(os.path.join(out, "pre.json"), "w"), indent=1, default=float)
