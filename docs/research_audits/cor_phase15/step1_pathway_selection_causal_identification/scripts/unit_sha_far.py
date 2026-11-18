"""XVIII unit check (pre-review A addendum, A1): the share-clamp solver on far targets.  Random log-odds tensors l (n, h, 6) with current
head-mean shares from 0.01 to 0.6 and targets from 0.01 to 0.6 (all 144 combinations); the solver must reach the target within 1e-4 with |beta| < 20.
Executes the harness's sha_beta (source up to the training loop) with a synthetic schedule.  Writes logs/unit_sha_far.json."""
import os, sys, json, tempfile
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xviii_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace('OUT = os.path.join(HERE, "..", "checkpoints", "ft", tag); os.makedirs(OUT, exist_ok=True)', "OUT = None")
import numpy as np
grid = [0.01, 0.03, 0.08, 0.15, 0.25, 0.4, 0.5, 0.6, 0.02, 0.12, 0.234, 0.3]
t = tempfile.mkdtemp(); sp = os.path.join(t, "slog.json"); json.dump(dict(shares=[[i, [grid[i % 12]] * 4, 1.0] for i in range(12)]), open(sp, "w"))
os.environ.update(FX_INTERV="sha:100", FX_EDGES="own2", FX_SHA_SCHED=sp, FX_SHA_LO="0", FX_SHA_CONTENT="real"); sys.argv = ["x", "uP160", "23", "23", "801"]
g = {"__file__": os.path.join(H, "xviii_train.py"), "__name__": "unit"}; exec(SRC, g); import jax, jax.numpy as jnp
rng = np.random.default_rng(5); worst = 0.0; fails = 0; maxb = 0.0
for ci, cur in enumerate(grid):
    for ti in range(12):
        l0 = rng.normal(0, 1.5, (64, 4, 6)); c = np.log(cur / (1 - cur)); l = l0 - l0.mean() + c          # crude centring near the current share
        s = np.full((64, 4, 43, 43), -3.0, np.float32)
        # encode the log-odds l into logits: put the target edges' logits at l (one T edge per row would suffice; use all 6 equal)
        XMn = np.asarray(g["XM"]); 
        for q in range(6): s[:, :, 36 + q, :] = np.where(XMn[36 + q] > 0, l[:, :, q:q + 1] - np.log(6) + np.log(37), 0.0)
        b = np.asarray(g["sha_beta"](jnp.asarray(s), jnp.int32(ti))); ach = np.asarray(jax.nn.sigmoid(b[None, :, None] + np.asarray(g["sha_ell"](jnp.asarray(s)))).mean((0, 2)))
        err = float(np.abs(ach - grid[ti]).max()); worst = max(worst, err); maxb = max(maxb, float(np.abs(b).max())); fails += (err > 1e-4) or (np.abs(b).max() >= 20)
r = dict(cases=144, fails=int(fails), worst_err=worst, max_abs_beta=maxb); print(r); json.dump(r, open(os.path.join(H, "..", "logs", "unit_sha_far.json"), "w"), indent=1)
