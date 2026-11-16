"""XIV Stage I development: readout attention mass by key group, per layer, on the two bases (design of the off-route sham).
Writes logs/i_attn_dev.json.  Development only (stream 97_000_000, not used for confirmation)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax.numpy as jnp
import xc, ftlib as F, native as NV
X = NV.Bm.sample_fast(NV.ENV, 400, np.random.default_rng(97_000_000))["X"]
own = np.zeros((6, 43), bool); oth = np.zeros((6, 43), bool)
for q in range(6): own[q, q*6:(q+1)*6] = True; oth[q, 0:36] = True; oth[q, q*6:(q+1)*6] = False
res = {}
for name, spec in (("E", "uP160:fx:21:21:21:base"), ("M", "uP160:fx:23:23:23:base")):
    for k in (25, 100, 546, 1211, 4000):
        if k not in xc.steps_available(spec): continue
        M = xc.load(spec, k); rec = {}; F.forward(M, jnp.asarray(X), rec=rec)
        r = {}
        for l in range(3):
            A = np.asarray(rec[f"pattern{l}"])[:, :, 36:42, :]          # n,H,6,43
            g = dict(own=(A * own).sum(-1), other=(A * oth).sum(-1), readouts=A[..., 36:42].sum(-1), episode=A[..., 42])
            r[f"L{l}"] = {a: dict(mean=float(b.mean()), max_head=float(b.mean((0, 2)).max())) for a, b in g.items()}
        res[f"{name}@{k}"] = r; print(name, k, {l: {a: round(v["mean"], 3) for a, v in d.items()} for l, d in r.items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "i_attn_dev.json"), "w"), indent=1)
