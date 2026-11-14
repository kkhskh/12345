"""Workstream D1: why do P-count parents fail the readiness criterion?  Per-group / per-target-count accuracy of the pretraining
auxiliary readout (pcount: for every question q and group g, predict n_g(q) = number of OTHER questions on which g's two sources agree).
Usage: diag_pretrain.py PATH_TO_pre.npz ... -> prints and writes logs/diag_pretrain.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"): sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax, jax.numpy as jnp
import ckpt as C, train_tasks as TT
out = {}
d = TT.SAMPLERS["pcount"](6000, np.random.default_rng(41_000_000))
for f in sys.argv[1:]:
    p = C.load(f); pj = jax.tree_util.tree_map(jnp.asarray, p); fn = jax.jit(lambda X: TT.readout_feats(pj, X)[1])
    Z = np.concatenate([np.asarray(fn(jnp.asarray(d["X"][i:i + 1000]))) for i in range(0, 6000, 1000)])
    pred = (Z @ p["WX"]).reshape(6000, 6, 3, 6).argmax(-1); ok = pred == d["T"]
    r = dict(aux_acc=float(ok.mean()), per_group=ok.mean((0, 1)).round(3).tolist(), per_question=ok.mean((0, 2)).round(3).tolist(),
             per_target_count={int(c): float(ok[d["T"] == c].mean()) for c in range(6)})
    out[f.split("/")[-2]] = r; print(f.split("/")[-2], r["aux_acc"].__round__(3), "groups", r["per_group"], "counts", {k: round(v, 2) for k, v in r["per_target_count"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "diag_pretrain.json"), "w"), indent=1)
