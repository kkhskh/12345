"""Check that the stop-gradient trainer used in Investigation VII (frozen layers 0-1 and embeddings under stop_gradient; only layer 2,
final LN, WA (and unused WC/WR) receive gradients) reproduces Investigation VI's transfer step (train_ext-style zero-masked gradients and
updates) for the same parent, initialisation and data stream.  20 steps from xP80/pre.npz.  Writes logs/sg_equivalence.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax, nnlib as NL
N, Bm = NL.N, NL.Bm
seed = 80; p0 = NL.load(NL.dev_ckpt("xP80", 0)); fresh = N.init("M", NL.ENV, seed + 7777)
p = dict(p0); p["layers"] = list(p0["layers"]); p["WR"] = fresh["WR"]; p["layers"][2] = fresh["layers"][2]
STEPS = 20; opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, 4000 // 10), 4000), weight_decay=1e-4)
def run(mode):
    q = jax.tree_util.tree_map(lambda a: a, p); st = opt.init(q)
    def fwd_sg(pp, X):
        r = dict(pp); r["layers"] = [jax.tree_util.tree_map(jax.lax.stop_gradient, l) if i < 2 else l for i, l in enumerate(pp["layers"])]
        for k in NL.FROZEN_KEYS: r[k] = jax.lax.stop_gradient(pp[k])
        return N.forward(r, X)
    @jax.jit
    def step(pp, st, X, T):
        l, g = jax.value_and_grad(N.loss_fn)(pp, X, T, "T4", fwd_sg if mode == "sg" else None)
        if mode == "orig":
            for fl in (0, 1): g["layers"][fl] = jax.tree_util.tree_map(jnp.zeros_like, g["layers"][fl])
            for k_ in NL.FROZEN_KEYS: g[k_] = jnp.zeros_like(g[k_])
        up, st = opt.update(g, st, pp)
        for fl in (0, 1): up["layers"][fl] = jax.tree_util.tree_map(jnp.zeros_like, up["layers"][fl])
        for k_ in NL.FROZEN_KEYS: up[k_] = jnp.zeros_like(up[k_])
        return optax.apply_updates(pp, up), st, l
    rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * 4 + 31 * 3)
    for i in range(STEPS):
        d = Bm.sample_fast(NL.ENV, 256, rng); q, st, l = step(q, st, jnp.asarray(d["X"]), N.batch_targets(NL.ENV, d))
    return q
a, b = run("orig"), run("sg")
diff = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
scale = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(a))
json.dump(dict(steps=STEPS, max_abs_param_diff=diff, max_abs_param=scale), open(os.path.join(HERE, "..", "logs", "sg_equivalence.json"), "w"), indent=1); print(diff, scale)
