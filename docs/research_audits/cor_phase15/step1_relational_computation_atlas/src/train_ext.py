"""Training variants for the atlas (same optimiser/schedule as the latent-reliability package's nets.train):
- init from any checkpoint, with optional re-initialisation of selected layers (PP3)
- optional per-episode source permutation (unknown groups, PP4)"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_latent_reliability", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import bayes as B, nets as N


def permute(X, rng):
    perms = np.argsort(rng.random((len(X), 6)), -1)
    return np.take_along_axis(X, perms[:, None, :], -1)


def train(task, seed, steps, p0=None, reinit_layers=(), perm=False, lr=1e-3, batch=256, log=None, size="M", freeze_layers=(), freeze_embed=False):
    env = B.Env(task, K=3, Q=6)
    fresh = N.init(size, env, seed + 7777)
    if p0 is None:
        p = N.init(size, env, seed)
    else:
        p = jax.tree_util.tree_map(jnp.asarray, p0)
        p["WR"] = fresh["WR"]                                   # R head size depends on the task
        if "WC" not in p: p["WC"] = fresh["WC"]
        for l in reinit_layers:
            p["layers"][l] = fresh["layers"][l]
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps), weight_decay=1e-4)
    st = opt.init(p)

    @jax.jit
    def step(p, st, X, T):
        l, g = jax.value_and_grad(N.loss_fn)(p, X, T, task)
        for fl in freeze_layers:                              # post hoc PP3 follow-ups: frozen transferred layers
            g["layers"][fl] = jax.tree_util.tree_map(jnp.zeros_like, g["layers"][fl])
        if freeze_embed:
            for k_ in ["Eval", "Esrc", "Eq", "Eread", "Eep"]: g[k_] = jnp.zeros_like(g[k_])
        up, st = opt.update(g, st, p)
        if freeze_layers or freeze_embed:                      # AdamW weight decay must not move frozen weights either
            for fl in freeze_layers:
                up["layers"][fl] = jax.tree_util.tree_map(jnp.zeros_like, up["layers"][fl])
            if freeze_embed:
                for k_ in ["Eval", "Esrc", "Eq", "Eread", "Eep"]: up[k_] = jnp.zeros_like(up[k_])
        return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * ["T0", "T1", "T2", "T3", "T4", "T5"].index(task) + 31 * 3 + (17 if perm else 0))
    t0 = time.time(); hist = []
    for i in range(steps):
        d = B.sample_fast(env, batch, rng)
        if perm:
            d["X"] = permute(d["X"], rng)
        p, st, l = step(p, st, jnp.asarray(d["X"]), N.batch_targets(env, d))
        if (i + 1) % max(steps // 10, 1) == 0:
            hist.append((i + 1, float(l)))
            if log: log(f"{task} s{seed} step {i + 1} loss {float(l):.4f}")
    return jax.tree_util.tree_map(np.asarray, p), hist, time.time() - t0


def evaluate(p, task, n, seed, perm=False):
    env = B.Env(task, K=3, Q=6); rng = np.random.default_rng(seed)
    d = B.sample_fast(env, n, rng)
    post = B.episode_posterior(env, d["X"])                      # known-group Bayes on the unpermuted data
    X = permute(d["X"], rng) if perm else d["X"]
    acc = []
    for i in range(0, n, 2000):
        o = N.forward(p, jnp.asarray(X[i:i + 2000])); acc.append(np.asarray(o["answer"]).argmax(-1) == d["Y"][i:i + 2000])
    acc = float(np.concatenate(acc).mean())
    return dict(acc=acc, bayes_known_groups=float((post["post_y"].argmax(-1) == d["Y"]).mean()),
                own=float((post["own_y"].argmax(-1) == d["Y"]).mean()), X=X, Y=d["Y"])
