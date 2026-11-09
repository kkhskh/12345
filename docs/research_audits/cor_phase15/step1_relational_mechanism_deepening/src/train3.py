"""Generic trainer for Investigation III (same model, optimiser and schedule as the continuation's train_tasks.train:
AdamW lr 1e-3, warmup min(200, steps/10), cosine decay to 0 at `steps`, wd 1e-4, batch 256, fresh samples every step).
Adds: intermediate snapshots, per-question-type evaluation hooks, optional auxiliary map supervision (TRANS) from the
episode token, and an optional initial checkpoint (curricula).
Tasks: trans (all 36 pairs), trans_holdout (24 training pairs), trans_aux (all pairs + sigma/tau auxiliary CE),
       gate, adapt, plocal, pgroup, pcount (continuation samplers)."""
import os, sys, time, json
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp, optax
import tasks3 as T3, tasks_ext as TX, train_tasks as TT, nets as N, bayes as B, fwd as F, ckpt as C

SAMPLERS = {"trans": lambda n, r: T3.sample_trans_pairs(n, r, T3.ALL_PAIRS), "trans_holdout": lambda n, r: T3.sample_trans_pairs(n, r, T3.TRAIN_PAIRS),
            "trans_aux": lambda n, r: T3.sample_trans_pairs(n, r, T3.ALL_PAIRS), "gate": TX.sample_gate, "adapt": T3.sample_adapt,
            "plocal": TX.sample_plocal, "pgroup": TX.sample_pgroup, "pcount": TX.sample_pcount}
EXTRA = {"pgroup": 6, "pcount": 18, "trans_aux": 12}


def init(task, seed):
    p = N.init("M", B.Env("T1", K=3, Q=6), seed); k = jax.random.PRNGKey(seed + 99); d = p["Eread"].shape[0]
    if task.startswith("trans"): p["Eval"] = 0.5 * jax.random.normal(k, (TX.K + 1, d))
    if task in EXTRA: p["WX"] = 0.1 * jax.random.normal(jax.random.fold_in(k, 1), (d, EXTRA[task]))
    return p


def feats(p, X):
    rec = {}
    def hk(n, a):
        if n == "final": rec[n] = a
        return a
    o = F.forward(p, X, hk); return o, rec["final"]


def loss(p, X, Y, Tg, S, Tt, task):
    o, z = feats(p, X); lp = jax.nn.log_softmax(o["answer"], -1)
    if task in ("pgroup", "pcount"):
        out = z[:, 36:42] @ p["WX"]; k = 2 if task == "pgroup" else 6
        l2 = jax.nn.log_softmax(out.reshape(out.shape[:2] + (3, k)), -1); return -jnp.take_along_axis(l2, Tg[..., None], -1).mean()
    l = -jnp.take_along_axis(lp, Y[..., None], -1).mean()
    if task == "trans_aux":
        a = z[:, 42] @ p["WX"]; ls = jax.nn.log_softmax(a[:, :6], -1); lt = jax.nn.log_softmax(a[:, 6:], -1)
        l = l + 0.5 * (-jnp.take_along_axis(ls, S[:, None], -1).mean() - jnp.take_along_axis(lt, Tt[:, None], -1).mean())
    return l


def train(task, seed, steps, log=None, snaps=(), snapdir=None, p0=None, rng_offset=0):
    p = init(task, seed) if p0 is None else jax.tree_util.tree_map(jnp.asarray, p0)
    if p0 is not None and task in EXTRA and "WX" not in p: p["WX"] = init(task, seed)["WX"]
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)
    @jax.jit
    def step(p, st, X, Y, Tg, S, Tt):
        l, g = jax.value_and_grad(loss)(p, X, Y, Tg, S, Tt, task); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(51_000_000 + 1000 * seed + list(SAMPLERS).index(task) + rng_offset); hist = []; t0 = time.time()
    for i in range(steps):
        d = SAMPLERS[task](256, rng); Tg = d.get("T", np.zeros((256, 6, 3), int))
        S = d.get("sigma", np.zeros(256, int)); Tt = d.get("tau", np.zeros(256, int))
        p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), jnp.asarray(Tg), jnp.asarray(S), jnp.asarray(Tt))
        if (i + 1) % max(steps // 20, 1) == 0:
            hist.append((i + 1, float(l)))
            if log: log(f"{task} s{seed} step {i + 1} loss {float(l):.4f}")
        if snapdir and (i + 1) in snaps: C.save(os.path.join(snapdir, f"step{i + 1}.npz"), jax.tree_util.tree_map(np.asarray, p))
    return jax.tree_util.tree_map(np.asarray, p), hist, time.time() - t0


def answer_probs(p, X, bs=2000):
    pj = jax.tree_util.tree_map(jnp.asarray, p); f = jax.jit(lambda Z: jax.nn.softmax(F.forward(pj, Z)["answer"], -1))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)])
