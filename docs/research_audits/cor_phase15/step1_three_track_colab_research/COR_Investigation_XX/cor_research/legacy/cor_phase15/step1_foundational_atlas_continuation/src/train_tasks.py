"""Training loop for the continuation's new tasks (same optimiser/schedule as nets.train: AdamW lr 1e-3, warmup
min(200, steps/10), cosine decay, wd 1e-4, batch 256, fresh samples every step)."""
import os, sys, time
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability", "src")
sys.path.insert(0, HERE); sys.path.insert(0, LR); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_computation_atlas", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import nets as N, bayes as B, fwd as F, tasks_ext as T

SAMPLERS = {"plocal": T.sample_plocal, "pgroup": T.sample_pgroup, "pcount": T.sample_pcount, "gate": T.sample_gate, "trans": T.sample_trans}
EXTRA = {"pgroup": 3 * 2, "pcount": 3 * 6}


def init(task, seed, size="M"):
    p = N.init(size, B.Env("T1", K=3, Q=6), seed); k = jax.random.PRNGKey(seed + 99)
    d = p["Eread"].shape[0]
    if task == "trans":
        p["Eval"] = 0.5 * jax.random.normal(k, (T.K + 1, d))
    if task in EXTRA:
        p["WX"] = 0.1 * jax.random.normal(jax.random.fold_in(k, 1), (d, EXTRA[task]))
    return p


def readout_feats(p, X):
    rec = {}
    def hk(n, a):
        if n == "final": rec[n] = a
        return a
    o = F.forward(p, X, hk); return o, rec["final"][:, 36:42]


def loss(p, X, Y, Tg, task):
    o, z = readout_feats(p, X)
    if task in ("plocal", "gate", "trans"):
        lp = jax.nn.log_softmax(o["answer"], -1); return -jnp.take_along_axis(lp, Y[..., None], -1).mean()
    out = z @ p["WX"]
    if task == "pgroup":
        lp = jax.nn.log_softmax(out.reshape(out.shape[:2] + (3, 2)), -1)
    else:
        lp = jax.nn.log_softmax(out.reshape(out.shape[:2] + (3, 6)), -1)
    return -jnp.take_along_axis(lp, Tg[..., None], -1).mean()


def train(task, seed, steps, log=None, size="M"):
    p = init(task, seed, size); samp = SAMPLERS[task]
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)

    @jax.jit
    def step(p, st, X, Y, Tg):
        l, g = jax.value_and_grad(loss)(p, X, Y, Tg, task); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(31_000_000 + 1000 * seed + list(SAMPLERS).index(task)); hist = []; t0 = time.time()
    for i in range(steps):
        d = samp(256, rng); Tg = d.get("T", np.zeros((256, 6, 3), int))
        p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), jnp.asarray(Tg))
        if (i + 1) % max(steps // 10, 1) == 0:
            hist.append((i + 1, float(l)))
            if log: log(f"{task} s{seed} step {i + 1} loss {float(l):.4f}")
    return jax.tree_util.tree_map(np.asarray, p), hist, time.time() - t0


def evaluate(p, task, n=6000, seed=0):
    d = SAMPLERS[task](n, np.random.default_rng(seed)); pj = jax.tree_util.tree_map(jnp.asarray, p)
    f = jax.jit(lambda X: readout_feats(pj, X))
    outs, zs = [], []
    for i in range(0, n, 1000):
        o, z = f(jnp.asarray(d["X"][i:i + 1000])); outs.append(np.asarray(o["answer"])); zs.append(np.asarray(z))
    A = np.concatenate(outs); Z = np.concatenate(zs); r = {}
    if task in ("plocal", "gate", "trans"):
        r["acc"] = float((A.argmax(-1) == d["Y"]).mean())
    if task in EXTRA:
        out = Z @ p["WX"]; k = 2 if task == "pgroup" else 6
        r["aux_acc"] = float((out.reshape(out.shape[:2] + (3, k)).argmax(-1) == d["T"]).mean())
    return r
