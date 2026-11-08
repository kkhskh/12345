"""Episode Transformer (bidirectional, pre-norm) and baselines. JAX on CPU.

Token layout per episode (Q questions, 6 sources):
  source token (q, s) = E_val[x_qs] + E_src[s] + E_q[q]                      positions q*6 + s      (36)
  readout token q     = E_read + E_q[q]                                      positions 36 + q        (Q)
  episode token       = E_ep                                                 position 36 + Q         (1)
Heads (by task):
  answer   : per readout token, K logits                                     (all tasks)
  conf     : per readout token, scalar (sigmoid) regressed to the Bayes max posterior   (T2)
  rel      : episode token, |R| logits supervised with the true R                        (T3)
Hooks: forward(..., hook=fn(name, arr)) at resid_pre{l}, attn_out{l}, pattern{l}, resid_mid{l}, mlp_hidden{l}, resid_post{l}, final.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import bayes as B

SIZES = {"S": dict(d=16, L=2, H=2, mlp=64), "M": dict(d=64, L=3, H=4, mlp=256), "L": dict(d=128, L=4, H=4, mlp=512)}


def ntok(Q):
    return 6 * Q + Q + 1


def init(size, env: B.Env, seed):
    c = SIZES[size]; d, Lr, H, m = c["d"], c["L"], c["H"], c["mlp"]
    ks = iter(jax.random.split(jax.random.PRNGKey(seed), 64))
    n = lambda shape, s: s * jax.random.normal(next(ks), shape)
    K, Q, nR = env.K, env.Q, len(env.R_values) if env.R_values[0] is not None else 1
    p = dict(Eval=n((K, d), 0.5), Esrc=n((6, d), 0.5), Eq=n((Q, d), 0.5), Eread=n((d,), 0.5), Eep=n((d,), 0.5),
             gf=jnp.ones(d), bf=jnp.zeros(d), WA=n((d, K), 0.1), WC=n((d, 1), 0.1), WR=n((d, nR), 0.1), layers=[])
    dh = d // H
    for _ in range(Lr):
        p["layers"].append(dict(g1=jnp.ones(d), b1=jnp.zeros(d), WQ=n((H, d, dh), d ** -0.5), WK=n((H, d, dh), d ** -0.5),
                                WV=n((H, d, dh), d ** -0.5), WO=n((H, dh, d), d ** -0.5), g2=jnp.ones(d), b2=jnp.zeros(d),
                                W1=n((d, m), d ** -0.5), c1=jnp.zeros(m), W2=n((m, d), m ** -0.5), c2=jnp.zeros(d)))
    return p


def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(v + eps) * g + b


def embed(p, X):
    N, Q, S = X.shape
    src = p["Eval"][X] + p["Esrc"][None, None, :, :] + p["Eq"][None, :, None, :]       # N,Q,6,d
    src = src.reshape(N, Q * S, -1)
    rd = jnp.broadcast_to(p["Eread"] + p["Eq"], (N, Q, p["Eread"].shape[0]))
    ep = jnp.broadcast_to(p["Eep"], (N, 1, p["Eep"].shape[0]))
    return jnp.concatenate([src, rd, ep], 1)


def forward(p, X, hook=None):
    hk = (lambda n, a: a) if hook is None else hook
    x = embed(p, X)
    N, T, d = x.shape
    Q = X.shape[1]
    for l, lay in enumerate(p["layers"]):
        x = hk(f"resid_pre{l}", x)
        z = ln(x, lay["g1"], lay["b1"])
        q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"])
        v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1])
        a = hk(f"pattern{l}", jax.nn.softmax(s, -1))
        o = jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
        o = hk(f"attn_out{l}", o)
        x = hk(f"resid_mid{l}", x + o)
        hmid = hk(f"mlp_hidden{l}", jax.nn.gelu(ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True))
        x = hk(f"resid_post{l}", x + hmid @ lay["W2"] + lay["c2"])
    zf = hk("final", ln(x, p["gf"], p["bf"]))
    rd = zf[:, 6 * Q:6 * Q + Q]
    return dict(answer=rd @ p["WA"], conf=jax.nn.sigmoid((rd @ p["WC"])[..., 0]), rel=zf[:, -1] @ p["WR"])


# ---------------------------------------------------------------- baselines (non-Transformer)
BASE_W = {"S": 64, "M": 256, "L": 512}


def _mlp_init(ks, dims, scale=1.0):
    return [dict(W=scale * jax.random.normal(next(ks), (a, b)) * a ** -0.5, b=jnp.zeros(b)) for a, b in zip(dims[:-1], dims[1:])]


def _mlp(layers, x, hook=None, name=""):
    for i, l in enumerate(layers):
        x = x @ l["W"] + l["b"]
        if i < len(layers) - 1:
            x = jax.nn.gelu(x, approximate=True)
            if hook is not None:
                x = hook(f"{name}{i}", x)
    return x


def init_base(arch, size, env, seed):
    w = BASE_W[size]; K, Q = env.K, env.Q
    nR = len(env.R_values)
    ks = iter(jax.random.split(jax.random.PRNGKey(seed), 32))
    if arch == "mlp_full":      # one-hot of the entire report matrix -> all answers
        return dict(net=_mlp_init(ks, [Q * 6 * K, w, w, w, Q * (K + 1) + nR]))
    if arch == "mlp_hist":      # per-question global vote histograms only
        return dict(net=_mlp_init(ks, [Q * K, w, w, w, Q * (K + 1) + nR]))
    if arch == "deepsets":      # per-question encoder, mean-pooled episode summary, per-question readout
        e = max(w // 4, 16)
        return dict(phi=_mlp_init(ks, [6 * K, w, w, e]), rho=_mlp_init(ks, [2 * e + 6 * K, w, w, K + 1]),
                    rel=_mlp_init(ks, [e, w, nR]))
    raise ValueError(arch)


def forward_base(arch, p, X, K, hook=None):
    N, Q, _ = X.shape
    oh = jax.nn.one_hot(X, K)                                   # N,Q,6,K
    if arch in ("mlp_full", "mlp_hist"):
        inp = oh.reshape(N, -1) if arch == "mlp_full" else oh.sum(2).reshape(N, -1)
        o = _mlp(p["net"], inp, hook, "h")
        a = o[:, :Q * K].reshape(N, Q, K); c = jax.nn.sigmoid(o[:, Q * K:Q * K + Q]); r = o[:, Q * K + Q:]
        return dict(answer=a, conf=c, rel=r)
    xq = oh.reshape(N, Q, 6 * K)
    h = _mlp(p["phi"], xq, hook, "phi")                         # N,Q,e
    if hook is not None:
        h = hook("h_q", h)
    m = h.mean(1, keepdims=True)
    if hook is not None:
        m = hook("pool", m)
    o = _mlp(p["rho"], jnp.concatenate([h, jnp.broadcast_to(m, h.shape), xq], -1), hook, "rho")
    return dict(answer=o[..., :K], conf=jax.nn.sigmoid(o[..., K]), rel=_mlp(p["rel"], m[:, 0]))


def make_forward(arch, K):
    if arch == "tf":
        return forward
    return lambda p, X, hook=None: forward_base(arch, p, X, K, hook)


def batch_targets(env, d):
    post = B.episode_posterior(env, d["X"])
    return dict(Y=jnp.asarray(d["Y"]), conf=jnp.asarray(post["post_y"].max(-1), jnp.float32),
                R=jnp.asarray(d["ridx"]))


def loss_fn(p, X, T, task, fwd=None):
    out = (fwd or forward)(p, X)
    lp = jax.nn.log_softmax(out["answer"], -1)
    l = -jnp.take_along_axis(lp, T["Y"][..., None], -1).mean()
    if task == "T2":
        l = l + 4.0 * ((out["conf"] - T["conf"]) ** 2).mean()
    if task == "T3":
        l = l - jnp.take_along_axis(jax.nn.log_softmax(out["rel"], -1), T["R"][:, None], -1).mean()
    return l


def train(task, size, seed, steps, batch=256, lr=1e-3, K=3, Q=6, log=None, arch="tf"):
    env = B.Env(task, K=K, Q=Q)
    p = init(size, env, seed) if arch == "tf" else init_base(arch, size, env, seed)
    fwd = make_forward(arch, K)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps), weight_decay=1e-4)
    st = opt.init(p)

    @jax.jit
    def step(p, st, X, T):
        l, g = jax.value_and_grad(loss_fn)(p, X, T, task, fwd)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * ["T0", "T1", "T2", "T3", "T4", "T5"].index(task) + 31 * K)
    t0 = time.time(); hist = []
    for i in range(steps):
        d = B.sample_fast(env, batch, rng)
        p, st, l = step(p, st, jnp.asarray(d["X"]), batch_targets(env, d))
        if (i + 1) % max(steps // 10, 1) == 0:
            hist.append((i + 1, float(l)))
            if log:
                log(f"   {task} {size} s{seed} step {i + 1} loss {float(l):.4f}")
    return jax.tree_util.tree_map(np.asarray, p), hist, time.time() - t0


def evaluate(p, task, n=20000, seed=999, K=3, Q=6, arch="tf"):
    env = B.Env(task, K=K, Q=Q)
    fwd = make_forward(arch, K)
    d = B.sample_fast(env, n, np.random.default_rng(seed))
    post = B.episode_posterior(env, d["X"])
    out = {}
    accs = []; nll = []; ece = []; confe = []
    for i in range(0, n, 2000):
        o = fwd(p, jnp.asarray(d["X"][i:i + 2000]))
        pr = np.asarray(jax.nn.softmax(o["answer"], -1))
        accs.append((pr.argmax(-1) == d["Y"][i:i + 2000]).mean())
        nll.append(-np.log(np.take_along_axis(pr, d["Y"][i:i + 2000][..., None], -1) + 1e-12).mean())
        confe.append(np.abs(np.asarray(o["conf"]) - post["post_y"][i:i + 2000].max(-1)).mean())
        # expected calibration error (10 bins) of the answer distribution
        c = pr.max(-1).ravel(); ok = (pr.argmax(-1) == d["Y"][i:i + 2000]).ravel()
        bins = np.minimum((c * 10).astype(int), 9)
        ece.append(sum(abs(ok[bins == b].mean() - c[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    bayes_acc = float((post["post_y"].argmax(-1) == d["Y"]).mean())
    own_acc = float((post["own_y"].argmax(-1) == d["Y"]).mean())
    return dict(acc=float(np.mean(accs)), nll=float(np.mean(nll)), ece=float(np.mean(ece)), conf_mae=float(np.mean(confe)),
                bayes_acc=bayes_acc, own_question_bayes_acc=own_acc,
                bayes_nll=float(-np.log(np.take_along_axis(post["post_y"], d["Y"][..., None], -1) + 1e-12).mean()))
