"""Investigation VII neural utilities (JAX; the architecture code is Investigation II's nets.py, read-only).
Frozen-feature forward (layers 0-1 + embeddings), downstream readers, generic trainer with stop-gradient on the frozen part
(gradient-equivalent to train_ext.train's zero-masked gradients; verified in scripts/check_sg_equivalence.py), and a uniform
logits interface for evaluation."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax, jax.numpy as jnp, optax
import nets as N, bayes as Bm, ckpt as C
ENV = Bm.Env("T4", K=3, Q=6)
B6 = os.path.join(ROOT, "step1_perfect_agreement_boundary", "checkpoints")
FROZEN_KEYS = ["Eval", "Esrc", "Eq", "Eread", "Eep"]


def dev_ckpt(name, step=4000):
    """'xP80' / 'xT70' -> Investigation VI checkpoint path (development parents)."""
    a, s = name[1], name[2:]; return os.path.join(B6, f"x{'T1' if a == 'T' else 'P'}{s}", f"step{step}.npz" if step else "pre.npz")


def load(path):
    p = C.load(path); p.pop("WX", None); return jax.tree_util.tree_map(jnp.asarray, p)


def block(lay, x, hook=None, l=2):
    hk = (lambda n, a: a) if hook is None else hook
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    a = hk(f"pattern{l}", jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]), -1))
    o = hk(f"attn_out{l}", jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"]))
    x = x + o
    h = hk(f"mlp_hidden{l}", jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True))
    return x + h @ lay["W2"] + lay["c2"]


def frozen_z(pf, X):
    """resid_pre2: embeddings + frozen layers 0-1 (all 43 tokens), under stop_gradient."""
    pf = jax.lax.stop_gradient(pf); x = N.embed(pf, X)
    for l in (0, 1): x = block(pf["layers"][l], x, l=l)
    return x


def tf_head(pd, z, nblocks):
    x = z
    for i in range(nblocks): x = block(pd["blocks"][i], x, l=2 + i)
    zf = N.ln(x, pd["gf"], pd["bf"]); return zf[:, 36:42] @ pd["WA"]


def mlp(layers, x):
    for i, l in enumerate(layers):
        x = x @ l["W"] + l["b"]
        if i < len(layers) - 1: x = jax.nn.gelu(x, approximate=True)
    return x


def init_mlp(key, dims):
    ks = jax.random.split(key, len(dims)); return [dict(W=jax.random.normal(k, (a, b)) * a ** -0.5, b=jnp.zeros(b)) for k, a, b in zip(ks, dims[:-1], dims[1:])]


def reader_init(kind, seed, pparent=None):
    """Downstream readers on the frozen resid_pre2 state Z (N, 43, 64).
    A   : one fresh Transformer block + fresh final LN (g=1, b=0) + fresh WA        (original architecture, everything fresh)
    B   : two fresh Transformer blocks + fresh LN + fresh WA                         (more flexible Transformer readout)
    C   : MLP on the flattened 43x64 state -> 512 -> 512 -> 18                        (non-attention, full state)
    Eread: shared MLP on the readout token 36+q only (64 -> 256 -> 256 -> 3)        (restricted site)
    Esrc: MLP on the 36 source tokens only, flattened (2304 -> 512 -> 512 -> 18)     (restricted site, no readout token)
    Fresh Transformer parameters are drawn with nets.init's distributions (seeded)."""
    fresh = N.init("M", ENV, seed); key = jax.random.PRNGKey(seed + 12345)
    if kind in ("A", "B"):
        blocks = [fresh["layers"][2]] + ([N.init("M", ENV, seed + 1)["layers"][2]] if kind == "B" else [])
        return dict(blocks=blocks, gf=jnp.ones(64), bf=jnp.zeros(64), WA=fresh["WA"])
    if kind == "C": return dict(mlp=init_mlp(key, [43 * 64, 512, 512, 18]))
    if kind == "Eread": return dict(mlp=init_mlp(key, [64, 256, 256, 3]))
    if kind == "Esrc": return dict(mlp=init_mlp(key, [36 * 64, 512, 512, 18]))
    raise ValueError(kind)


def reader_logits(kind, pd, z):
    n = z.shape[0]
    if kind == "A": return tf_head(pd, z, 1)
    if kind == "B": return tf_head(pd, z, 2)
    if kind == "C": return mlp(pd["mlp"], z.reshape(n, -1)).reshape(n, 6, 3)
    if kind == "Eread": return mlp(pd["mlp"], z[:, 36:42])
    if kind == "Esrc": return mlp(pd["mlp"], z[:, :36].reshape(n, -1)).reshape(n, 6, 3)
    raise ValueError(kind)


def targets(d, weights=None):
    """CE targets: generator labels Y (N,6) and exact Bayes posterior (N,6,3) for soft-target (sufficiency) training."""
    post = Bm.episode_posterior(ENV, d["X"])["post_y"]
    return dict(Y=jnp.asarray(d["Y"]), P=jnp.asarray(post, jnp.float32), w=jnp.ones(d["Y"].shape, jnp.float32) if weights is None else jnp.asarray(weights, jnp.float32))


def make_loss(logit_fn, target):
    def loss(params, X, T):
        lp = jax.nn.log_softmax(logit_fn(params, X), -1)
        if target == "label": l = -jnp.take_along_axis(lp, T["Y"][..., None], -1)[..., 0]
        elif target == "bayes": l = -(T["P"] * lp).sum(-1)
        else: raise ValueError(target)
        return (l * T["w"]).sum() / T["w"].sum()
    return loss


def train(logit_fn, params, steps, seed_stream, target="label", lr=1e-3, wd=1e-4, batch=256, weight_fn=None, log=None, snaps=(), on_snap=None, sampler=None, xfeat=None):
    """AdamW warmup-cosine (as train_ext); fresh generator samples every step from rng(seed_stream).  weight_fn(d) -> (N,6) per-decision
    loss weights (training-time supervision only)."""
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps), weight_decay=wd); st = opt.init(params)
    loss = make_loss(logit_fn, target)
    @jax.jit
    def step(p, st, X, T):
        l, g = jax.value_and_grad(loss)(p, X, T); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(seed_stream); hist = []
    for i in range(steps):
        d = (sampler or (lambda n, r: Bm.sample_fast(ENV, n, r)))(batch, rng); T = targets(d, None if weight_fn is None else weight_fn(d))
        Xin = jnp.asarray(d["X"]) if xfeat is None else (jnp.asarray(d["X"]), jnp.asarray(xfeat(d["X"])))
        params, st, l = step(params, st, Xin, T)
        if (i + 1) % max(steps // 20, 1) == 0:
            hist.append((i + 1, float(l)))
            if log: log(f"step {i + 1} loss {float(l):.4f}")
        if (i + 1) in snaps and on_snap: on_snap(i + 1, params)
    return params, hist


def full_model_logits(p):
    f = jax.jit(lambda X: N.forward(p, X)["answer"])
    return lambda X: np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 2000]))) for i in range(0, len(X), 2000)])


def reader_model_logits(kind, pf, pd):
    f = jax.jit(lambda X: reader_logits(kind, pd, frozen_z(pf, X)))
    return lambda X: np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 2000]))) for i in range(0, len(X), 2000)])
