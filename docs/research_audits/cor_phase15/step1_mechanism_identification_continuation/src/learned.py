"""Small learned models (PREREGISTRATION_LEARNED_MODELS.md §3-§4). JAX on CPU.

Model: tokens [A, B, (P), R]. Source token = value-encoder(value) + identity embedding.
Pre-norm causal Transformer, final LN, unembedding at R.
`forward` takes an optional hook(name, array) -> array used by the analysis for interventions.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semantics import sample_binary, sample_k  # noqa: E402

jax.config.update("jax_enable_x64", False)

ARCH = {"TF1": dict(L=1, mlp=True), "TF2": dict(L=2, mlp=True), "TF2AO": dict(L=2, mlp=False)}
D_MODEL, HEADS, DH, MLPW = 16, 2, 8, 32
STEPS, BATCH, LR = 3000, 512, 3e-3
TASKS = ["T1", "T2", "T3", "T4", "T5a", "T5b"]


def n_classes(task):
    return 3 if task.startswith("T5") else 2


def n_tokens(task):
    return 4 if task in ("T3", "T5b") else 3


def init(task, arch, seed):
    k = jax.random.PRNGKey(seed)
    ks = iter(jax.random.split(k, 64))
    n = lambda shape, s: s * jax.random.normal(next(ks), shape)
    K = n_classes(task)
    vin = 3 if task.startswith("T5") else 1
    p = dict(Wval=n((vin, D_MODEL), 0.1), Eid=n((4, D_MODEL), 0.1),
             Wp=n((3 if task == "T5b" else 1, D_MODEL), 0.1),
             gf=jnp.ones(D_MODEL), bf=jnp.zeros(D_MODEL), WU=n((D_MODEL, K), 0.1), bU=jnp.zeros(K), layers=[])
    for _ in range(ARCH[arch]["L"]):
        lay = dict(g1=jnp.ones(D_MODEL), b1=jnp.zeros(D_MODEL),
                   WQ=n((HEADS, D_MODEL, DH), 0.02), WK=n((HEADS, D_MODEL, DH), 0.02),
                   WV=n((HEADS, D_MODEL, DH), 0.02), WO=n((HEADS, DH, D_MODEL), 0.02), bO=jnp.zeros(D_MODEL))
        if ARCH[arch]["mlp"]:
            lay.update(g2=jnp.ones(D_MODEL), b2=jnp.zeros(D_MODEL), W1=n((D_MODEL, MLPW), 0.1),
                       c1=jnp.zeros(MLPW), W2=n((MLPW, D_MODEL), 0.1), c2=jnp.zeros(D_MODEL))
        p["layers"].append(lay)
    return p


def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(v + eps) * g + b


def embed(p, task, inp):
    """inp: dict with 'a','b' (N,) and optional 'p' (N,) for binary; 'zA','zB' (N,3) and 'p' (N,) int for T5."""
    if task.startswith("T5"):
        A = inp["zA"] @ p["Wval"] + p["Eid"][0]
        B = inp["zB"] @ p["Wval"] + p["Eid"][1]
    else:
        A = inp["a"][:, None] @ p["Wval"] + p["Eid"][0]
        B = inp["b"][:, None] @ p["Wval"] + p["Eid"][1]
    toks = [A, B]
    if task == "T3":
        toks.append(inp["p"][:, None] @ p["Wp"] + p["Eid"][2])
    if task == "T5b":
        toks.append(jax.nn.one_hot(inp["p"], 3) @ p["Wp"] + p["Eid"][2])
    N = A.shape[0]
    toks.append(jnp.broadcast_to(p["Eid"][3], (N, D_MODEL)))
    return jnp.stack(toks, 1)


def forward(p, task, inp, hook=None, X=None):
    hk = (lambda name, a: a) if hook is None else hook
    x = embed(p, task, inp) if X is None else X
    x = hk("embed", x)
    T = x.shape[1]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for l, lay in enumerate(p["layers"]):
        z = ln(x, lay["g1"], lay["b1"])
        q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"])
        k = jnp.einsum("ntd,hde->nhte", z, lay["WK"])
        v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        q = hk(f"q{l}", q); k = hk(f"k{l}", k); v = hk(f"v{l}", v)
        s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(DH)
        s = jnp.where(mask, s, -1e9)
        s = hk(f"scores{l}", s)
        a = jax.nn.softmax(s, -1)
        a = hk(f"pattern{l}", a)
        ho = jnp.einsum("nhts,nhse,hed->nhtd", a, v, lay["WO"])
        ho = hk(f"head_out{l}", ho)
        x = x + ho.sum(1) + lay["bO"]
        x = hk(f"resid_mid{l}", x)
        if "W1" in lay:
            z2 = ln(x, lay["g2"], lay["b2"])
            hid = jax.nn.gelu(z2 @ lay["W1"] + lay["c1"], approximate=True)
            hid = hk(f"mlp_hidden{l}", hid)
            x = x + hid @ lay["W2"] + lay["c2"]
        x = hk(f"resid_post{l}", x)
    zf = ln(x[:, -1], p["gf"], p["bf"])
    zf = hk("final", zf)
    return zf @ p["WU"] + p["bU"]


def margin(logits):
    return logits[:, 0] - logits[:, 1]


def to_inp(task, bt):
    if task.startswith("T5"):
        return dict(zA=jnp.asarray(bt.zA, jnp.float32), zB=jnp.asarray(bt.zB, jnp.float32), p=jnp.asarray(bt.p))
    return dict(a=jnp.asarray(bt.a, jnp.float32), b=jnp.asarray(bt.b, jnp.float32), p=jnp.asarray(bt.p, jnp.float32))


def soft_target(task, bt):
    if task.startswith("T5"):
        return jnp.asarray(bt.soft, jnp.float32)
    pc = jnp.asarray(bt.soft, jnp.float32)
    return jnp.stack([pc, 1 - pc], 1)


def sample(task, n, rng):
    return sample_k(task, n, rng) if task.startswith("T5") else sample_binary(task, n, rng)


def train(task, arch, seed, log=print):
    p = init(task, arch, seed)
    sched = optax.cosine_decay_schedule(LR, STEPS)
    opt = optax.adam(sched)
    st = opt.init(p)

    def loss(p, inp, tgt):
        lg = forward(p, task, inp)
        return -(tgt * jax.nn.log_softmax(lg)).sum(-1).mean()

    @jax.jit
    def step(p, st, inp, tgt):
        l, g = jax.value_and_grad(loss)(p, inp, tgt)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l

    rng = np.random.default_rng(10_000 * (seed + 1) + TASKS.index(task))
    t0 = time.time()
    for i in range(STEPS):
        bt = sample(task, BATCH, rng)
        p, st, l = step(p, st, to_inp(task, bt), soft_target(task, bt))
    return jax.tree_util.tree_map(np.asarray, p), float(l), time.time() - t0


def accuracy(p, task, bt):
    lg = np.asarray(forward(p, task, to_inp(task, bt)))
    if task.startswith("T5"):
        return float((lg.argmax(1) == bt.y).mean())
    return float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == bt.y).mean())


# ------------------------------------------------------------------ baselines


def baseline_features(kind, task, inp):
    if task.startswith("T5"):
        zA, zB = inp["zA"], inp["zB"]
        P = jax.nn.one_hot(inp["p"], 3) if task == "T5b" else jnp.zeros((zA.shape[0], 3))
        if kind == "B-sum":
            return jnp.concatenate([zA + zB, P], 1)
        return jnp.concatenate([zA, zB, P], 1)
    a, b, pp = inp["a"][:, None], inp["b"][:, None], inp["p"][:, None]
    if kind == "B-sum":
        return jnp.concatenate([a + b, pp], 1)
    return jnp.concatenate([a, b, pp], 1)


def train_baseline(kind, task, seed):
    K = n_classes(task)
    k = iter(jax.random.split(jax.random.PRNGKey(seed + 777), 16))
    nin = 6 if (task.startswith("T5") and kind == "B-sum") else (9 if task.startswith("T5") else (2 if kind == "B-sum" else 3))
    if kind in ("B-sum", "B-lin"):
        p = dict(W=0.1 * jax.random.normal(next(k), (nin, K)), c=jnp.zeros(K))
        f = lambda p, inp: baseline_features(kind, task, inp) @ p["W"] + p["c"]
    elif kind == "B-MLP":
        p = dict(W1=0.3 * jax.random.normal(next(k), (nin, 32)), c1=jnp.zeros(32),
                 W2=0.3 * jax.random.normal(next(k), (32, 32)), c2=jnp.zeros(32),
                 W3=0.3 * jax.random.normal(next(k), (32, K)), c3=jnp.zeros(K))
        def f(p, inp):
            h = jax.nn.gelu(baseline_features(kind, task, inp) @ p["W1"] + p["c1"], approximate=True)
            h = jax.nn.gelu(h @ p["W2"] + p["c2"], approximate=True)
            return h @ p["W3"] + p["c3"]
    elif kind == "B-bil":
        p = dict(M=0.1 * jax.random.normal(next(k), (K, 3, 3)), W=0.1 * jax.random.normal(next(k), (9, K)), c=jnp.zeros(K))
        def f(p, inp):
            bil = jnp.einsum("ni,kij,nj->nk", inp["zA"], p["M"], inp["zB"])
            return bil + baseline_features("B-lin", task, inp) @ p["W"] + p["c"]
    else:
        raise ValueError(kind)
    opt = optax.adam(optax.cosine_decay_schedule(LR, STEPS)); st = opt.init(p)

    @jax.jit
    def step(p, st, inp, tgt):
        l, g = jax.value_and_grad(lambda p: -(tgt * jax.nn.log_softmax(f(p, inp))).sum(-1).mean())(p)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(20_000 * (seed + 1) + TASKS.index(task))
    for _ in range(STEPS):
        bt = sample(task, BATCH, rng)
        p, st, l = step(p, st, to_inp(task, bt), soft_target(task, bt))
    return p, f


def run_all(seeds, tag):
    root = HERE.parent
    ck = root / "checkpoints"; ck.mkdir(exist_ok=True)
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    T0 = time.time()
    res = {}
    for task in TASKS:
        val = sample(task, 20000, np.random.default_rng(1000))
        for seed in seeds:
            for arch in ARCH:
                p, l, dt = train(task, arch, seed)
                acc = accuracy(p, task, val)
                with open(ck / f"{task}_{arch}_s{seed}.pkl", "wb") as fh:
                    pickle.dump(p, fh)
                res[f"{task}/{arch}/s{seed}"] = dict(acc=acc, final_loss=l, sec=dt)
                log(f"{task} {arch} seed {seed}: val acc {acc:.4f}  final loss {l:.4f}  {dt:.1f}s")
            for kind in (["B-sum", "B-lin", "B-MLP"] + (["B-bil"] if task.startswith("T5") else [])):
                bp, f = train_baseline(kind, task, seed)
                lg = np.asarray(f(bp, to_inp(task, val)))
                acc = float((lg.argmax(1) == val.y).mean()) if task.startswith("T5") else \
                    float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == val.y).mean())
                res[f"{task}/{kind}/s{seed}"] = dict(acc=acc)
                with open(ck / f"{task}_{kind}_s{seed}.pkl", "wb") as fh:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, bp), fh)
                log(f"{task} {kind} seed {seed}: val acc {acc:.4f}")
    log(f"total wall time {time.time() - T0:.1f}s")
    (root / "logs" / f"learned_train_{tag}.txt").write_text("\n".join(lines) + "\n")
    (root / "logs" / f"learned_train_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    tag = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(",")]
    run_all(seeds, tag)
