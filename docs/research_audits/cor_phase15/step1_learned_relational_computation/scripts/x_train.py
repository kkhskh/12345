"""Condition X: Transformers trained from scratch on ordinary source tokens (no pretraining, no engineered counts).
Usage: x_train.py NBLOCKS TARGET STEPS SEED      NBLOCKS 1|3 (architecture "M": d=64, 4 heads; embeddings as nets.embed)
TARGET label|bayes.  All parameters trainable; data stream 20_000_000 + 1000*SEED; AdamW warmup-cosine 1e-3.  Eval: Inv VII eval_dev."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import v8lib as V
sys.path.insert(0, V.V7); import nnlib as NL, evalrun as ER, ckpt as C
N = NL.N
nb, target, STEPS, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); tag = f"X{nb}_{target}_s{STEPS}_seed{seed}"
OUT = os.path.join(HERE, "..", "checkpoints", "xtok"); os.makedirs(OUT, exist_ok=True)
p0 = N.init("M", NL.ENV, 20_000 + seed); p = dict(Eval=p0["Eval"], Esrc=p0["Esrc"], Eq=p0["Eq"], Eread=p0["Eread"], Eep=p0["Eep"], gf=p0["gf"], bf=p0["bf"], WA=p0["WA"],
    layers=[p0["layers"][i] for i in range(nb)])
def logits(p, X):
    x = N.embed(p, X)
    for i, l in enumerate(p["layers"]): x = NL.block(l, x, l=i)
    return N.ln(x, p["gf"], p["bf"])[:, 36:42] @ p["WA"]
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, Y, P):
    def loss(p):
        lp = jax.nn.log_softmax(logits(p, X), -1)
        return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean() if target == "label" else -(P * lp).sum(-1).mean()
    l, g = jax.value_and_grad(loss)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
fwd = jax.jit(logits); fn = lambda p: (lambda X: np.concatenate([np.asarray(fwd(p, jnp.asarray(X[i:i + 2000]))) for i in range(0, len(X), 2000)]))
ev = dict(np.load(os.path.join(V.V7, "..", "logs", "eval_dev.npz"), allow_pickle=True)); rng = np.random.default_rng(20_000_000 + 1000 * seed); evals = {}; hist = []; t0 = time.time()
for i in range(STEPS):
    d = NL.Bm.sample_fast(NL.ENV, 256, rng); T = NL.targets(d); p, st, l = step(p, st, jnp.asarray(d["X"]), T["Y"], T["P"])
    if (i + 1) % max(STEPS // 20, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in (4000, 16000, STEPS):
        r = ER.run(fn(p), ev); evals[i + 1] = r; h = ER.headline(r); print(f"[{tag}] step {i + 1} " + " ".join(f"{k} {v:+.3f}" for k, v in h.items()), flush=True)
V.save_tree(os.path.join(OUT, tag + ".npz"), jax.tree_util.tree_map(np.asarray, p))
json.dump(dict(nblocks=nb, target=target, steps=STEPS, seed=seed, seconds=time.time() - t0, hist=hist, evals=evals), open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
