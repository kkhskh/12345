"""Reader D (positive control, oracle features): an MLP (45 -> 256 -> 256 -> 3) on generator-derived sufficient features of a decision
(leave-q counts one-hot 3x6, eligibility m_g one-hot 3x3, q-reports one-hot 6x3), trained with exact-Bayes soft targets, same optimiser,
batch and step budget (4000) as the neural readers.  Shows whether the R1 rule is learnable in this budget when the needed statistic
is given explicitly (not an equally informed neural competitor).  Also a variant WITHOUT m (counts + q-reports only: 36 features)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import nnlib as NL, r1lib as L, evalrun as ER
E = L.E
def feats(X, with_m=True):
    n = E.loo(X); V = np.stack([L.votes(X, qq) for qq in range(6)], 1); perf = (n == 5) & (V >= 0); m = perf.sum(-1, keepdims=True) - perf
    f = [np.eye(6)[n].reshape(*n.shape[:2], 18), np.eye(3)[X].reshape(*X.shape[:2], 18)]
    if with_m: f.insert(1, np.eye(3)[np.minimum(m, 2)].reshape(*n.shape[:2], 9))
    return np.concatenate(f, -1).astype(np.float32)                 # N,6,F
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_dev.npz"), allow_pickle=True)); out = {}
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000; VARIANTS = (True, False) if STEPS == 4000 else (False,)
for with_m in VARIANTS:
    F = 45 if with_m else 36; params = NL.init_mlp(jax.random.PRNGKey(5), [F, 256, 256, 3])
    def logit_fn(t, X): return NL.mlp(t, X)                          # X here is the feature array (N,6,F)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, STEPS), weight_decay=1e-4); st = opt.init(params)
    @jax.jit
    def step(p, st, Fx, P):
        def loss(p): return -(P * jax.nn.log_softmax(logit_fn(p, Fx), -1)).sum(-1).mean()
        l, g = jax.value_and_grad(loss)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(17_500_000)
    for i in range(STEPS):
        d = NL.Bm.sample_fast(NL.ENV, 256, rng); T = NL.targets(d); params, st, l = step(params, st, jnp.asarray(feats(d["X"], with_m)), T["P"])
    fn = lambda X: np.asarray(logit_fn(params, jnp.asarray(feats(X, with_m))))
    r = ER.run(fn, ev); out["with_m" if with_m else "without_m"] = r; h = ER.headline(r)
    print("oracle", STEPS, "with_m" if with_m else "without_m", " ".join(f"{k} {v:+.3f}" for k, v in h.items()), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", f"oracle_reader{"" if STEPS == 4000 else "_" + str(STEPS)}.json"), "w"), indent=1)
