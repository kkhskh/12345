"""Learned-composition experiment, Conditions O / R (per-decision readers).  Usage: compose_train.py READER COND TARGET STEPS SEED
READER linear|mlp|attn|attn2|relnet ; COND R (raw ingredients) | O (raw + hand-computed m) | Rnoid (raw, no group identity)
TARGET label (generator labels; ordinary task supervision) | bayes (exact-Bayes soft targets).  Fresh samples every step from
stream 19_000_000 + 1000*SEED (batch 256 episodes = 1536 decisions); AdamW warmup-cosine lr 1e-3, wd 1e-4.
Evaluated on Investigation VII's development set (logs of step1_r1_causal_origin_and_selective_rescue/eval_dev.npz).
Writes checkpoints/compose/<tag>.npz/.json."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import v8lib as V, readers as RD
sys.path.insert(0, V.V7); import nnlib as NL, evalrun as ER, ckpt as C
reader, cond, target, STEPS, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
kind, depth = ("attn", 2) if reader == "attn2" else (reader, 1)
oracle, drop_id = cond == "O", cond == "Rnoid"
tag = f"{reader}_{cond}_{target}_s{STEPS}_seed{seed}"; OUT = os.path.join(HERE, "..", "checkpoints", "compose_conf" if os.environ.get("PHASE", "dev") == "conf" else "compose"); os.makedirs(OUT, exist_ok=True)
F = 6 + 6 + (0 if drop_id else 3) + (3 if oracle else 0)
p = RD.init(kind, F, seed, depth=depth, H={"relnet": 64}.get(kind, 128), d=40); npar = RD.count_params(p)
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, f, Y, P):
    def loss(p):
        lp = jax.nn.log_softmax(RD.apply(kind, p, f), -1)
        return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean() if target == "label" else -(P * lp).sum(-1).mean()
    l, g = jax.value_and_grad(loss)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(19_000_000 + 1000 * seed); t0 = time.time(); hist = []
CONF = os.environ.get("PHASE", "dev") == "conf"   # confirmation: eval_conf8 (stream 18_900_000), checkpoints/compose_conf
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_conf8.npz") if CONF else os.path.join(V.V7, "..", "logs", "eval_dev.npz"), allow_pickle=True)); evals = {}
fwd = jax.jit(lambda p, f: RD.apply(kind, p, f))
def fn_of(p): return lambda X: np.asarray(fwd(p, jnp.asarray(RD.features(X, oracle, drop_id))))
snaps = sorted(set([s for s in (1000, 4000, 16000, 48000) if s <= STEPS] + [STEPS]))
for i in range(STEPS):
    d = NL.Bm.sample_fast(NL.ENV, 256, rng); T = NL.targets(d)
    p, st, l = step(p, st, jnp.asarray(RD.features(d["X"], oracle, drop_id)), T["Y"], T["P"])
    if (i + 1) % max(STEPS // 20, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in snaps:
        r = ER.run(fn_of(p), ev); evals[i + 1] = r; h = ER.headline(r)
        print(f"[{tag}] params {npar} step {i + 1} " + " ".join(f"{k} {v:+.3f}" for k, v in h.items()), flush=True)
V.save_tree(os.path.join(OUT, tag + ".npz"), jax.tree_util.tree_map(np.asarray, p))
json.dump(dict(reader=reader, cond=cond, target=target, steps=STEPS, seed=seed, params=npar, seconds=time.time() - t0, hist=hist, evals=evals), open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
