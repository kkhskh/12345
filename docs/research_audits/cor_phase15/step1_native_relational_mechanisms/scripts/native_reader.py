"""Workstream A3: native-state-only readers on FROZEN parent states (resid_pre2).  No engineered side information at training or inference.
Usage: native_reader.py PARENT KIND [STEPS=8000] [SEED=0] [TARGET=label|bayes] [UPTO=2] [EVAL=dev|<path>]
Training pool: 50_000 generator episodes (stream 31_000_000 + 1000*SEED + hash(parent)), all 6 decisions each; states computed once.
Writes checkpoints/native/<parent>_<kind>_s<STEPS>_seed<SEED>[_bayes][_u<UPTO>].{npz,json}."""
import json, os, sys, time, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import native as NV
parent, kind = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
target = sys.argv[5] if len(sys.argv) > 5 else "label"; upto = int(sys.argv[6]) if len(sys.argv) > 6 else 2
evname = sys.argv[7] if len(sys.argv) > 7 else "dev"
tag = f"{parent}_{kind}_s{STEPS}_seed{seed}" + ("_bayes" if target == "bayes" else "") + (f"_u{upto}" if upto != 2 else "") + ("" if evname == "dev" else "_" + os.path.basename(evname).split(".")[0])
OUT = os.path.join(HERE, "..", "checkpoints", "native"); os.makedirs(OUT, exist_ok=True)
pf = NV.load_parent(parent); t0 = time.time()
rng = np.random.default_rng(31_000_000 + 1000 * seed + zlib.crc32(parent.encode()) % 1000)
NP = 50_000; d = NV.Bm.sample_fast(NV.ENV, NP, rng); Xp = d["X"]; Yp = d["Y"]
Pp = NV.NL.targets(d)["P"] if target == "bayes" else None
Zp = NV.states_np(pf, Xp, upto, dtype=np.float16)                      # NP,43,64
print(f"[{tag}] pool states {Zp.shape} in {time.time() - t0:.0f}s", flush=True)
p = NV.reader_init(kind, seed); npar = NV.count_params(p)
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(300, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, src, ro, al, y, P):
    def loss(p):
        lp = jax.nn.log_softmax(NV.reader_apply(kind, p, src, ro, al), -1)
        return (-jnp.take_along_axis(lp, y[:, None], -1)[:, 0]).mean() if target == "label" else -(P * lp).sum(-1).mean()
    l, g = jax.value_and_grad(loss)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
fwd = jax.jit(lambda p, src, ro, al: NV.reader_apply(kind, p, src, ro, al))
def fn_of(p):
    def f(X):
        Z = NV.states_np(pf, X, upto); out = np.zeros((len(X), 6, 3), np.float32)
        for q in range(6):
            src, ro, _ = NV.decision_tokens(Z, np.full(len(X), q)); out[:, q] = np.asarray(fwd(p, jnp.asarray(src), jnp.asarray(ro), jnp.asarray(Z[:, :36])))
        return out
    return f
ev = NV.load_eval() if evname == "dev" else NV.load_eval(evname); evals = {}; hist = []; brng = np.random.default_rng(seed + 5)
for i in range(STEPS):
    e = brng.integers(0, NP, 512); q = brng.integers(0, 6, 512)
    Ze = Zp[e]; src, ro, _ = NV.decision_tokens(Ze, q)
    P = jnp.asarray(Pp[e, q]) if Pp is not None else jnp.zeros((512, 3))
    p, st, l = step(p, st, jnp.asarray(src, jnp.float32), jnp.asarray(ro, jnp.float32), jnp.asarray(Ze[:, :36], jnp.float32), jnp.asarray(Yp[e, q]), P)
    if (i + 1) % max(STEPS // 20, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in (1000, STEPS // 4, STEPS):
        r = NV.ER.run(fn_of(p), ev); s = NV.summarize(r); evals[i + 1] = dict(summary=s, families=r["families"], quality=r["quality"])
        print(f"[{tag}] step {i + 1} " + " ".join(f"{k} {v:+.3f}" for k, v in s.items()), flush=True)
leaves, tree = jax.tree_util.tree_flatten(p); np.savez(os.path.join(OUT, tag + ".npz"), *[np.asarray(x) for x in leaves])
json.dump(dict(parent=parent, kind=kind, steps=STEPS, seed=seed, target=target, upto=upto, params=npar, seconds=time.time() - t0, hist=hist, evals=evals),
          open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
