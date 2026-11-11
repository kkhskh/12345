"""Investigation VI cohort.  Usage: cohort6_train.py ARM SEED      ARM in {T1, P}
Pretraining with the ORIGINAL protocol of each curriculum (T1: nets.train("T1","M",seed,6000,K=3,arch="tf");
P-count: train_tasks.train("pcount",seed,3000)), then ONE T4 frozen-feature transfer of 4000 steps (identical code to
train_ext.train — copied below with snapshot saving added; layers 0-1 + embeddings frozen, layer 2 reset from
init(seed + 7777), AdamW warmup-cosine over 4000 steps) with snapshots at the fixed schedule 250, 500, 1000, 2000, 4000.
Independence across curricula (mandate §7): T1 seeds 70-77, P-count seeds 80-87 — distinct seeds give distinct initial
weights, data streams and transfer streams in both trainers (verified in scripts/check_independence.py).
Outputs checkpoints/x{ARM}{SEED}/pre.npz, pre.json, step{N}.npz, step{N}.json (20,000-episode T4 evaluation, stream 6_300_000)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src", "step1_relational_mechanism_deepening/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax, jax.numpy as jnp, optax
import nets as N, bayes as Bm, ckpt as C, train_ext as TE
arm, seed = sys.argv[1], int(sys.argv[2]); SNAPS = (250, 500, 1000, 2000, 4000); STEPS = SNAPS[-1]
SMOKE = os.environ.get("COHORT_SMOKE") == "1"
if SMOKE: SNAPS, STEPS = (5, 10), 10
out = os.path.join("/tmp/c6smoke" if SMOKE else os.path.join(HERE, "..", "checkpoints"), f"x{arm}{seed}"); os.makedirs(out, exist_ok=True)
lf = open(os.path.join(out, "train.log"), "a"); log = lambda s: (lf.write(s + "\n"), lf.flush())
if not os.path.exists(os.path.join(out, "pre.npz")):
    if arm == "T1":
        p, h, dt = N.train("T1", "M", seed, 10 if SMOKE else 6000, K=3, log=log, arch="tf"); ev = N.evaluate(p, "T1", K=3, seed=999, arch="tf")
    else:
        import train_tasks as TT
        p, h, dt = TT.train("pcount", seed, 10 if SMOKE else 3000, log=log); ev = TT.evaluate(p, "pcount", seed=41_000_000)
    C.save(os.path.join(out, "pre.npz"), p)
    json.dump(dict(arm=arm, seed=seed, seconds=dt, eval=ev, hist=h), open(os.path.join(out, "pre.json"), "w"), indent=1, default=float)
p0 = C.load(os.path.join(out, "pre.npz")); p0.pop("WX", None)
# ---- identical to train_ext.train(task="T4", seed, STEPS, p0, reinit_layers=(2,), freeze_layers=(0,1), freeze_embed=True) + snapshots
task = "T4"; env = Bm.Env(task, K=3, Q=6); fresh = N.init("M", env, seed + 7777)
p = jax.tree_util.tree_map(jnp.asarray, p0); p["WR"] = fresh["WR"]
if "WC" not in p: p["WC"] = fresh["WC"]
p["layers"][2] = fresh["layers"][2]
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, T):
    l, g = jax.value_and_grad(N.loss_fn)(p, X, T, task)
    for fl in (0, 1): g["layers"][fl] = jax.tree_util.tree_map(jnp.zeros_like, g["layers"][fl])
    for k_ in ["Eval", "Esrc", "Eq", "Eread", "Eep"]: g[k_] = jnp.zeros_like(g[k_])
    up, st = opt.update(g, st, p)
    for fl in (0, 1): up["layers"][fl] = jax.tree_util.tree_map(jnp.zeros_like, up["layers"][fl])
    for k_ in ["Eval", "Esrc", "Eq", "Eread", "Eep"]: up[k_] = jnp.zeros_like(up[k_])
    return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * ["T0", "T1", "T2", "T3", "T4", "T5"].index(task) + 31 * 3)
t0 = time.time(); hist = []
for i in range(STEPS):
    d = Bm.sample_fast(env, 256, rng); p, st, l = step(p, st, jnp.asarray(d["X"]), N.batch_targets(env, d))
    if (i + 1) % max(STEPS // 20, 1) == 0: hist.append((i + 1, float(l))); log(f"T4 s{seed} step {i + 1} loss {float(l):.4f}")
    if (i + 1) in SNAPS:
        q = jax.tree_util.tree_map(np.asarray, p); C.save(os.path.join(out, f"step{i + 1}.npz"), q)
        ev = TE.evaluate(q, "T4", 500 if SMOKE else 20000, 6_300_000); ev.pop("X"); ev.pop("Y")
        json.dump(dict(arm=arm, seed=seed, step=i + 1, seconds=time.time() - t0, eval=ev, hist=list(hist)), open(os.path.join(out, f"step{i + 1}.json"), "w"), indent=1, default=float)
