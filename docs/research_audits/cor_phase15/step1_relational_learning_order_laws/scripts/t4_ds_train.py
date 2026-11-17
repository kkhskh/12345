"""XV Workstream E: fresh DeepSets T4 models with dense snapshots.  Identical to step1_latent_reliability nets.train("T4", "M", seed, 6000,
arch="deepsets") (init_base, AdamW 1e-3 warmup 200 cosine to 6000, wd 1e-4, batch 256, Q = 6, K = 3, data rng 1_000_003*(seed+1)+7919*4+93),
except that parameters are saved at snapshots (every 50 steps to 3000, every 250 to 6000) and, with BLIND=1, no loss is printed or kept.
Usage: t4_ds_train.py SEED OUTDIR"""
import json, os, sys, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_latent_reliability", "src"))
import numpy as np, jax, jax.numpy as jnp, optax, nets as N, bayes as B
seed, OUT = int(sys.argv[1]), sys.argv[2]; steps = 6000; os.makedirs(OUT, exist_ok=True)
env = B.Env("T4", K=3, Q=6); p = N.init_base("deepsets", "M", env, seed); fwd = N.make_forward("deepsets", 3)
ih = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(p))).hexdigest()
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, steps), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, T):
    l, g = jax.value_and_grad(N.loss_fn)(p, X, T, "T4", fwd, False); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * 4 + 31 * 3); snaps = set(range(50, 3001, 50)) | set(range(3250, steps + 1, 250)); t0 = time.time()
for i in range(steps):
    d = B.sample_fast(env, 256, rng); p, st, l = step(p, st, jnp.asarray(d["X"]), N.batch_targets(env, d))
    if (i + 1) in snaps: np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(p)])
json.dump(dict(seed=seed, steps=steps, seconds=time.time() - t0, init_sha256=ih, blind=os.environ.get("BLIND") == "1"), open(os.path.join(OUT, "done.json"), "w"))
