"""XIV Stage III: GATE training with dense snapshots.  Identical to the continuation trainer train_tasks.train("gate", seed, 4000)
(same init N.init("M", T1 env, seed), AdamW 1e-3 warmup 200 cosine to 4000, wd 1e-4, batch 256, data rng 31_000_000 + 1000*seed + 3),
except that parameters are saved at every snapshot step.  Usage: g_train.py SEED [STOP]
Snapshots: every 25 steps to 1500, every 100 to 2000, every 500 to 4000.  Writes checkpoints/gate/s<SEED>/step<k>.npz + hist.json."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import train_tasks as TT
seed = int(sys.argv[1]); STOP = int(sys.argv[2]) if len(sys.argv) > 2 else 4000; steps = 4000
snaps = set(range(25, 1501, 25)) | set(range(1600, 2001, 100)) | set(range(2500, 4001, 500))
OUT = os.path.join(HERE, "..", "checkpoints", "gate", f"s{seed}" + ("" if STOP == 4000 else f"_stop{STOP}")); os.makedirs(OUT, exist_ok=True)
p = TT.init("gate", seed); samp = TT.SAMPLERS["gate"]
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, Y, Tg):
    l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, "gate"); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(31_000_000 + 1000 * seed + list(TT.SAMPLERS).index("gate")); hist = []; t0 = time.time()
for i in range(STOP):
    d = samp(256, rng); Tg = d.get("T", np.zeros((256, 6, 3), int))
    p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), jnp.asarray(Tg))
    if (i + 1) % 400 == 0: hist.append((i + 1, float(l))); print(f"gate s{seed} step {i + 1} loss {float(l):.4f} {time.time() - t0:.0f}s", flush=True)
    if (i + 1) in snaps:
        leaves, tree = jax.tree_util.tree_flatten(p); np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(a) for a in leaves])
json.dump(dict(seed=seed, stop=STOP, hist=hist, seconds=time.time() - t0, keys=str(jax.tree_util.tree_structure(p))), open(os.path.join(OUT, "hist.json"), "w"))
