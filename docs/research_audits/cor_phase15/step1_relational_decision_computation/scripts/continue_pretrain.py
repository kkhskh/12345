"""Workstream D1/D2: readiness-gated continuation of P-count pretraining.
Loads a P-count parent (pre.npz), continues training with the IDENTICAL loss / optimiser family as train_tasks.train (AdamW, warmup-cosine to
1e-3, wd 1e-4, batch 256, fresh samples) for STEPS more steps with a fresh schedule and a block-specific data stream
(31_000_000 + 1000*seed + 2 + 100_000*block).  Writes OUT/pre.npz, OUT/pre.json (aux_acc on the standard 41_000_000 evaluation stream, per
group), OUT/pretrain.log.  Usage: continue_pretrain.py IN_pre.npz OUT_DIR SEED BLOCK STEPS"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"): sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax, jax.numpy as jnp, optax
import ckpt as C, train_tasks as TT
src, out, seed, block, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); os.makedirs(out, exist_ok=True)
p = jax.tree_util.tree_map(jnp.asarray, C.load(src)); opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, Y, Tg):
    l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, "pcount"); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(31_000_000 + 1000 * seed + 2 + 100_000 * block); lf = open(os.path.join(out, "pretrain.log"), "a"); hist = []; t0 = time.time()
for i in range(steps):
    d = TT.SAMPLERS["pcount"](256, rng); p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), jnp.asarray(d["T"]))
    if (i + 1) % max(steps // 10, 1) == 0: hist.append((i + 1, float(l))); lf.write(f"pcount s{seed} block{block} step {i + 1} loss {float(l):.4f}\n"); lf.flush()
pn = jax.tree_util.tree_map(np.asarray, p); C.save(os.path.join(out, "pre.npz"), pn)
d = TT.SAMPLERS["pcount"](6000, np.random.default_rng(41_000_000)); fn = jax.jit(lambda X: TT.readout_feats(p, X)[1])
Z = np.concatenate([np.asarray(fn(jnp.asarray(d["X"][i:i + 1000]))) for i in range(0, 6000, 1000)]); ok = (Z @ pn["WX"]).reshape(6000, 6, 3, 6).argmax(-1) == d["T"]
ev = dict(aux_acc=float(ok.mean()), per_group=ok.mean((0, 1)).round(4).tolist())
json.dump(dict(arm="P", seed=seed, block=block, steps=steps, source=os.path.relpath(src, os.path.join(HERE, "..")), seconds=time.time() - t0, eval=ev, hist=hist), open(os.path.join(out, "pre.json"), "w"), indent=1)
print(out, ev, f"{time.time() - t0:.0f}s", flush=True)
