"""XVI deterministic replay: re-run a frozen-recipe training run (scripts/xvi_train.py, same code path: same jit step, schedule, optimizer,
data stream and continuation skip) WITHOUT diagnostics, and save parameter snapshots at the requested steps.  Used (i) to regenerate
exact checkpoints near transitions for the exploratory probes (the confirmation runs saved parameters every 500 steps only) and
(ii) for the reproducibility check against archived snapshots (bitwise).
Usage: xvi_replay.py COND SEED OUTDIR SCHED STEPS STOP SAVE_STEPS(comma list) [INIT_NPZ]   (STEPS = the run's schedule length)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import train_tasks as TT, xvitask as XT, xvidiag as XD, gatelib as GL
cond, seed, OUT, SCHED, STEPS, STOP = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
SAVE = {int(x) for x in sys.argv[7].split(",") if x}; INIT = sys.argv[8] if len(sys.argv) > 8 else None
a, b1, b0, nu = XD.parse(cond); LR, WU = 1e-3, 200; os.makedirs(OUT, exist_ok=True)
sched = optax.warmup_cosine_decay_schedule(0.0, LR, WU, STEPS) if SCHED == "cosine" else (lambda c: LR * jnp.minimum(1.0, c / WU))
p = TT.init("gate", seed) if INIT is None else GL.load(INIT)
opt = optax.adamw(sched, weight_decay=1e-4); st = opt.init(p)
gn = lambda t: jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree_util.tree_leaves(t)))
@jax.jit
def step(p, st, X, Y, Tg):
    l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, "gate"); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l, gn(g), gn(up)
rng = np.random.default_rng(41_000_000 + 1000 * seed)
if SCHED == "cont":
    for _ in range(4000): XT.sample(256, rng, a, b1, b0, nu)
Tg0 = jnp.zeros((256, 6, 3), jnp.int32)
for i in range(STOP):
    d = XT.sample(256, rng, a, b1, b0, nu); p, st, *_ = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), Tg0)
    if (i + 1) in SAVE: np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(p)])
print("replayed", cond, seed, SCHED, STOP, sorted(SAVE))
