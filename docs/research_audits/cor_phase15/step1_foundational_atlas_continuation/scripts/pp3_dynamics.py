"""PP3 learning dynamics (mandate §4.3): re-run the atlas T1 s0 pretraining (identical protocol and RNG to nets.train) with
snapshots at steps 250, 500, 1000, 2000, 4000, 6000; check the final snapshot against the stored T1_M_K3_s0 checkpoint;
for each snapshot: T1 accuracy, linear probes at the last-layer input (scripts/pp3_probes.py targets), and the frozen-
layer transfer to T4 with a SHORT fine-tune (1000 steps, layers 0-1 + embeddings frozen, last layer reset; eval 6_300_000,
5000 episodes).  Writes logs/pp3_dynamics.json and checkpoints/dyn/*.npz."""
import json, os, sys, time
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src")); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp, optax, nets as N, bayes as B, ckpt as C, train_ext as TE
out = os.path.join(HERE, "..", "checkpoints", "dyn"); os.makedirs(out, exist_ok=True)
task, steps, seed, K = "T1", 6000, 0, 3; env = B.Env(task, K=K, Q=6); p = N.init("M", env, seed)
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, T):
    l, g = jax.value_and_grad(N.loss_fn)(p, X, T, task, N.forward, False); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * ["T0", "T1", "T2", "T3", "T4", "T5"].index(task) + 31 * K)
snaps = [250, 500, 1000, 2000, 4000, 6000]
for i in range(steps):
    d = B.sample_fast(env, 256, rng); p, st, l = step(p, st, jnp.asarray(d["X"]), N.batch_targets(env, d))
    if i + 1 in snaps: C.save(os.path.join(out, f"T1_s0_step{i + 1}.npz"), jax.tree_util.tree_map(np.asarray, p)); print("snap", i + 1, float(l), flush=True)
ref = C.load(os.path.join(LR, "checkpoints", "main", "T1_M_K3_s0.npz")); fin = C.load(os.path.join(out, "T1_s0_step6000.npz"))
diff = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(fin)))
res = dict(max_param_diff_vs_stored_T1_s0=diff, snaps={})
import pp3_probes_lib as PL
for s in snaps:
    q = C.load(os.path.join(out, f"T1_s0_step{s}.npz")); ev1 = TE.evaluate(q, "T1", 5000, 6_300_001); ev1.pop("X", None); ev1.pop("Y", None)
    pr = PL.probe(jax.tree_util.tree_map(jnp.asarray, q))
    pt, _, _ = TE.train("T4", 0, 1000, p0=q, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True)
    ev4 = TE.evaluate(pt, "T4", 5000, 6_300_000)
    res["snaps"][s] = dict(T1_acc=ev1["acc"], transfer_T4_acc_1000steps=ev4["acc"], probes=pr); print(s, res["snaps"][s], flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "logs", "pp3_dynamics.json"), "w"), indent=1)
