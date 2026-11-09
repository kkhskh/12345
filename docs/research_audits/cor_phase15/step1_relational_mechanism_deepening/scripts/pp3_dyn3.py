"""Unconfounded PP3 dynamics (mandate A6): re-run the atlas T1 s0 pretraining (identical protocol/RNG to nets.train) to step
2000 with snapshots at 250, 500, 1000, 1100, ..., 2000 (the continuation already holds 4000 and 6000 and showed the re-run is
bit-identical).  For every snapshot: (i) T1 accuracy; (ii) probes INDEPENDENT of fine-tuning (logistic, held-out) at source
tokens and at readout tokens of the last-layer input: agree_q, count_loo, perfect_loo; (iii) frozen-feature transfer with the
PRINCIPAL 2000-step protocol (eval 20,000 episodes stream 6_300_000).  Writes logs/pp3_dyn3.json (incrementally)."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import tasks3  # noqa  (sets paths)
import numpy as np, jax, jax.numpy as jnp, optax, nets as N, bayes as B, ckpt as C, train_ext as TE, gen_family as G, fwd as F
from sklearn.linear_model import LogisticRegression
out = os.path.join(HERE, "..", "checkpoints", "dyn3"); os.makedirs(out, exist_ok=True)
CONT = os.path.join(HERE, "..", "..", "step1_foundational_atlas_continuation", "checkpoints", "dyn")
task, steps, seed, K = "T1", 6000, 0, 3; env = B.Env(task, K=K, Q=6); p = N.init("M", env, seed)
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, Tt):
    l, g = jax.value_and_grad(N.loss_fn)(p, X, Tt, task, N.forward, False); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * ["T0", "T1", "T2", "T3", "T4", "T5"].index(task) + 31 * K)
snaps = [250, 500, 1000, 1200, 1400, 1500, 1600, 1800, 2000]   # budget cut from every 100 steps (COMPUTE_LOG)
if not all(os.path.exists(os.path.join(out, f"step{s}.npz")) for s in snaps):   # resume support (a run was OOM-killed)
    for i in range(2000):
        d = B.sample_fast(env, 256, rng); p, st, l = step(p, st, jnp.asarray(d["X"]), N.batch_targets(env, d))
        if i + 1 in snaps: C.save(os.path.join(out, f"step{i + 1}.npz"), jax.tree_util.tree_map(np.asarray, p))
c2000 = C.load(os.path.join(CONT, "T1_s0_step2000.npz")); n2000 = C.load(os.path.join(out, "step2000.npz"))
diff = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(jax.tree_util.tree_leaves(c2000), jax.tree_util.tree_leaves(n2000)))
f4 = G.Fam(); D = G.sample(f4, 2500, np.random.default_rng(9_100_000))["X"]
A = np.stack([(D[:, :, a] == D[:, :, b]) for a, b in G.GROUPS], -1).astype(int); loo = A.sum(1, keepdims=True) - A
targets = {"agree_q": A, "count_loo": loo, "perfect_loo": (loo == 5).astype(int)}
def probes(q):
    q = jax.tree_util.tree_map(jnp.asarray, q); rec = {}
    def hk(n, a):
        if n == "resid_pre2": rec["r"] = a
        return a
    F.forward(q, jnp.asarray(D), hk); R = np.asarray(rec["r"]); r = {}
    tr = np.arange(len(D)) < 1500
    for site in ("source", "readout"):
        for k, Tt in targets.items():
            if site == "source":
                Z = np.concatenate([R[:, 6 * qq + s] for qq in range(6) for s in range(6)]); t = np.concatenate([Tt[:, qq, s // 2] for qq in range(6) for s in range(6)])
                m = np.concatenate([tr] * 36)
            else:   # readout token q carries all three groups: one probe per group
                accs = []
                for g in range(3):
                    Z = R[:, 36:42].reshape(-1, R.shape[-1]); t = Tt[:, :, g].reshape(-1); m = np.repeat(tr, 6)
                    mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-6
                    accs.append(float(LogisticRegression(max_iter=3000).fit((Z[m] - mu) / sd, t[m]).score((Z[~m] - mu) / sd, t[~m])))
                r[f"{site}|{k}"] = float(np.mean(accs)); continue
            mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-6
            r[f"{site}|{k}"] = float(LogisticRegression(max_iter=3000).fit((Z[m][::3] - mu) / sd, t[m][::3]).score((Z[~m] - mu) / sd, t[~m]))
    return r
LOG = os.path.join(HERE, "..", "logs", "pp3_dyn3.json")
res = json.load(open(LOG)) if os.path.exists(LOG) else dict(snaps={})
res["max_param_diff_step2000_vs_continuation"] = diff
paths = [(s, os.path.join(out, f"step{s}.npz")) for s in snaps] + [(4000, os.path.join(CONT, "T1_s0_step4000.npz")), (6000, os.path.join(CONT, "T1_s0_step6000.npz"))]
for s, path in paths:
    if str(s) in res["snaps"]: continue
    q = C.load(path); ev1 = TE.evaluate(q, "T1", 5000, 6_300_001); ev1.pop("X", None); ev1.pop("Y", None)
    pr = probes(q)
    pt, _, _ = TE.train("T4", 0, 2000, p0=q, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True)
    ev4 = TE.evaluate(pt, "T4", 20000, 6_300_000)
    res["snaps"][s] = dict(T1_acc=ev1["acc"], transfer_T4_acc_2000steps=ev4["acc"], probes=pr); print(s, res["snaps"][s], flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "logs", "pp3_dyn3.json"), "w"), indent=1)
