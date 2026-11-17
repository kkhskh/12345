"""XV learning-signal diagnostic (exploratory, P5).  For a checkpoint, the gate-attributable expected gradient:
  g_full  = mean gradient of the training loss on B batches of the true generator;
  g_null  = mean gradient on the SAME evidence/labels with the gate sources redrawn so that g is independent of the evidence regime
            (marginals of every source unchanged; the trust relation destroyed);
  signal  = || g_full - g_null ||  (the part of the expected gradient that exists because of the relation),
  noise   = per-batch std of (g_full_b - g_null_b) / sqrt(B), and the ratio signal / ||g_full||.
Usage: xv_gradsig.py COND RUNDIR STEPS(comma; 0 = initialisation) SEED OUT.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp, train_tasks as TT, xvtask as XT, gatelib as GL
cond, rd, steps, seed, outp = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5]
a, b, nu = int(cond.split(":")[0]), float(cond.split(":")[1]), float(cond.split(":")[2]); K = 3
gfun = jax.jit(jax.grad(lambda p, X, Y: TT.loss(p, X, Y, jnp.zeros((X.shape[0], 6, 3), jnp.int32), "gate")))
def flat(g): return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(g)])
def nullgate(X, rng):
    """redraw gate sources with an independent gate state (P = 1/2), keeping each source's marginal."""
    Xn = X.copy(); n = X.shape[:2]; g = rng.random(n) < 0.5
    for s in range(a - 1): Xn[:, :, s] = rng.integers(0, K, n)
    part = Xn[:, :, :a - 1].sum(-1) % K; Xn[:, :, a - 1] = np.where(g, (-part) % K, ((-part) % K + rng.integers(1, K, n)) % K); return Xn
res = dict(cond=cond, run=rd, seed=seed, points=[])
for k in steps:
    p = TT.init("gate", seed) if k == 0 else GL.load(os.path.join(rd, f"step{k}.npz")); rng = np.random.default_rng(97_700_000 + k); D, F = [], []
    for bi in range(16):
        d = XT.sample(256, rng, a, b, nu); gf = flat(gfun(p, jnp.asarray(d["X"]), jnp.asarray(d["Y"]))); gn = flat(gfun(p, jnp.asarray(nullgate(d["X"], rng)), jnp.asarray(d["Y"])))
        D.append(gf - gn); F.append(gf)
    D, F = np.array(D), np.array(F); sig = float(np.linalg.norm(D.mean(0))); noise = float(np.sqrt((D.var(0)).sum() / len(D)))
    res["points"].append(dict(step=k, gate_signal=sig, gate_noise=noise, gate_snr=sig / noise, gate_frac=sig / float(np.linalg.norm(F.mean(0))), grad_norm=float(np.linalg.norm(F.mean(0)))))
    print(res["points"][-1], flush=True)
json.dump(res, open(outp, "w"), indent=1)
