"""XVI learning-signal estimator (rebuilt after XV Review B finding 1; mandate D5).  EXPLORATORY.
Quantity: mu = E[ grad L(true batch) - grad L(gate-randomised batch) ], where the gate-randomised batch is the SAME batch with the gate
sources redrawn independently (free sources uniform, last source chosen so that a fresh g' ~ Bernoulli(1/2) is visible): every source
marginal, every label and the evidence are unchanged; only the dependence between the visible gate and the evidence regime is removed.
Estimator: with B independent paired differences D_i, U = (||sum_i D_i||^2 - sum_i ||D_i||^2) / (B (B - 1)) is UNBIASED for ||mu||^2
(E[D_i . D_j] = ||mu||^2 for i != j).  Its standard error is the jackknife over the B pairs.  (XV's estimator ||mean D|| / noise has
expectation about 1 when mu = 0; this one has expectation exactly 0.)
Controls, computed on the same checkpoint:
  null      D_i = grad L(rand_i) - grad L(rand'_i), two independent gate-randomisations: mu = 0 exactly.
  planted   null D_i + v for a fixed random vector v with ||v||^2 set to 10 x the null standard error: U must recover ||v||^2.
  label     D_i = grad L(true_i) - grad L(true_i with labels permuted across the batch): a natural positive control (the source-label
            signal), nonzero at initialisation.
Also: Utot = the same unbiased estimate of ||E grad L(true)||^2 (the full expected gradient).
Usage: xvi_gradsig2.py COND CKPT OUT.json [B=48] [SEED=0]   (CKPT 'init:SEED' for a fresh initialisation)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp
import train_tasks as TT, xvitask as XT, xvidiag as XD, gatelib as GL
cond, ck, outp = sys.argv[1], sys.argv[2], sys.argv[3]; B = int(sys.argv[4]) if len(sys.argv) > 4 else 48; SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
a, b1, b0, nu = XD.parse(cond); K = 3
p = TT.init("gate", int(ck.split(":")[1])) if ck.startswith("init:") else GL.load(ck)
Tg0 = jnp.zeros((256, 6, 3), jnp.int32)
gfn = jax.jit(lambda p, X, Y: jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(jax.grad(TT.loss)(p, X, Y, Tg0, "gate"))]))
G = lambda X, Y: np.asarray(gfn(p, jnp.asarray(X), jnp.asarray(Y)), dtype=np.float64)
def regate(X, rng):
    Xr = X.copy(); n, q = X.shape[:2]
    for s in range(a - 1): Xr[:, :, s] = rng.integers(0, K, (n, q))
    part = Xr[:, :, :a - 1].sum(-1) % K; g = rng.random((n, q)) < 0.5; op = (-part) % K; cl = (op + rng.integers(1, K, (n, q))) % K
    Xr[:, :, a - 1] = np.where(g, op, cl); return Xr
def U(Ds):
    Ds = np.asarray(Ds); S = Ds.sum(0); sq = (Ds ** 2).sum(1); n = len(Ds)
    u = (S @ S - sq.sum()) / (n * (n - 1))
    jk = np.array([((S - Ds[i]) @ (S - Ds[i]) - (sq.sum() - sq[i])) / ((n - 1) * (n - 2)) for i in range(n)])
    return float(u), float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
rng = np.random.default_rng(1_000_003 + SEED); t0 = time.time(); Dg, Dn, Dl, Gt = [], [], [], []
for i in range(B):
    d = XT.sample(256, rng, a, b1, b0, nu); X, Y = d["X"], d["Y"]
    gt = G(X, Y); gr = G(regate(X, rng), Y); gr2 = G(regate(X, rng), Y); gl = G(X, rng.permutation(Y.ravel()).reshape(Y.shape))
    Dg.append(gt - gr); Dn.append(gr - gr2); Dl.append(gt - gl); Gt.append(gt)
ug, sg = U(Dg); un, sn = U(Dn); ul, sl = U(Dl); ut, st = U(Gt)
v = np.random.default_rng(7).standard_normal(len(Dn[0])); v *= np.sqrt(max(10 * sn, 1e-30)) / np.linalg.norm(v)
vv = float(v @ v); up, sp = U([x + v for x in Dn])
res = dict(cond=cond, ckpt=ck, B=B, seed=SEED, seconds=time.time() - t0,
           gate=dict(U=ug, se=sg, z=ug / sg if sg > 0 else None, norm=float(np.sqrt(max(ug, 0)))),
           null=dict(U=un, se=sn, z=un / sn if sn > 0 else None),
           planted=dict(true_sq_norm=vv, U=up, se=sp, recovered_ratio=up / vv if vv > 0 else None),
           label=dict(U=ul, se=sl, z=ul / sl if sl > 0 else None),
           total=dict(U=ut, se=st, norm=float(np.sqrt(max(ut, 0)))))
json.dump(res, open(outp, "w"), indent=1); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv_, 6) if isinstance(vv_, float) else vv_) for kk, vv_ in v.items()}) for k, v in res.items()}))
