"""Mathematical results for Investigation X (numpy): causal-abstraction identifiability across exact implementations of one Bayes rule.
Three exact implementations of p(y) ∝ sum_g W_g L_g(y) (S - e_g)  (T4, C=1):
  I1 'pooled'  : intermediate z1 = (S, A_y (3), B_y (3))                -> p ∝ S A - B
  I2 'per-group': intermediate z2 = (m_0, m_1, m_2, U_0(3), U_1(3), U_2(3)) with m_g = sum_{h!=g} e_h (computed pairwise, no S),
                  U_g = W_g L_g                                          -> p ∝ sum_g m_g U_g
  I3 'pairwise': intermediate z3 = (T_{g,h}(3) for ordered pairs g!=h), T_{gh} = W_g L_g e_h     -> p ∝ sum_{g!=h} T_{gh}
All three are the same function (checked).  High-level intervention 'set S := S_src, keep everything else of the base' (the Investigation X
'S' variable).  For each implementation, find the best LINEAR interchange subspace R (dim k) of its intermediate:
  z_patched = z_b + R R^T (z_s - z_b),  output from z_patched,  objective = mean |log p_patched - log p_cf|  over pairs with S_src != S_base,
optimised by Adam over R (several restarts), k = 1..3.  Also exact constructions: I1 -> R = e_S ; I2 -> R = (1,1,1)/sqrt3 in m-coordinates.
-> logs/math_abstraction.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, genlib as GL
import jax, jax.numpy as jnp, optax
np.seterr(all="ignore")
env = GL.Env(G=3, C=1); rng = np.random.default_rng(71)
d = GL.sample(env, 40000, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
rel, col, noi = GL._group_lik(env, X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
e = ((cnt == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1); U = W[..., None] * Lg; U = U / U.sum((1, 2), keepdims=True)   # scale-free
def z1(): return np.concatenate([S[:, None], U.sum(1), (U * e[..., None]).sum(1)], 1)
def out1(z): return z[:, :1] * z[:, 1:4] - z[:, 4:7]
def z2(): m = np.stack([sum(e[:, h] for h in range(3) if h != g) for g in range(3)], 1); return np.concatenate([m, U.reshape(len(U), 9)], 1)
def out2(z): return (z[:, :3, None] * z[:, 3:].reshape(-1, 3, 3)).sum(1)
pairs = [(g, h) for g in range(3) for h in range(3) if g != h]
def z3(): return np.concatenate([U[:, g] * e[:, h:h + 1] for g, h in pairs], 1)
def out3(z): return z.reshape(-1, 6, 3).sum(1)
logp = lambda P: np.log(np.maximum(P, 1e-30) / np.maximum(P, 1e-30).sum(-1, keepdims=True))
ref = logp(out1(z1()))

out = {"functional_equivalence_max_diff": dict(I2=float(np.abs(logp(out2(z2())) - ref).max()), I3=float(np.abs(logp(out3(z3())) - ref).max()))}
# pairs (base, source); two high-level interventions on the base: VS 'set S := S_src (keep base e_g, U)'; VE 'set e := e_src (keep base U)'
two, one = np.where(S >= 2)[0], np.where(S == 1)[0]; nb = 4000
b = np.concatenate([rng.choice(two, nb // 2), rng.choice(one, nb // 2)]); s = np.where(S[b] >= 2, rng.choice(one, nb), rng.choice(two, nb))
def norm(P): P = np.maximum(P, 0) + 1e-12; return P / P.sum(-1, keepdims=True)
CF = {"VS": norm(S[s][:, None] * U[b].sum(1) - (U[b] * e[b][..., None]).sum(1)),
      "VE": norm((U[b] * (S[s][:, None] - e[s])[..., None]).sum(1))}
OUTJ = {"1": lambda z: z[:, :1] * z[:, 1:4] - z[:, 4:7], "2": lambda z: (z[:, :3, None] * z[:, 3:].reshape(-1, 3, 3)).sum(1), "3": lambda z: z.reshape(-1, 6, 3).sum(1)}
def err(zp, of, cf, xp=np):
    P = xp.maximum(OUTJ[of](zp), 0) + 1e-12; P = P / P.sum(-1, keepdims=True); return xp.abs(P - cf).sum(-1).mean()
def fit(zfun, of, cfk, k, restarts=4, steps=1500):
    Z = jnp.asarray(zfun()); zb, zs = Z[b], Z[s]; D = Z.shape[1]; cf = jnp.asarray(CF[cfk]); best = None
    def lossf(R):
        Q, _ = jnp.linalg.qr(R); return err(zb + ((zs - zb) @ Q) @ Q.T, of, cf, jnp)
    g = jax.jit(jax.value_and_grad(lossf))
    for r in range(restarts):
        R = jax.random.normal(jax.random.PRNGKey(r), (D, k)); opt = optax.adam(1e-2); st = opt.init(R)
        for _ in range(steps): l, gr = g(R); up, st = opt.update(gr, st); R = optax.apply_updates(R, up)
        v = float(lossf(R)); best = v if best is None or not np.isfinite(best) else min(best, v)
    return best
def exact(zfun, of, cfk, Rm):
    Z = zfun(); zb, zs = Z[b], Z[s]; Q, _ = np.linalg.qr(Rm); return float(err(zb + ((zs - zb) @ Q) @ Q.T, of, CF[cfk]))
Pb = norm(out1(z1()[b]))
out["no_intervention_error"] = {k: float(np.abs(Pb - CF[k]).sum(-1).mean()) for k in CF}
R1 = np.zeros((7, 1)); R1[0, 0] = 1; R2 = np.zeros((12, 3)); R2[:3, :3] = np.eye(3)
out["exact_constructions"] = dict(I1_VS_S_coordinate=exact(z1, "1", "VS", R1), I2_VE_m_coordinates=exact(z2, "2", "VE", R2),
                                  I1_VE_S_coordinate=exact(z1, "1", "VE", R1), I2_VS_m_coordinates=exact(z2, "2", "VS", R2))
out["best_linear_subspace_error_L1prob"] = {}
for name, zf, of in (("I1", z1, "1"), ("I2", z2, "2"), ("I3", z3, "3")):
    for cfk in ("VS", "VE"):
        for k in (1, 2, 3, 4):
            v = fit(zf, of, cfk, k); out["best_linear_subspace_error_L1prob"][f"{name}_{cfk}_k{k}"] = v; print(name, cfk, k, round(v, 4), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "math_abstraction.json"), "w"), indent=1); print(json.dumps({k: out[k] for k in ("functional_equivalence_max_diff", "no_intervention_error", "exact_constructions")}, indent=1))
