"""Workstream A2/A4 (XIII-B): WHICH OPERATION combines per-group evidence?  Fit two parametric algorithm families to each model's output
(centred log-probabilities at q on natural episodes, stream 75_300_000) and compare.
  MIX (mixture of experts / posterior mixture; the exact Bayes form at alpha = kappa = tau = 1):
      p(y) ∝ sum_g w_g L_g(y)^tau,   log w_g = alpha log W_g + kappa log max(f_g, eps),   f_g = S - e_g   (eps = 1e-3)
  POE (product of experts / log-linear pooling of per-group evidence):
      log p(y) = sum_g c_g log L_g(y) + const,  c_g = softplus(a log W_g + b log max(f_g, eps) + c0)
  MIXF (mixture with FREE per-cell eligibility weights instead of the Bayes count factor; tests the exclusion law):
      log w_g = alpha log W_g + k[S, e_g]   (5 free cell offsets)
Goodness: R^2 of centred log-probabilities, KL(model || family fit).  Bayes itself is fitted as a sanity check (MIX must give R^2 = 1).
Held-out: fit on 3000 episodes, evaluate on 3000 others.  -> logs/fit_family<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("FF_STREAM", 75_300_000))); N_ = 6000
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
W, Lg, pf, e, S = RP.parts(X, q); f = S - e; lW = np.log(W); lL = np.log(Lg); lf = np.log(np.maximum(f, 1e-3))
cell = (np.clip(S, 0, 3).astype(int) * 2 + e.astype(int))      # (N,G) cell id of (S, e_g): values in {0..7}
cells = sorted(set(cell.ravel().tolist())); cid = np.searchsorted(cells, cell)
tr, te = ii[:3000], ii[3000:]
def cenl(lp): return lp - lp.mean(-1, keepdims=True)
def mix_lp(th, idx, free=False):
    al, tau = th[0], th[1]
    lw = al * lW[idx] + (np.asarray(th[2:])[cid[idx]] if free else th[2] * lf[idx])
    z = lw[..., None] + tau * lL[idx]; m = z.max(1, keepdims=True); lp = np.log(np.exp(z - m).sum(1)) + m[:, 0]; return cenl(lp)
def poe_lp(th, idx):
    a, b, c0 = th; cg = np.logaddexp(0, a * lW[idx] + b * lf[idx] + c0); return cenl((cg[..., None] * lL[idx]).sum(1))
def r2(y, yh): return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def kl(y, yh):
    P = RP.sm(y); Q = RP.sm(yh); return float((P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1).mean())
def fit(y):
    out = {}
    for nm, fn, x0 in (("MIX", lambda th, idx: mix_lp(th, idx), [1.0, 1.0, 1.0]), ("POE", poe_lp, [0.1, 0.5, 0.0]),
                       ("MIXF", lambda th, idx: mix_lp(th, idx, True), [1.0, 1.0] + [0.0] * len(cells))):
        loss = lambda th: float(((fn(th, tr) - y[tr]) ** 2).mean())
        best = None
        for start in (x0, list(np.array(x0) * 0.5 + 0.1)):
            r = minimize(loss, start, method="Nelder-Mead" if len(x0) <= 3 else "Powell", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
            if best is None or r.fun < best.fun: best = r
        yh = fn(best.x, te); out[nm] = dict(params=[float(v) for v in best.x], R2_test=r2(y[te], yh), KL_test=kl(y[te], yh))
    return out
res = {}
PB = GL.posterior_closed(GL.Env(3, 1), X, q); res["BAYES"] = fit(cenl(np.log(PB + 1e-12)))
print("BAYES", {k: (round(v["R2_test"], 4), [round(p, 2) for p in v["params"][:3]]) for k, v in res["BAYES"].items()}, flush=True)
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; y = cenl(np.log(RP.sm(L) + 1e-12))
    r = fit(y); r["KL_bayes_model"] = float((PB * (np.log(PB + 1e-12) - np.log(RP.sm(L) + 1e-12))).sum(-1).mean()); res[sp] = r
    print(sp[:6], {k: (round(v["R2_test"], 4), round(v["KL_test"], 4), [round(p, 2) for p in v["params"][:3]]) for k, v in r.items() if isinstance(v, dict)}, flush=True)
res["cells"] = cells
json.dump(res, open(os.path.join(HERE, "..", "logs", "fit_family" + os.environ.get("FF_TAG", "") + ".json"), "w"), indent=1)
