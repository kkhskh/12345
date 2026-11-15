"""POST-FREEZE, EXPLORATORY (XIII-B; added after math review 1 showed that fit_family.py's POE family is handicapped -- unnormalised exponents).
Nested POWER-MEAN combination family (review 1's construction):
    log p(y) = (tau / rho) * log sum_g w_g exp(rho * log L_g(y)) + const,   w = softmax_g(alpha log W_g + kappa log max(f_g, 1) + delta [f_g = 0])
  rho = 1 : posterior MIXTURE (sum of likelihoods);  rho -> 0 : NORMALISED log-linear pooling (weighted product of experts).
Also fitted: MIX4 (rho fixed at 1) and NPOE (rho fixed at 1e-3), all with the same weight parametrisation (fair comparison: PM has one more parameter).
Natural episodes, stream FP_STREAM (default 75_300_000 = fit_family's dev stream); 3000 fit / 3000 held-out.  -> logs/fit_powermean<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax
from scipy.optimize import minimize
from scipy.special import logsumexp as lse
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore"); import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(int(os.environ.get("FP_STREAM", 75_300_000))); N_ = 6000
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
W, Lg, pf, e, S = RP.parts(X, q); f = (S - e); lW = np.log(W); lL = np.log(Lg); tr, te = ii[:3000], ii[3000:]
cenl = lambda lp: lp - lp.mean(-1, keepdims=True)
def fam(th, idx, rho_fixed=None):
    al, ka, de, tau = th[:4]; rho = rho_fixed if rho_fixed is not None else float(np.clip(th[4], 1e-3, 5))
    z = al * lW[idx] + ka * np.log(np.maximum(f[idx], 1)) + de * (f[idx] == 0); lw = z - lse(z, 1, keepdims=True)
    return cenl(tau * lse(lw[..., None] + rho * lL[idx], 1) / rho)
def r2(y, yh): return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def fit(y):
    out = {}
    for nm, rf, x0 in (("PM", None, [1, 1, -5, 1, 1.0]), ("MIX4", 1.0, [1, 1, -5, 1]), ("NPOE", 1e-3, [1, 1, -5, 1])):
        loss = lambda th: float(((fam(th, tr, rf) - y[tr]) ** 2).mean()); best = None
        for st in (x0, [0.8, 0.5, -8, 1] + ([0.5] if rf is None else [])):
            r = minimize(loss, st, method="Powell", options=dict(maxiter=6000, xtol=1e-4, ftol=1e-8))
            if best is None or r.fun < best.fun: best = r
        out[nm] = dict(params=[float(v) for v in best.x], R2_test=r2(y[te], fam(best.x, te, rf)))
        if rf is None: out[nm]["rho"] = float(np.clip(best.x[4], 1e-3, 5))
    return out
res = {}; PB = GL.posterior_closed(RP.ENV1, X, q); res["BAYES"] = fit(cenl(np.log(PB + 1e-12)))
print("BAYES", {k: (round(v["R2_test"], 4), round(v.get("rho", float("nan")), 2)) for k, v in res["BAYES"].items()}, flush=True)
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; y = cenl(np.log(RP.sm(L) + 1e-12)); res[sp] = fit(y)
    print(sp[:24], {k: (round(v["R2_test"], 4), round(v.get("rho", float("nan")), 2)) for k, v in res[sp].items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "fit_powermean" + os.environ.get("FP_TAG", "") + ".json"), "w"), indent=1)
