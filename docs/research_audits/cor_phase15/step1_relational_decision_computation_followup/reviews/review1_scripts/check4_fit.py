import os, sys, numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); os.environ["JAX_PLATFORMS"] = "cpu"
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, PKG + "/src")
from scipy.optimize import minimize
import genlib as GL, relpairs as RP, native as NV
rng = np.random.default_rng(75_300_000); N_ = 6000
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
W, Lg, pf, e, S = RP.parts(X, q); f = S - e; lW = np.log(W); lL = np.log(Lg)
lf_floor = np.log(np.maximum(f, 1e-3)); lf_ind = np.where(f > 0, np.log(np.maximum(f, 1)), -1e4)
tr, te = ii[:3000], ii[3000:]
cenl = lambda lp: lp - lp.mean(-1, keepdims=True)
def lse(z, ax): m = z.max(ax, keepdims=True); return np.log(np.exp(z - m).sum(ax, keepdims=True)) + m
def mix(th, idx, lf): al, tau, ka = th; z = (al * lW[idx] + ka * lf[idx])[..., None] + tau * lL[idx]; return cenl(lse(z, 1)[:, 0])
def poe(th, idx): a, b, c0 = th; cg = np.logaddexp(0, a * lW[idx] + b * lf_floor[idx] + c0); return cenl((cg[..., None] * lL[idx]).sum(1))
def poe_norm(th, idx):
    lam, al, ka = th; z = al * lW[idx] + ka * lf_floor[idx]; r = np.exp(z - lse(z, 1)); return cenl(lam * (r[..., None] * lL[idx]).sum(1))
def power(th, idx):
    al, ka, rho = th; rho = np.clip(rho, 1e-3, 5); z = al * lW[idx] + ka * lf_floor[idx]; lr = z - lse(z, 1)
    return cenl(lse(lr[..., None] + rho * lL[idx], 1)[:, 0] / rho)
def r2(y, yh): return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
FAMS = dict(MIX=(lambda th, idx: mix(th, idx, lf_floor), [1, 1, 1]), MIX_indicator=(lambda th, idx: mix(th, idx, lf_ind), [1, 1, 1]),
            POE=(poe, [0.1, 0.5, 0.0]), POE_normalised=(poe_norm, [1, 1, 1]), POWERMEAN=(power, [1, 1, 0.5]))
def fit(y, name):
    out = {}
    for nm, (fn, x0) in FAMS.items():
        best = None
        for st in (x0, list(np.array(x0) * 0.5 + 0.1), [2, 2, 2]):
            r = minimize(lambda th: float(((fn(th, tr) - y[tr]) ** 2).mean()), st, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-8))
            if best is None or r.fun < best.fun: best = r
        out[nm] = (round(r2(y[te], fn(best.x, te)), 4), [round(float(v), 3) for v in best.x])
    print(name, out, flush=True)
PB = GL.posterior_closed(GL.Env(3, 1), X, q); fit(cenl(np.log(PB + 1e-12)), "BAYES")
# responsibilities spread: how often is the posterior a genuine mixture?
r = RP.resp(W, Lg, f); print("frac episodes with max responsibility < 0.9:", float((r.max(1) < 0.9).mean()), " <0.99:", float((r.max(1) < 0.99).mean()))
if len(sys.argv) > 1:
    import jax, ftlib as F
    for sp in sys.argv[1:]:
        parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
        L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; fit(cenl(np.log(RP.sm(L) + 1e-12)), sp[:6])
