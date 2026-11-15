"""Reviewer check: identifiability of rho in the power-mean family and competing 1-parameter extensions of MIX4.
Same data stream/fit-test split as fit_powermean.py addendum (FP_STREAM=79_600_000)."""
import os, sys, json, time
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax
from scipy.optimize import minimize
from scipy.special import logsumexp as lse
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore"); t0 = time.time()
rng = np.random.default_rng(79_600_000); N_ = 6000
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
W, Lg, pf, e, S = RP.parts(X, q); f = (S - e); lW = np.log(W); lL = np.log(Lg); tr, te = ii[:3000], ii[3000:]
cenl = lambda lp: lp - lp.mean(-1, keepdims=True)
POOL = cenl(lL.sum(1))  # unweighted product of all groups' candidate likelihoods (what a layer-0 readout of all q values could give)
def weights(th, idx):
    al, ka, de = th[:3]; z = al * lW[idx] + ka * np.log(np.maximum(f[idx], 1)) + de * (f[idx] == 0); return z - lse(z, 1, keepdims=True)
def fam(name, th, idx):
    lw = weights(th, idx); tau = th[3]
    if name == "MIX4": return cenl(tau * lse(lw[..., None] + lL[idx], 1))
    if name == "PM":
        rho = float(np.clip(th[4], 1e-3, 5)); return cenl(tau * lse(lw[..., None] + rho * lL[idx], 1) / rho)
    if name == "PMfix":   # profile: rho fixed externally
        return cenl(tau * lse(lw[..., None] + RHO * lL[idx], 1) / RHO)
    if name == "FLOOR":   # mixture of lapse-floored likelihoods  L' = (1-eps) L + eps * mean(L)
        eps = 1 / (1 + np.exp(-th[4])); Lf = np.log((1 - eps) * Lg[idx] + eps * Lg[idx].mean(-1, keepdims=True)); return cenl(tau * lse(lw[..., None] + Lf, 1))
    if name == "PMFLOOR":
        rho = float(np.clip(th[4], 1e-3, 5)); eps = 1 / (1 + np.exp(-th[5])); Lf = np.log((1 - eps) * Lg[idx] + eps * Lg[idx].mean(-1, keepdims=True)); return cenl(tau * lse(lw[..., None] + rho * Lf, 1) / rho)
    if name == "BLEND":   # log-linear blend of the mixture with unweighted pooling of all groups (two-pathway account)
        return cenl(tau * lse(lw[..., None] + lL[idx], 1) + th[4] * POOL[idx])
    if name == "SAT":     # output saturation: soft clip of the centred mixture log-probs
        c = np.exp(th[4]); m = cenl(tau * lse(lw[..., None] + lL[idx], 1)); return c * np.tanh(m / c)
    if name == "WPOW":    # responsibility-softened mixture: weights tempered AFTER multiplying by evidence (posterior over g tempered)
        g = float(th[4]); lr = lw[..., None] + lL[idx]; lp = lse(lr, 1); post = lr - lp[:, None]   # log p(g|y,x)
        return cenl(tau * lp)  # placeholder (not used)
def r2(y, yh): return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def fit(name, y, x0s):
    loss = lambda th: float(((fam(name, th, tr) - y[tr]) ** 2).mean()); best = None
    for x0 in x0s:
        r = minimize(loss, x0, method="Powell", options=dict(maxiter=4000, xtol=1e-4, ftol=1e-9))
        if best is None or r.fun < best.fun: best = r
    return dict(params=[float(v) for v in best.x], R2_test=r2(y[te], fam(name, best.x, te)), R2_train=r2(y[tr], fam(name, best.x, tr)))
specs = sys.argv[1:]
res = {}
for sp in specs:
    if sp == "BAYES": y = cenl(np.log(GL.posterior_closed(RP.ENV1, X, q) + 1e-12))
    else:
        parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
        L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; y = cenl(np.log(RP.sm(L) + 1e-12))
    out = {}
    out["MIX4"] = fit("MIX4", y, [[1, 1, -5, 1], [0.8, 0.5, -8, 1]])
    out["PM"] = fit("PM", y, [[1, 1, -5, 1, 1.0], [0.8, 0.5, -8, 1, 0.5]])
    out["FLOOR"] = fit("FLOOR", y, [out["MIX4"]["params"] + [-3.0], out["MIX4"]["params"] + [-1.0]])
    out["BLEND"] = fit("BLEND", y, [out["MIX4"]["params"] + [0.0], out["MIX4"]["params"] + [0.2]])
    out["SAT"] = fit("SAT", y, [out["MIX4"]["params"] + [2.0], out["MIX4"]["params"] + [1.0]])
    out["PMFLOOR"] = fit("PMFLOOR", y, [out["PM"]["params"] + [-3.0], out["FLOOR"]["params"][:4] + [1.0, out["FLOOR"]["params"][4]]])
    prof = {}
    global RHO
    for RHO in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r = fit("PMfix", y, [out["PM"]["params"][:4]]); prof[RHO] = dict(R2_test=r["R2_test"], R2_train=r["R2_train"], alpha=r["params"][0], kappa=r["params"][1], tau=r["params"][3])
    out["profile_rho"] = prof
    # where does PM gain? residual reduction split by responsibility entropy under the fitted MIX4 weights
    lw = weights(out["MIX4"]["params"], te); lr = lw[..., None] + lL[te]; post = np.exp(lr - lse(lr, 1)[:, None]) .mean(-1)   # avg over y of p(g|y)
    w = np.exp(lw); H = -(w * np.log(w + 1e-30)).sum(1)
    e_mix = ((fam("MIX4", out["MIX4"]["params"], te) - y[te]) ** 2).sum(1); e_pm = ((fam("PM", out["PM"]["params"], te) - y[te]) ** 2).sum(1)
    qs = np.quantile(H, [0, 0.5, 0.8, 1.0])
    out["gain_by_weight_entropy"] = {f"H[{qs[k]:.2f},{qs[k+1]:.2f}]": dict(n=int(((H >= qs[k]) & (H <= qs[k + 1])).sum()), mse_mix=float(e_mix[(H >= qs[k]) & (H <= qs[k + 1])].mean()), mse_pm=float(e_pm[(H >= qs[k]) & (H <= qs[k + 1])].mean())) for k in range(3)}
    res[sp] = out
    print(sp[:22], {k: round(v["R2_test"], 5) for k, v in out.items() if isinstance(v, dict) and "R2_test" in v}, "rho", round(out["PM"]["params"][4], 3), "blend b", round(out["BLEND"]["params"][4], 3), "floor eps", round(1/(1+np.exp(-out["FLOOR"]["params"][4])),4), "sat c", round(np.exp(out["SAT"]["params"][4]),3), "pmfloor rho/eps", round(out["PMFLOOR"]["params"][4],3), round(1/(1+np.exp(-out["PMFLOOR"]["params"][5])),4), round(time.time() - t0), "s", flush=True)
    print("   profile", {k: (round(v["R2_test"], 5), round(v["alpha"], 2), round(v["tau"], 2)) for k, v in prof.items()}, flush=True)
    print("   gain by H", out["gain_by_weight_entropy"], flush=True)
json.dump(res, open("/tmp/review13b_mech/pm_alt_" + "_".join(s.split(":")[0] for s in specs) + ".json", "w"), indent=1)
