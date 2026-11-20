"""DEVELOPMENT ONLY (Investigation XX, Track B design gate).  Is there an on-support functional test that separates
'different decision rule' from 'same rule, different precision'?

Candidate decision rules for T4 (G=3 groups of 2 sources, one reliable alpha=0.8, one colluding (always agree), one noise; K=3; Q=6):
  bayes    exact posterior (genlib.posterior_closed): p(y) ∝ Σ_g W_g m_g L_g(y)
  local    role-blind local Bayes: the exact posterior given ONLY question q's six reports (cross-question evidence discarded)
  noelig   drops the colluder-eligibility gate m_g:  p(y) ∝ Σ_g W_g L_g(y)
  nocount  drops the cross-question reliability weight W_g:  p(y) ∝ Σ_g m_g L_g(y)
  maprole  plug-in: answer from the single most probable reliable group (argmax_g W_g m_g)
For each model (output distribution on natural episodes) fit, by minimising KL(model || Q):
  F0  precision-only family:  Q = (1-eps) softmax(tau log P_bayes) + eps/K
  Fk  rule-mixture family:    Q = (1-eps) softmax(tau log((1-w) P_bayes + w P_k)) + eps/K
and report the KL reduction of Fk over F0 and the fitted w.  A precision-only null is built by simulating 'Bayes + i.i.d. Gaussian logit
noise' with sigma matched to each model's KL and fitting the same families.  Usage: trackb_dev_rules.py OUT.json RUN_DIR@PARENT ..."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); LEG = os.path.join(H, "..", "cor_research", "legacy", "cor_phase15")
sys.path.insert(0, os.path.join(LEG, "step1_pathway_selection_causal_identification", "src"))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import xviiim as XM
GL, RP, F = XM.GL, XM.RP, XM.F
ENV = RP.ENV1; ENVQ1 = GL.Env(3, 1, Q=1)
def rules(X, q):
    n = len(X); i = np.arange(n); PB = GL.posterior_closed(ENV, X, q)
    PL = GL.posterior_closed(ENVQ1, X[i, q][:, None], np.zeros(n, int))
    rel, col, noi = GL._group_lik(ENV, X); relm = rel.mean(-1); m, e = GL.eligibility(X, ENV.G, q)
    lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi
    nrm = lambda P: P / P.sum(-1, keepdims=True)
    PN = nrm((W[..., None] * Lg).sum(1)); PC = nrm((m[..., None] * Lg).sum(1) + 1e-12)
    g = (W * m).argmax(1); PM = nrm(Lg[i, g] + 1e-12)
    return dict(bayes=PB, local=PL, noelig=PN, nocount=PC, maprole=PM)
def fit(Pm, PB, Pk=None):
    lB = np.log(PB + 1e-12)
    def Q(th):
        tau, eps = np.exp(th[0]), 1 / (1 + np.exp(-th[1]))
        base = lB if Pk is None else np.log((1 - 1 / (1 + np.exp(-th[2]))) * PB + 1 / (1 + np.exp(-th[2])) * Pk + 1e-12)
        z = tau * base; z = z - z.max(-1, keepdims=True); S = np.exp(z); S /= S.sum(-1, keepdims=True); return (1 - eps) * S + eps / 3
    obj = lambda th: float((Pm * (np.log(Pm + 1e-12) - np.log(Q(th) + 1e-12))).sum(-1).mean())
    best = None
    for w0 in ([-4.0] if Pk is None else [-4.0, -1.0, 1.0]):
        r = minimize(obj, np.array([0.0, -5.0] + ([] if Pk is None else [w0])), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    th = best.x; return dict(KL=best.fun, tau=float(np.exp(th[0])), eps=float(1 / (1 + np.exp(-th[1]))), w=(None if Pk is None else float(1 / (1 + np.exp(-th[2])))))
def analyse(Pm, R):
    out = dict(KL_raw=float((Pm * (np.log(Pm + 1e-12) - np.log(R["bayes"] + 1e-12))).sum(-1).mean()), F0=fit(Pm, R["bayes"]))
    for k in ("local", "noelig", "nocount", "maprole"): f = fit(Pm, R["bayes"], R[k]); f["dKL"] = out["F0"]["KL"] - f["KL"]; out[k] = f
    return out
if __name__ == "__main__":
    out = sys.argv[1]; rng = np.random.default_rng(20_000_001); n = 12000
    X = GL.sample(ENV, n, rng)["X"]; q = rng.integers(0, 6, n); R = rules(X, q)
    dis = {k: float((R[k].argmax(-1) != R["bayes"].argmax(-1)).mean()) for k in R if k != "bayes"}
    res = dict(n=n, disagreement_with_bayes=dis, models={}, null={}); print("argmax disagreement with Bayes:", dis, flush=True)
    for spec in sys.argv[2:]:
        d, parent, *st = spec.split("@"); st = int(st[0]) if st else 1211; M = XM.load_dir(d, parent, st); f = jax.jit(lambda X_: F.forward(M, X_))
        L = np.asarray(F.batched(f, X))[np.arange(n), q]; Pm = RP.sm(L); a = analyse(Pm, R); res["models"][os.path.basename(d) + f"@{st}"] = a
        print(os.path.basename(d)[-30:] + f"@{st}", "KL %.4f" % a["KL_raw"], "F0 %.4f" % a["F0"]["KL"], {k: (round(a[k]["dKL"], 4), round(a[k]["w"], 3)) for k in ("local", "noelig", "nocount", "maprole")}, flush=True)
        json.dump(res, open(out, "w"), indent=1)
    lB = np.log(R["bayes"] + 1e-12)
    if os.environ.get("NO_NULL"): sys.exit(0)
    for sig in (0.25, 0.4, 0.6):
        z = lB + sig * np.random.default_rng(7).standard_normal(lB.shape); Pm = np.exp(z - z.max(-1, keepdims=True)); Pm /= Pm.sum(-1, keepdims=True)
        a = analyse(Pm, R); res["null"][f"logit_noise_{sig}"] = a
        print("NULL sigma", sig, "KL %.4f" % a["KL_raw"], "F0 %.4f" % a["F0"]["KL"], {k: (round(a[k]["dKL"], 4), round(a[k]["w"], 3)) for k in ("local", "noelig", "nocount", "maprole")}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
