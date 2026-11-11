"""EXPLORATORY (development).  Behavioural separability test.  For each model, fit to its centred logits (all 6 decisions of 6000
natural episodes plus the edited F45/F34/F0/S4/S5/O/F54o arrays at their scored decision) the evidence-update family
    p(y) ∝ [ sum_g w(n_g) v(m_g) L_g(y) ]^tau,   m_g = #{other perfect groups agreeing at q},
with log w(0) = 0, log v(1) = 0 as gauges (8 free parameters incl. tau), and the separable sub-family v ≡ 1 (groupwise; 6 params).
Exact Bayes is w = RHO^n, v(m) = m, tau = 1 (v(0) = 0).  Reports fitted log v(0), log v(2), the R^2 of each family, and the
R1/R2/R3/F54o responses predicted by the fitted members.  A groupwise solution predicts sign(F54o) = sign(w4 - w5) and
sign(R1) = sign(R3) = sign(w5 - w4) (FORMAL_THEOREMS_AND_LIMITS.md, Theorem 1)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.optimize import minimize
import r1lib as L, nnlib as NL
E, B = L.E, L.B
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_dev.npz"), allow_pickle=True))
# design: decisions (X, q) -> n, m, L
blocks = []
Xn = ev["X"]; Nn = len(Xn)
for qq in range(6): blocks.append(("base_all", Xn, np.full(Nn, qq)))
for k, (X, q) in L.eval_inputs(ev).items():
    if k != "base": blocks.append((k, X, q))
def feats(X, q):
    idx = np.arange(len(X)); n = E.loo(X)[idx, q]; agq = L.votes(X, q) >= 0; m = L.closed_form_weights(n, agq)[1]; return n, m, L.Lrel(X, q)
FE = [feats(X, q) for _, X, q in blocks]
nA = np.concatenate([f[0] for f in FE]); mA = np.concatenate([f[1] for f in FE]); LA = np.concatenate([f[2] for f in FE])
def logits(th, n=nA, m=mA, Lr=LA, sep=False):
    lw = np.r_[0, th[0:5]]; lv = np.zeros(3) if sep else np.r_[th[5], 0, th[6]]; tau = np.exp(th[-1])
    W = np.exp(np.clip(lw[n] + lv[m], -60, 60)); p = (W[:, :, None] * Lr).sum(1); lp = tau * np.log(p + 1e-300); return lp - lp.mean(-1, keepdims=True)
def fit(target, sep):
    th0 = np.r_[np.arange(1, 6) * np.log(L.RHO), ([] if sep else [-5.0, np.log(2)]), 0.0]
    f = lambda th: ((logits(th, sep=sep) - target) ** 2).mean()
    r = minimize(f, th0, method="L-BFGS-B", options=dict(maxiter=3000)); return r.x, r.fun
def predict(th, sep):
    # responses of the fitted member on the eval families, evaluated exactly like a model
    import evalrun as ER
    lg = {}
    for k, (X, q) in L.eval_inputs(ev).items():
        n, m, Lr = feats(X, q); lg[k] = logits(th, n, m, Lr, sep)
    F = L.evaluate_logits(ev, lg); return dict(R1=F["F45"]["R1"]["dmM"], R2=F["F45"]["R2"]["dmM"], R3=F["F45"]["R3"]["dmM"], F34R1=F["F34"]["R1"]["dmM"], F54o=F["F54o"]["all"]["dmM"])
out = {}
models = sys.argv[1:] or [f"xT{s}" for s in range(70, 78)] + [f"xP{s}" for s in range(80, 88)]
for mname in models:
    fn = NL.full_model_logits(NL.load(NL.dev_ckpt(mname)))
    tg = []
    for k, X, q in blocks:
        A = fn(X); c = A[np.arange(len(X)), q]; tg.append(c - c.mean(-1, keepdims=True))
    tg = np.concatenate(tg); var = ((tg - 0) ** 2).mean()
    res = {}
    for nm, sep in (("interaction", False), ("groupwise", True)):
        th, lo = fit(tg, sep); res[nm] = dict(theta=th.tolist(), mse=float(lo), r2=float(1 - lo / var), pred=predict(th, sep))
    ti = res["interaction"]["theta"]; res["log_v0"], res["log_v2_minus_log2"] = ti[5], ti[6] - np.log(2)
    res["log_w5_minus_w4_groupwise"] = res["groupwise"]["theta"][4] - res["groupwise"]["theta"][3]
    out[mname] = res
    print(mname, "R2 int %.3f gw %.3f | log v0 %+.2f (Bayes -inf)  log v2-log2 %+.2f | gw log w5/w4 %+.2f | int pred R1 %+.2f F54o %+.2f | gw pred R1 %+.2f R3 %+.2f F54o %+.2f" % (
        res["interaction"]["r2"], res["groupwise"]["r2"], res["log_v0"], res["log_v2_minus_log2"], res["log_w5_minus_w4_groupwise"],
        res["interaction"]["pred"]["R1"], res["interaction"]["pred"]["F54o"], res["groupwise"]["pred"]["R1"], res["groupwise"]["pred"]["R3"], res["groupwise"]["pred"]["F54o"]), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "separability_fit_dev.json"), "w"), indent=1)
