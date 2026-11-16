"""XIV [num]: order-1 (source-wise) ceiling of the native T4 decision, and the agreement-only (order-2 within own question) rule.
Order-1 class: multinomial logistic on one-hot reports of all 36 sources for target question q (relational invisibility predicts that
other-question sources add nothing at order 1).  Fitted on 120,000 sampled (episode, q) with exact posterior targets (soft labels),
evaluated on 30,000 fresh ones: NLL to the posterior, Bayes-argmax agreement, KL.  Stream 97_400_000 (development of the mathematics).
Writes logs/t4_order1.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, native as NV, genlib as GL, relpairs as RP
from scipy.optimize import minimize
rng = np.random.default_rng(97_400_000)
def data(n):
    X = NV.Bm.sample_fast(NV.ENV, n, rng)["X"]; q = rng.integers(0, 6, n); P = GL.posterior_closed(RP.ENV1, X, q); return X, q, P
def feats(X, q, own_only):
    n = len(X); Xq = X[np.arange(n), q]                                   # own question sources (n, 6)
    F = [np.eye(3)[Xq[:, s]] for s in range(6)]
    if not own_only:
        for k in range(1, 6): Xo = X[np.arange(n), (q + k) % 6]; F += [np.eye(3)[Xo[:, s]] for s in range(6)]
    return np.concatenate(F, 1)
def fit(F, P):
    d = F.shape[1]
    def f(w):
        W = w[:-3].reshape(d, 3); Z = F @ W + w[-3:]; Z -= Z.max(1, keepdims=True); L = Z - np.log(np.exp(Z).sum(1, keepdims=True)); Pm = np.exp(L)
        G = Pm - P; return -(P * L).sum() / len(F), np.concatenate([(F.T @ G).ravel(), G.sum(0)]) / len(F)
    return minimize(f, np.zeros(d * 3 + 3), jac=True, method="L-BFGS-B", options=dict(maxiter=3000)).x
def ev(w, F, P):
    d = F.shape[1]; Z = F @ w[:-3].reshape(d, 3) + w[-3:]; Pm = np.exp(Z - Z.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
    return dict(NLL=float(-(P * np.log(Pm)).sum(1).mean()), agree_bayes=float((Pm.argmax(1) == P.argmax(1)).mean()), KL=float((P * (np.log(P + 1e-12) - np.log(Pm))).sum(1).mean()))
Xa, qa, Pa = data(120000); Xb, qb, Pb = data(30000); out = dict(bayes_NLL=float(-(Pb * np.log(Pb + 1e-12)).sum(1).mean()))
for name, own in (("order1_own_question", True), ("order1_all_sources", False)):
    w = fit(feats(Xa, qa, own), Pa); out[name] = ev(w, feats(Xb, qb, own), Pb); print(name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "t4_order1.json"), "w"), indent=1); print(out)
