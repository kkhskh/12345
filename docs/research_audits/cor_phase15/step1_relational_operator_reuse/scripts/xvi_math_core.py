"""Core of the XVI exact mathematics: order-k log-linear design over S source positions and the expected-NLL-optimal order-k rule.
Order-k class: logits(y | x) = sum over source subsets A with |A| <= k of a free table w_A(x_A, y).  Fitted by L-BFGS on the exact
expected NLL under (px, PB) over all K^S inputs (XV's xv_math_core.fit generalised from S = 6 to any S)."""
import itertools
import numpy as np
from scipy.optimize import minimize
K = 3
_D = {}


def design(k, XA):
    key = (k, XA.shape)
    if key in _D: return _D[key]
    S = XA.shape[1]; cols = []
    for A in itertools.chain.from_iterable(itertools.combinations(range(S), r) for r in range(1, k + 1)):
        code = sum(XA[:, s] * K ** i for i, s in enumerate(A)); M = np.zeros((len(XA), K ** len(A))); M[np.arange(len(XA)), code] = 1; cols.append(M)
    _D[key] = np.concatenate(cols, 1); return _D[key]


def fit(k, px, PB, XA, maxiter=20000):
    F = design(k, XA); d = F.shape[1]
    def f(w):
        Z = F @ w.reshape(d, K); Z -= Z.max(1, keepdims=True); L = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -(px[:, None] * PB * L).sum(), (F.T @ (px[:, None] * (np.exp(L) - PB))).ravel()
    r = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-12, ftol=1e-15))
    Z = F @ r.x.reshape(d, K); P = np.exp(Z - Z.max(1, keepdims=True)); return P / P.sum(1, keepdims=True), bool(r.success)
