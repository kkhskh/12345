"""Core of xv_math.py (design / order-k fit), imported by src/xvdiag.py."""
import itertools, os, sys
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import xvtask as T
XA = T.XALL; K = 3
def design(k):
    cols = []
    for A in itertools.chain.from_iterable(itertools.combinations(range(6), r) for r in range(1, k + 1)):
        code = sum(XA[:, s] * K ** i for i, s in enumerate(A)); M = np.zeros((len(XA), K ** len(A))); M[np.arange(len(XA)), code] = 1; cols.append(M)
    return np.concatenate(cols, 1)
def fit(k, px, PB):
    F = design(k); d = F.shape[1]
    def f(w):
        Z = F @ w.reshape(d, K); Z -= Z.max(1, keepdims=True); L = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -(px[:, None] * PB * L).sum(), (F.T @ (px[:, None] * (np.exp(L) - PB))).ravel()
    r = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-12, ftol=1e-15))
    Z = F @ r.x.reshape(d, K); P = np.exp(Z - Z.max(1, keepdims=True)); return P / P.sum(1, keepdims=True), bool(r.success)
