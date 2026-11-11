"""Formal review check 3: with the T4 posterior floor, pointwise KL(p*||p_M) lower-bounds the log-odds error
(min KL at given ||e||_inf = B over directions), so AVERAGE KL does control AVERAGE log-odds error (and, via the density-ratio row,
E_Pi|Delta|).  Worst case taken over the 400 T4 posteriors with the smallest minimum coordinate."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(PKG, "src")); import cri as R; E = R.E
rng = np.random.default_rng(5); K = 3
c = lambda l: l - l.mean(-1, keepdims=True)
d = E.sample(20000, 13); X = d["X"]; pb = E.bayes_from(X, E.loo(X)).reshape(-1, K); pb = pb[np.isfinite(pb).all(1)]
ps = pb[np.argsort(pb.min(1))[:400]]
U = c(rng.normal(size=(4000, K))); U /= np.abs(U).max(1, keepdims=True)        # directions with ||u||_inf = 1
for B in (0.05, 0.2, 1, 3, 10):
    lp = np.log(ps)[:, None, :] + B * U[None]; lp -= lp.max(-1, keepdims=True); pm = np.exp(lp); pm /= pm.sum(-1, keepdims=True)
    kl = (ps[:, None, :] * (np.log(ps)[:, None, :] - np.log(pm))).sum(-1).min()
    print("B=%5.2f  min KL over worst-floor posteriors and directions = %.5f   KL/B^2 = %.4f  KL/B = %.4f" % (B, kl, kl / B ** 2, kl / B))
