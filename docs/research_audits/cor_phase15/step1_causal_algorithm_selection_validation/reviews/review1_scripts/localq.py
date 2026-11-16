"""Reviewer re-derivation of the LOCALQ gap: exact posterior given ONLY each group's self-agreement bits at every question and all six values at q*,
versus the full exact posterior (xii_bayes.posterior_shared).  Independent implementation (factorisation over questions)."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import genlib as GL, xii_bayes as XB
K, a, be = 3, 0.8, 0.6; E2 = GL.Env(3, 2, beta=be)
p_ag_rel = a*a + (1-a)**2/(K-1)
def local_post(X, q):
    n = len(X); i = np.arange(n); A = X.reshape(n, 6, 3, 2); ag = A[..., 0] == A[..., 1]          # (n,6,3)
    rel, _, _ = GL._group_lik(E2, X); out = []
    for g, hs in E2.roles:
        h1, h2 = hs
        pr = np.where(ag[:, :, g], p_ag_rel, 1 - p_ag_rel)
        b1, b2 = ag[:, :, h1], ag[:, :, h2]
        pc = np.where(b1 & b2, be + (1-be)/K**2, np.where(b1 ^ b2, (1-be)*(1/K)*(1-1/K), (1-be)*(1-1/K)**2))
        lq = np.log(pr) + np.log(pc); mask = np.ones_like(lq, bool); mask[i, q] = False; tot = np.where(mask, lq, 0).sum(1)
        srcs = A[i, q][:, list(hs), :].reshape(n, 4); al = (srcs == srcs[:, :1]).all(-1); colq = be*al/K + (1-be)/K**4
        out.append(tot[:, None] + np.log(rel[i, q, g, :]) + np.log(colq)[:, None])
    L = np.stack(out); m = L.max((0, 2), keepdims=True); P = np.exp(L - m).sum(0); return P / P.sum(-1, keepdims=True)
accs = []
for rep in range(5):
    r = np.random.default_rng(100 + rep); d = GL.sample(E2, 20000, r); X, Y = d["X"], d["Y"]; q = r.integers(0, 6, 20000); i = np.arange(20000)
    Pf = XB.posterior_shared(E2, X, q); Pl = local_post(X, q)
    accs.append(((Pf.argmax(-1) == Y[i, q]).mean(), (Pl.argmax(-1) == Y[i, q]).mean()))
accs = np.array(accs); gap = accs[:, 0] - accs[:, 1]
print("full acc %.4f  local acc %.4f  gap %.4f +- %.4f (SE over 5x20k)" % (accs[:, 0].mean(), accs[:, 1].mean(), gap.mean(), gap.std(ddof=1)/np.sqrt(5)))
# self-consistency: local posterior must equal the m-marginalised full posterior; check E[Pf | local stats] ~ Pl via calibration of Pl
