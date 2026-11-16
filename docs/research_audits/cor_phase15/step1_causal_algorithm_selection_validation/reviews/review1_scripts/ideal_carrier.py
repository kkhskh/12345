"""Reviewer check: what do SUFF / SRCF / REVERSE / RESTORE slopes look like for an IDEAL carrier
(model == exact Bayes; the patched subspace carries m exactly and nothing else)?  Pure numpy."""
import sys, itertools, numpy as np
sys.path.insert(0, sys.argv[1])
import genlib as GL, xii_bayes as XB
E2 = GL.Env(3, 2, beta=0.6); K = 3
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def fid(o, p):
    o, p = o.ravel(), p.ravel(); b = (o*p).sum()/(p*p).sum(); ss = ((o-o.mean())**2).sum(); return b, 1-((o-b*p)**2).sum()/ss
PAIRS = [(0, 1), (0, 2), (1, 2)]
def mstat(X):
    a = X.reshape(len(X), 6, 3, 2); out = np.zeros((len(X), 6, 3))
    for j, (g, h) in enumerate(PAIRS):
        eq = (a[:, :, g, 0] == a[:, :, g, 1]) & (a[:, :, h, 0] == a[:, :, h, 1]) & (a[:, :, g, 0] == a[:, :, h, 0])
        out[:, :, j] = eq.sum(1, keepdims=True) - eq
    return out
Hm = {0: 2, 1: 1, 2: 0}   # index in PAIRS of H(-g): g=0 -> (1,2)=idx2, g=1 -> (0,2)=idx1, g=2 -> (0,1)=idx0
cA = np.log(1 + 0.6 * K**3 / 0.4)
def logL(X, q):
    """per-role log numerator L[g, n, y] exactly as xii_bayes.posterior_shared"""
    n = len(X); idx = np.arange(n); rel, _, noi = GL._group_lik(E2, X); relm = rel.mean(-1); a_ = X.reshape(*X.shape[:2], 3, 2); out = []
    for g, hs in E2.roles:
        srcs = a_[:, :, list(hs), :].reshape(n, 6, -1); al = (srcs == srcs[..., :1]).all(-1); colH = 0.6 * al / K + 0.4 / K**4
        lq = np.log(relm[:, :, g]) + np.log(colH); mask = np.ones_like(lq, bool); mask[idx, q] = False; tot = np.where(mask, lq, 0).sum(1)
        out.append(tot[:, None] + np.log(rel[idx, q, g, :]) + np.log(colH[idx, q])[:, None])
    return np.stack(out), [g for g, _ in E2.roles]
def post_with_m(X, q, mnew):
    L, gs = logL(X, q); mold = mstat(X)[np.arange(len(X)), q]
    L = L + np.stack([cA * (mnew[:, Hm[g]] - mold[:, Hm[g]]) for g in gs])[:, :, None]
    mx = L.max((0, 2), keepdims=True); P = np.exp(L - mx).sum(0); return P / P.sum(-1, keepdims=True)
r = np.random.default_rng(1); n = 6000; d = GL.sample(E2, n, r); X = d["X"]; q = r.integers(0, 6, n); h = r.integers(0, 3, n); X2 = X.copy()
perms = [np.array(p) for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
for j in range(n):
    pm = perms[r.integers(5)]; cols = [2*h[j], 2*h[j]+1]; oth = [t for t in range(6) if t != q[j]]; X2[j][np.ix_(oth, cols)] = pm[X[j][np.ix_(oth, cols)]]
ii = np.arange(n); ma, mb = mstat(X)[ii, q], mstat(X2)[ii, q]; dm = np.abs(mb - ma).sum(-1) > 0
X, X2, q, ma, mb = X[dm][:2400], X2[dm][:2400], q[dm][:2400], ma[dm][:2400], mb[dm][:2400]; n = len(q)
Ba, Bb = XB.posterior_shared(E2, X, q), XB.posterior_shared(E2, X2, q); dB = cen(Bb) - cen(Ba)
# sanity: Proposition S -> replacing m of Xa by m of Xb reproduces B(Xb)
print("PropS check max|B(Xa|m_b)-B(Xb)| =", np.abs(post_with_m(X, q, mb) - Bb).max())
ru = np.random.default_rng(2); un = np.array([ru.choice(np.where(q == q[j])[0][np.where(q == q[j])[0] != j]) for j in range(n)])
nat = GL.sample(E2, 20000, np.random.default_rng(3)); qn = np.random.default_rng(4).integers(0, 6, 20000); mbar = mstat(nat["X"])[np.arange(20000), qn].mean(0)
print("mean m (natural) =", mbar)
print("SUFF ideal      slope/R2 = %.3f / %.3f" % fid(cen(post_with_m(X, q, mb)) - cen(Ba), dB))
print("SRCF ideal (unrelated donor m) slope/R2 = %.3f / %.3f" % fid(cen(post_with_m(X, q, ma[un])) - cen(Ba), dB))
print("SRCF ideal mean|d| / mean|dB| = %.3f / %.3f" % (np.abs(cen(post_with_m(X, q, ma[un])) - cen(Ba)).mean(), np.abs(dB).mean()))
print("REVERSE ideal   slope = %.3f" % fid(cen(post_with_m(X2, q, ma)) - cen(Ba), dB)[0])
Mbar = np.repeat(mbar[None], n, 0)
print("RESTORE ideal (base meanrep, donor intact) slope/R2 = %.3f / %.3f" % fid(cen(Bb) - cen(post_with_m(X, q, Mbar)), dB))
print("MEANREP both ideal remaining slope = %.3f" % fid(cen(post_with_m(X2, q, Mbar)) - cen(post_with_m(X, q, Mbar)), dB)[0])
# ---- P4b: KL damage on 'unrelated' natural episodes for an IDEAL carrier under mean-replacement of m
Xn, qn2 = nat["X"][:5000], qn[:5000]; Bn = XB.posterior_shared(E2, Xn, qn2); Bm_ = post_with_m(Xn, qn2, np.repeat(mbar[None], 5000, 0))
print("ideal MEANREP KL(B || B|m=mbar) on natural episodes = %.3f" % (Bn * (np.log(Bn + 1e-12) - np.log(Bm_ + 1e-12))).sum(-1).mean())
Bz = post_with_m(Xn, qn2, np.zeros((5000, 3)))
print("ideal ZERO-m KL(B || B|m=0) on natural episodes = %.3f" % (Bn * (np.log(Bn + 1e-12) - np.log(Bz + 1e-12))).sum(-1).mean())
