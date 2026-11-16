"""Reviewer check: power of SHUFD_vs_borrowed.  Ideal row model for the C=1, beta=1 generator: the mixture-head rows carry exactly the role
responsibilities r_g (relpairs.resp); output p(y) = sum_g r_g * L_g(y)/sum_y L_g(y)  (== posterior_closed).  Pure numpy."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import genlib as GL
env = GL.Env(3, 1); K = 3
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def fid(o, p):
    o, p = o.ravel(), p.ravel(); b = (o*p).sum()/(p*p).sum(); ss = ((o-o.mean())**2).sum(); return b, 1-((o-b*p)**2).sum()/ss
def parts(X, q):
    n = len(X); i = np.arange(n); rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; m, e = GL.eligibility(X, 3, q); return W, Lg, m
def resp(X, q):
    W, Lg, m = parts(X, q); r = W * m * Lg.sum(-1); return r / np.maximum(r.sum(1, keepdims=True), 1e-300), Lg / Lg.sum(-1, keepdims=True)
def out(r, ell): P = (r[..., None] * ell).sum(1); return P / P.sum(-1, keepdims=True)
rng = np.random.default_rng(5); N = 6000; X = GL.sample(env, N, rng)["X"]; q = rng.integers(0, 6, N); Xd = X.copy(); h = rng.integers(0, 3, N); qp = (q + rng.integers(1, 6, N)) % 6
for i in range(N):
    a0, a1 = X[i, qp[i], 2*h[i]], X[i, qp[i], 2*h[i]+1]; Xd[i, qp[i], 2*h[i]+1] = a0 if a0 != a1 else (a0 + rng.integers(1, 3)) % 3
PB, PBd = GL.posterior_closed(env, X, q), GL.posterior_closed(env, Xd, q)
keep = np.where(np.abs(cen(PBd) - cen(PB)).max(-1) > 0.05)[0][:800]; X, Xd, q = X[keep], Xd[keep], q[keep]; n = len(q); i = np.arange(n)
PB, PBd = PB[keep], PBd[keep]; dB = cen(PBd) - cen(PB)
rb, ell = resp(X, q); rd, _ = resp(Xd, q)
print("ideal row model reproduces posterior: max diff", np.abs(out(rb, ell) - PB).max())
rs = np.random.default_rng(6); perm = np.array([rs.choice(np.where((q == q[j]) & (i != j))[0]) for j in range(n)])
print("TRUE  R2 vs dB:", "%.3f/%.3f" % fid(cen(out(rd, ell)) - cen(PB), dB))
rn = np.clip(rb + (rd - rb)[perm], 0, None); rn = rn / np.maximum(rn.sum(1, keepdims=True), 1e-300)
dS = cen(out(rn, ell)) - cen(PB)
print("SHUFD (ideal carrier) slope/R2 vs base dB    : %.3f / %.3f" % fid(dS, dB))
print("SHUFD (ideal carrier) slope/R2 vs borrowed dB: %.3f / %.3f" % fid(dS, dB[perm]))
# what the borrowed displacement WOULD predict in the base context = the 'transferred-role-shift' target
print("mean|dS| = %.3f, mean|dB| = %.3f" % (np.abs(dS).mean(), np.abs(dB).mean()))
