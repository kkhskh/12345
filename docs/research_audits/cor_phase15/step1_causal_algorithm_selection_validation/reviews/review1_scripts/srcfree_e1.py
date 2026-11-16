"""Reviewer check (E1/P6): own_share of the SRCFREE control for an IDEAL carrier (C=1, beta=1 generator, model == exact Bayes; the patched
subspace carries ALL relational information, i.e. the agreement counts n_g at the other questions, and nothing else).
DAS ideal = donor's counts transplanted into base (own_share 1 by construction); SRCFREE ideal = an unrelated same-q episode's counts."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import genlib as GL
env = GL.Env(3, 1); K, a = 3, 0.8; noi = 1 / K**2
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
r_ag = (a*a + (K-1)*((1-a)/(K-1))**2) / K; r_dis = (2*a*(1-a)/(K-1) + (K-2)*((1-a)/(K-1))**2) / K
def post_from(cnt, Xq):
    """cnt (n,3) agreement counts over the 5 other questions; Xq (n,6) source values at q -> exact closed-form posterior"""
    a_ = Xq.reshape(-1, 3, 2); ag = a_[..., 0] == a_[..., 1]; y = np.arange(K)
    p1 = np.where(a_[..., 0, None] == y, a, (1-a)/(K-1)); p2 = np.where(a_[..., 1, None] == y, a, (1-a)/(K-1)); Lg = p1*p2/noi
    W = (r_ag/noi)**cnt * (r_dis/noi)**(5-cnt); e = (cnt == 5) & ag; m = e.sum(1, keepdims=True) - e
    P = (W[..., None] * m[..., None] * Lg).sum(1); return P / P.sum(-1, keepdims=True)
rng = np.random.default_rng(11); X = GL.sample(env, 60000, rng)["X"]; q = rng.integers(0, 6, len(X)); a6 = X.reshape(len(X), 6, 3, 2); eq = a6[..., 0] == a6[..., 1]
fam = {k: ([], [], []) for k in ("E5", "E3", "QF5", "QF4")}
for i in range(len(X)):
    qq = q[i]; cnt = eq[i].sum(0) - eq[i, qq]; agq = eq[i, qq]; kind = ["E5", "E3", "QF5", "QF4"][rng.integers(4)]
    if len(fam[kind][2]) >= 1500: continue
    x = X[i].copy()
    if kind in ("E5", "E3"):
        lo, hi = (4, 5) if kind == "E5" else (3, 4); hs = [h for h in range(3) if agq[h] and cnt[h] in (lo, hi)]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [p for p in range(6) if p != qq]
        if cnt[h] == hi: cand = [p for p in others if eq[i, p, h]]; p = cand[rng.integers(len(cand))]; x[p, 2*h+1] = (x[p, 2*h] + rng.integers(1, 3)) % 3
        else: cand = [p for p in others if not eq[i, p, h]]; p = cand[rng.integers(len(cand))]; x[p, 2*h+1] = x[p, 2*h]
    else:
        n_ = 5 if kind == "QF5" else 4; hs = [h for h in range(3) if cnt[h] == n_]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; x[qq, 2*h+1] = (x[qq, 2*h] + rng.integers(1, 3)) % 3 if agq[h] else x[qq, 2*h]
    e2 = x.reshape(6, 3, 2)
    if not (eq[i].all(0).any() and (e2[..., 0] == e2[..., 1]).all(0).any()): continue
    fam[kind][0].append(X[i]); fam[kind][1].append(x); fam[kind][2].append(qq)
def counts(X, q): e = (X.reshape(len(X), 6, 3, 2)[..., 0] == X.reshape(len(X), 6, 3, 2)[..., 1]); return e.sum(1) - e[np.arange(len(X)), q]
for k, (Xb, Xd, qq) in fam.items():
    Xb, Xd, qq = np.array(Xb), np.array(Xd), np.array(qq); n = len(qq); i = np.arange(n)
    Xbq, Xdq = Xb.reshape(n, 6, 6)[i, qq], Xd.reshape(n, 6, 6)[i, qq]
    Bb = post_from(counts(Xb, qq), Xbq); Bd = post_from(counts(Xd, qq), Xdq)
    assert np.allclose(Bb, GL.posterior_closed(env, Xb, qq), atol=1e-10)
    own = cen(Bd) - cen(Bb)
    ru = np.random.default_rng(7); un = np.array([ru.choice(np.where(qq == qq[j])[0][np.where(qq == qq[j])[0] != j]) for j in range(n)])
    ok = np.ones(n, bool); Bs = post_from(counts(Xb, qq)[un], Xbq); ok &= np.isfinite(Bs).all(1)
    d = cen(Bs) - cen(Bb)
    share = (d[ok]*own[ok]).sum() / (own[ok]**2).sum()
    print(f"{k}: n={n}  ideal SRCFREE own_share = {share:.3f}  (finite {ok.mean():.2f}); mean|own|={np.abs(own).mean():.3f}, mean|srcfree d|={np.abs(d[ok]).mean():.3f}")
