"""XIII-B shared helpers: relational edit pairs, Bayes parts, fidelity, and THE single mixture-head selection rule used by every XIII-B script
(review 2/3 of XIII: scripts previously selected heads differently)."""
import numpy as np
import genlib as GL, native as NV

ENV1 = GL.Env(3, 1)
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
sm = lambda L: (lambda E: E / E.sum(-1, keepdims=True))(np.exp(L - L.max(-1, keepdims=True)))


def fid(o, p):
    o, p = np.asarray(o).ravel(), np.asarray(p).ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); ss = ((o - o.mean()) ** 2).sum()
    return dict(slope=b, R2=float(1 - ((o - b * p) ** 2).sum() / ss) if ss > 1e-10 else 0.0, mean_abs_obs=float(np.abs(o).mean()), mean_abs_pred=float(np.abs(p).mean()))


def parts(X, q, env=ENV1):
    """W, L (n,G,K), pf, e, S at target question q (C=1 closed-form ingredients)."""
    n = len(X); i = np.arange(n); rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; a_ = X.reshape(n, 6, 3, 2)[i, q]
    pf = (GL.counts(X, 3)[i, q] == 5).astype(float); e = pf * (a_[..., 0] == a_[..., 1]); return W, Lg, pf, e, e.sum(1, keepdims=True)


def resp(W, Lg, f):
    r = W * np.maximum(f, 0) * Lg.sum(-1); return r / np.maximum(r.sum(1, keepdims=True), 1e-300)


def relational_pairs(n, rng, min_change=0.05, pool=4):
    """base natural episodes and donors that differ ONLY in one group's source pair at ONE question q' != q (agree <-> disagree).
    Values at q are identical, so the Bayes change is purely relational.  Returns dict(Xb, Xd, q, qp, h, e_change, PB, PBd)."""
    d = NV.Bm.sample_fast(NV.ENV, pool * n, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
    Xd = X.copy(); h = rng.integers(0, 3, len(X)); qp = (q + rng.integers(1, 6, len(X))) % 6
    for i in range(len(X)):
        a0, a1 = X[i, qp[i], 2 * h[i]], X[i, qp[i], 2 * h[i] + 1]
        Xd[i, qp[i], 2 * h[i] + 1] = a0 if a0 != a1 else (a0 + rng.integers(1, 3)) % 3
    PB = GL.posterior_closed(ENV1, X, q); PBd = GL.posterior_closed(ENV1, Xd, q)
    _, _, _, eb, _ = parts(X, q); _, _, _, ed, _ = parts(Xd, q)
    keep = np.where(np.abs(cen(PBd) - cen(PB)).max(-1) > min_change)[0][:n]
    return dict(Xb=X[keep], Xd=Xd[keep], q=q[keep], qp=qp[keep], h=h[keep], e_change=(eb[keep] != ed[keep]).any(1), PB=PB[keep], PBd=PBd[keep])


MIX_RULE = dict(mass=0.8, share=0.2)   # XIII frozen class rule (l2_scan): mass on q source tokens >= 0.8 AND direct-logit variance share >= 0.2


def head_stats(M, X, q, F, jnp):
    """Per layer-2 head: readout-token attention mass on the 6 source tokens of q, and direct-logit variance share (LN linearised as in l2_scan)."""
    rec = {}; F.forward(M, jnp.asarray(X), rec=rec); A = np.asarray(rec["pattern2"]); n = len(X); i = np.arange(n); qs = q[:, None] * 6 + np.arange(6)[None]
    mass = np.array([np.take_along_axis(A[i, h, 36 + q], qs, 1).sum(1).mean() for h in range(4)])
    x3 = np.asarray(rec["resid3"])[i, 36 + q]; mu = x3.mean(-1, keepdims=True); sd = np.sqrt(((x3 - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
    WA = np.asarray(M["WA"]); g = np.asarray(M["gf"])
    def contrib(v): c = (v / sd * g) @ WA; return c - c.mean(-1, keepdims=True)
    tot = contrib(x3 - mu); vt = tot.var()
    heads = np.asarray(rec["heads2"])[i, :, 36 + q]                         # (n,H,d)
    share = np.array([float((contrib(heads[:, h] - heads[:, h].mean(-1, keepdims=True)) * tot).sum() / max((tot * tot).sum(), 1e-12)) for h in range(4)])
    return mass, share


def mixture_heads(mass, share, rule=MIX_RULE):
    """THE head-selection rule for every XIII-B script: mass >= 0.8 and share >= 0.2 (XIII frozen rule).  'mass-only' list reported separately."""
    return [h for h in range(4) if mass[h] >= rule["mass"] and share[h] >= rule["share"]], [h for h in range(4) if mass[h] >= rule["mass"]]
