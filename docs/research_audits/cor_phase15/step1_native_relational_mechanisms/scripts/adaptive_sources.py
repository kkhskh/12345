"""Workstream F4: an adaptive-source variant and its exact Bayes rule; sufficiency of static statistics (numpy only).
Model (G=3 groups of 2, K=3, Q=6, alpha=0.8): role R=(g reliable, h colluder) uniform as in T4.  On each question the colluding group's
common report c_q equals the reliable group's FIRST source's current report with probability gamma (it 'follows' the reliable group),
otherwise c_q ~ U(K).  gamma = 0 is exactly T4.  The colluder is therefore no longer independent of the answer: it carries diluted,
correlated evidence, and the cross-group relation (does h echo g?) becomes decision-relevant.
Exact posterior by enumeration over roles (per-question likelihoods conditional on g's reports).
Sufficiency tests (max within-cell spread of the exact posterior over 300,000 decisions):
  T_within  = (within-group leave-q counts n_g, target-question reports)                [the T4 sufficient statistic]
  T_cross   = T_within + for each ordered pair (g,h): #questions q' != q with x_{h,0} = x_{g,0}   (cross-group echo counts)
Also: posterior shift caused by the echo relation at fixed T_within (how much a T_within-only reader must lose).  -> logs/adaptive_sources.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
np.seterr(all="ignore")
K, Q, G, ALPHA = 3, 6, 3, 0.8
ROLES = [(g, h) for g in range(G) for h in range(G) if g != h]
def sample(n, gamma, rng):
    r = rng.integers(0, len(ROLES), n); g = np.array([ROLES[k][0] for k in r]); h = np.array([ROLES[k][1] for k in r])
    Y = rng.integers(0, K, (n, Q)); X = rng.integers(0, K, (n, Q, 2 * G)); ii = np.arange(n)
    for j in (0, 1):
        ok = rng.random((n, Q)) < ALPHA; X[ii, :, 2 * g + j] = np.where(ok, Y, (Y + rng.integers(1, K, (n, Q))) % K)
    echo = rng.random((n, Q)) < gamma; c = np.where(echo, X[ii, :, 2 * g], rng.integers(0, K, (n, Q)))
    X[ii, :, 2 * h] = c; X[ii, :, 2 * h + 1] = c
    return X, Y
def posterior(X, q, gamma):
    n = len(X); idx = np.arange(n); y = np.arange(K); out = np.zeros((n, K)); logs = []
    a = X.reshape(n, Q, G, 2)
    for g, h in ROLES:
        relp = lambda v: np.where(v[..., None] == y, ALPHA, (1 - ALPHA) / (K - 1))
        rel = relp(a[:, :, g, 0]) * relp(a[:, :, g, 1])                       # n,Q,K   p(x_g | Y=y)
        eq = a[:, :, h, 0] == a[:, :, h, 1]
        col = eq * (gamma * (a[:, :, h, 0] == a[:, :, g, 0]) + (1 - gamma) / K)  # n,Q    p(x_h | x_g, colluder)
        o = [k for k in range(G) if k not in (g, h)][0]; noi = 1.0 / K ** 2
        lq = np.log(rel.mean(-1)) + np.log(col) + np.log(noi)                 # marginal over Y_q'
        mask = np.ones((n, Q), bool); mask[idx, q] = False; tot = np.where(mask, lq, 0).sum(1)
        ly = np.log(rel[idx, q] / K) + np.log(col[idx, q])[:, None] + np.log(noi)
        logs.append(tot[:, None] + ly)
    L = np.stack(logs); m = L.max((0, 2), keepdims=True); P = np.exp(L - m).sum(0); return P / P.sum(-1, keepdims=True)
def spread(P, keys):
    k = np.unique(keys, axis=0, return_inverse=True)[1].ravel(); order = np.argsort(k); ks = k[order]; Ps = P[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]; ends = np.r_[starts[1:], len(ks)]; mx = 0.0
    for s0, e0 in zip(starts, ends):
        if e0 - s0 > 1: mx = max(mx, float((Ps[s0:e0].max(0) - Ps[s0:e0].min(0)).max()))
    return mx, int(len(starts))
out = {}
for gamma in (0.0, 0.5, 0.9):
    rng = np.random.default_rng(int(gamma * 1000) + 5); X, Y = sample(300_000, gamma, rng); q = rng.integers(0, Q, len(X)); idx = np.arange(len(X))
    P = posterior(X, q, gamma); a = X.reshape(len(X), Q, G, 2); eqm = (a[..., 0] == a[..., 1]).astype(int)
    nw = eqm.sum(1) - eqm[idx, q]; rep = a[idx, q].reshape(len(X), -1)
    Tw = np.concatenate([nw, rep], 1)
    cross = []
    for g, h in itertools.permutations(range(G), 2):
        e = (a[:, :, h, 0] == a[:, :, g, 0]).astype(int); cross.append(e.sum(1) - e[idx, q])
    Tc = np.concatenate([Tw, np.stack(cross, 1)], 1)
    sw, cw = spread(P, Tw); sc, cc = spread(P, Tc)
    # check gamma=0 posterior equals T4 closed form behaviour: compare with genlib brute force
    out[f"gamma{gamma}"] = dict(spread_T_within=sw, cells_T_within=cw, spread_T_cross=sc, cells_T_cross=cc,
                               acc_bayes=float((P.argmax(-1) == Y[idx, q]).mean()))
    print(gamma, out[f"gamma{gamma}"], flush=True)
sys.path.insert(0, os.path.join(HERE, "..", "src")); import genlib as GL
Xc, _ = sample(3000, 0.0, np.random.default_rng(1)); qc = np.random.default_rng(2).integers(0, Q, 3000)
out["gamma0_matches_T4_brute"] = float(np.abs(posterior(Xc, qc, 0.0) - GL.posterior_brute(GL.Env(G=3, C=1), Xc, qc)).max())
json.dump(out, open(os.path.join(HERE, "..", "logs", "adaptive_sources.json"), "w"), indent=1); print(out["gamma0_matches_T4_brute"])
