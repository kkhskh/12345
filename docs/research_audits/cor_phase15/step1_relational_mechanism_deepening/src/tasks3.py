"""Tasks for Foundational Investigation III (same 6-question x 6-source token layout as all earlier packages).

TRANS family (continuation's tasks_ext.sample_trans semantics: A reports Y, B reports sigma(Y), C reports tau(sigma(Y)),
each source correct w.p. beta else uniform over the K-1 others; question types AB / BC / C-only uniform):
  sample_trans_pairs(n, rng, allowed) : (sigma, tau) drawn uniformly from an allowed list of index pairs
  HOLDOUT_PAIRS                        : 12 of the 36 (sigma, tau) pairs, (si + ti) % 3 == 0; every sigma and every tau
                                         still occurs in 4 training pairs, every composite tau.sigma occurs in training
  map targets for auxiliary supervision: sigma index, tau index (0..5)
ADAPT (adaptive / change-point reliability, mandate Workstream F):
  one reliable group g1 for questions q < c and g2 for q >= c; the other groups are noise (T1 roles).
  Prior: w.p. 1/3 no change (c = 6, g2 = g1), else c uniform on {1..5}, g1 uniform, g2 uniform on the other two groups.
  bayes_adapt: exact posterior by enumeration of the 33 (g1, g2, c) configurations, leave-q-out as in all earlier packages.
  bayes_adapt_static: the static T1 Bayes reference (count-sufficient; assumes no change) applied to ADAPT data.
"""
import itertools, os, sys
import numpy as np
HERE = os.path.dirname(__file__)
for p_ in ["../../step1_foundational_atlas_continuation/src", "../../step1_relational_computation_atlas/src", "../../step1_latent_reliability/src"]:
    sys.path.insert(0, os.path.join(HERE, p_))
import tasks_ext as TX, gen_family as G
K, Q = 3, 6
PERMS = TX.PERMS; MASK = TX.MASK
ALL_PAIRS = [(i, j) for i in range(6) for j in range(6)]
HOLDOUT_PAIRS = [(i, j) for i, j in ALL_PAIRS if (i + j) % 3 == 0]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in HOLDOUT_PAIRS]


def sample_trans_pairs(n, rng, allowed=ALL_PAIRS, beta=0.9):
    idx = rng.integers(0, len(allowed), n); si = np.array([allowed[i][0] for i in idx]); ti = np.array([allowed[i][1] for i in idx])
    Y = rng.integers(0, K, (n, Q)); S = np.stack([PERMS[i] for i in si]); Tm = np.stack([PERMS[i] for i in ti])
    vA = Y; vB = np.take_along_axis(S, Y, 1); vC = np.take_along_axis(Tm, vB, 1)
    def noisy(v):
        ok = rng.random(v.shape) < beta; return np.where(ok, v, (v + rng.integers(1, K, v.shape)) % K)
    typ = rng.integers(0, 3, (n, Q)); X = np.zeros((n, Q, 6), int)
    X[:, :, 0] = np.where(typ == 0, noisy(vA), MASK); X[:, :, 1] = np.where(typ == 0, noisy(vA), MASK)
    X[:, :, 2] = np.where(typ <= 1, noisy(vB), MASK); X[:, :, 3] = np.where(typ <= 1, noisy(vB), MASK)
    X[:, :, 4] = np.where(typ >= 1, noisy(vC), MASK); X[:, :, 5] = np.where(typ >= 1, noisy(vC), MASK)
    return dict(X=X, Y=Y, sigma=si, tau=ti, typ=typ)


def composite_index(si, ti):
    """index of tau.sigma in PERMS"""
    out = []
    for a, b in zip(si, ti):
        c = PERMS[b][PERMS[a]]; out.append([k for k, P in enumerate(PERMS) if (P == c).all()][0])
    return np.array(out)


# ------------------------------------------------------------------ ADAPT
ADAPT_CFGS = [(g, g, 6) for g in range(3)] + [(g1, g2, c) for g1 in range(3) for g2 in range(3) if g2 != g1 for c in range(1, 6)]
ADAPT_PRIOR = np.array([(1 / 3) / 3 if c == 6 else (2 / 3) / 30 for (_, _, c) in ADAPT_CFGS])


def sample_adapt(n, rng, alpha=0.8):
    ci = rng.choice(len(ADAPT_CFGS), n, p=ADAPT_PRIOR); Y = rng.integers(0, K, (n, Q)); X = rng.integers(0, K, (n, Q, 6))
    for i, (g1, g2, c) in enumerate(ADAPT_CFGS):
        s = np.where(ci == i)[0]
        if not len(s): continue
        for q in range(Q):
            g = g1 if q < c else g2
            for src in (2 * g, 2 * g + 1):
                ok = rng.random(len(s)) < alpha; wrong = (Y[s, q] + rng.integers(1, K, len(s))) % K
                X[s, q, src] = np.where(ok, Y[s, q], wrong)
    return dict(X=X, Y=Y, cfg=ci)


def _adapt_joint(X, alpha=0.8):
    """J (N,Q,C,K) = P(X_q, Y_q = y | cfg)"""
    e = (1 - alpha) / (K - 1); ys = np.arange(K); out = []
    for (g1, g2, c) in ADAPT_CFGS:
        J = np.full(X.shape[:2] + (K,), 1.0 / K)
        for q in range(Q):
            g = g1 if q < c else g2
            for gg in range(3):
                a, b = 2 * gg, 2 * gg + 1
                if gg == g:
                    pa = np.where(X[:, q, a, None] == ys, alpha, e); pb = np.where(X[:, q, b, None] == ys, alpha, e); J[:, q] *= pa * pb
                else:
                    J[:, q] *= 1.0 / K ** 2
        out.append(J)
    return np.stack(out, 2)


def bayes_adapt(X, alpha=0.8):
    J = _adapt_joint(X, alpha); lM = np.log(J.sum(-1)); lp = np.log(ADAPT_PRIOR)
    loo = np.stack([lM[:, [k for k in range(Q) if k != q]].sum(1) for q in range(Q)], 1) + lp
    w = np.exp(loo - loo.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    py = (w[..., None] * J).sum(2); py /= py.sum(-1, keepdims=True)
    tot = lM.sum(1) + lp; pc = np.exp(tot - tot.max(-1, keepdims=True)); pc /= pc.sum(-1, keepdims=True)
    return dict(post_y=py, post_cfg=pc)


def bayes_adapt_static(X, alpha=0.8):
    return G.posterior(G.Fam(pc=0.0, alpha=alpha), X)["post_y"]
