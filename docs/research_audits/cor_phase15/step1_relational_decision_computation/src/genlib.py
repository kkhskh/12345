"""Investigation IX generator family and exact Bayes (numpy).

Generalised T4 ("role model"): G groups of 2 sources, K candidates, Q questions, reliability alpha.
A latent role assignment R picks, uniformly among all injective assignments, ONE reliable group and C colluding groups; the remaining
groups are noise.  Per question q (independently, Y_q ~ Uniform(K)):
  reliable source:  reports Y_q w.p. alpha, else one of the K-1 wrong answers uniformly           (independently per source)
  colluding group:  both sources report the same c_q ~ Uniform(K), independent of Y_q (with prob. beta; with prob 1-beta each source
                    reports independently uniformly -- 'partial copying'; beta = 1 is the original T4)
  noise source:     uniform over K, independent
G=3, C=1, beta=1, K=3, Q=6, alpha=0.8 is exactly Investigation VI-VIII's T4 (verified in scripts/verify_math.py against bayes.sample_fast
statistics and against Investigation VII's exact-Bayes logits).

Exact posterior p(Y_q = y | X) by brute-force enumeration over role assignments R (no closed form assumed) -- `posterior_brute`.
Closed form for C = 1, beta = 1 (Investigation VII; m_g WITHOUT any vote-matching factor) -- `posterior_closed`.
"""
import itertools
import numpy as np


class Env:
    def __init__(self, G=3, C=1, K=3, Q=6, alpha=0.8, beta=1.0):
        assert 1 + C <= G
        self.G, self.C, self.K, self.Q, self.alpha, self.beta = G, C, K, Q, alpha, beta
        self.roles = [(g, hs) for g in range(G) for hs in itertools.combinations([h for h in range(G) if h != g], C)]

    def __repr__(self): return f"Env(G={self.G},C={self.C},K={self.K},Q={self.Q},alpha={self.alpha},beta={self.beta})"


def sample(env, n, rng, ridx=None):
    G, K, Q = env.G, env.K, env.Q
    ridx = rng.integers(0, len(env.roles), n) if ridx is None else np.asarray(ridx)
    Y = rng.integers(0, K, (n, Q))
    X = rng.integers(0, K, (n, Q, 2 * G))
    off = rng.integers(1, K, (n, Q, 2 * G)); u = rng.random((n, Q, 2 * G))
    col = rng.integers(0, K, (n, Q)); cop = rng.random((n, Q)) < env.beta
    rel = np.zeros((n, G), bool); coll = np.zeros((n, G), bool)
    for i, r in enumerate(ridx):
        g, hs = env.roles[r]; rel[i, g] = True; coll[i, list(hs)] = True
    for s in range(2 * G):
        gi = s // 2
        rep = np.where(u[:, :, s] < env.alpha, Y, (Y + off[:, :, s]) % K)
        X[:, :, s] = np.where(rel[:, gi, None], rep, X[:, :, s])
        X[:, :, s] = np.where(coll[:, gi, None] & cop, col, X[:, :, s])
    return dict(X=X, Y=Y, ridx=ridx)


def _group_lik(env, X):
    """Per question, per group, likelihood factors of the group's two reports under each role, as functions of y.
    Returns rel (n,Q,G,K): p(x_pair | reliable, Y=y);  col (n,Q,G): p(x_pair | colluding);  noi: p(x_pair | noise) = 1/K^2."""
    K, a = env.K, env.alpha
    a_ = X.reshape(*X.shape[:2], env.G, 2)
    y = np.arange(K)
    p1 = np.where(a_[..., 0, None] == y, a, (1 - a) / (K - 1)); p2 = np.where(a_[..., 1, None] == y, a, (1 - a) / (K - 1))
    rel = p1 * p2
    eq = (a_[..., 0] == a_[..., 1])
    col = env.beta * eq / K + (1 - env.beta) / K ** 2
    return rel, col, 1.0 / K ** 2


def posterior_brute(env, X, q):
    """WARNING (Investigation XII): for C >= 2 this is the posterior of an INDEPENDENT-copying model; `sample` shares the copied value AND the
    copy event across colluder groups.  Exact for C = 1 only.  Use xii_bayes.posterior_shared for the actual sampler.
    Exact p(Y_q | X) for target question q (array of ints, one per episode), enumerating role assignments."""
    n = len(X); idx = np.arange(n); K = env.K
    rel, col, noi = _group_lik(env, X)                         # rel n,Q,G,K
    relm = rel.mean(-1)                                        # marginal over Y_{q'} (uniform prior)
    logw = []; numer = []
    for g, hs in env.roles:
        others = [h for h in range(env.G) if h != g and h not in hs]
        # questions q' != q: marginal likelihood; question q: keep y
        lq = np.log(relm[:, :, g]) + sum(np.log(col[:, :, h]) for h in hs) + len(others) * np.log(noi)   # n,Q
        mask = np.ones_like(lq, bool); mask[idx, q] = False
        tot = np.where(mask, lq, 0.0).sum(1)
        lyq = np.log(rel[idx, q, g, :]) + sum(np.log(col[idx, q, h]) for h in hs)[:, None] + len(others) * np.log(noi)
        numer.append(tot[:, None] + lyq)
    L = np.stack(numer)                                        # R,n,K
    m = L.max((0, 2), keepdims=True); P = np.exp(L - m).sum(0)
    return P / P.sum(-1, keepdims=True)


def counts(X, G):
    """n_g(q): number of OTHER questions on which group g's two sources agree.  (n,Q,G)"""
    a_ = X.reshape(*X.shape[:2], G, 2); eq = (a_[..., 0] == a_[..., 1]).astype(int)
    return eq.sum(1, keepdims=True) - eq


def eligibility(X, G, q):
    """Mandate 2.4 / Investigation VII:  m_g = sum_{h != g} 1[n_h = Q-1] * 1[group h's two sources agree at q].  (n,G)  -- no vote matching."""
    idx = np.arange(len(X)); Q = X.shape[1]; n = counts(X, G)[idx, q]
    a_ = X.reshape(len(X), Q, G, 2)[idx, q]; e = (n == Q - 1) & (a_[..., 0] == a_[..., 1])
    return e.sum(1, keepdims=True) - e, e


def posterior_closed(env, X, q):
    """Closed form (C=1, beta=1):  p(y) ∝ sum_g RHO^{n_g} * m_g * L_g(y),   RHO = E_rel[agree] / E_col[agree]... derived per question:
    reliable-group pair marginal vs colluder: ratio of per-question likelihoods.  Implemented generically from the per-question factors."""
    assert env.C == 1 and env.beta == 1.0
    n_ = len(X); idx = np.arange(n_)
    rel, col, noi = _group_lik(env, X); relm = rel.mean(-1)
    m, e = eligibility(X, env.G, q)
    # weight of 'g reliable' relative to 'g noise' on questions != q :  prod_{q'} relm/noi
    lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[idx, q])            # n,G
    Lg = rel[idx, q] / noi                                                  # n,G,K  (rel at q vs noise)
    P = (W[..., None] * m[..., None] * Lg).sum(1)
    return P / P.sum(-1, keepdims=True)


def deepsets_factor(env, X, q):
    """Theorem 1 representation: per-group phi_g = (e_g, W_g L_g(.), W_g e_g L_g(.)) ; pooled S, A, B ; p(y) ∝ S*A_y - B_y."""
    n_ = len(X); idx = np.arange(n_)
    rel, col, noi = _group_lik(env, X); relm = rel.mean(-1)
    _, e = eligibility(X, env.G, q)
    lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[idx, q]); Lg = rel[idx, q] / noi
    S = e.sum(1); A = (W[..., None] * Lg).sum(1); Bv = (W[..., None] * e[..., None] * Lg).sum(1)
    P = S[:, None] * A - Bv
    return P / P.sum(-1, keepdims=True), dict(S=S, A=A, B=Bv)
