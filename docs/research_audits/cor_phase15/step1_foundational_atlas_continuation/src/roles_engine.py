"""Exact Bayes for generalised group-role generators (3 known pairs, K candidates, Q questions).
A role is a function P(xa, xb | y) -> (..., K).  A configuration assigns a role to each of the 3 groups; the prior is
over a finite list of configurations.  Per-episode latent quantities (e.g. a mixture over reliable accuracy) are
expressed as distinct roles in distinct configurations."""
import numpy as np
GROUPS = [(0, 1), (2, 3), (4, 5)]
def rel(alpha, K=3, bias=0.0, kappa=0.0):
    """reliable: correct w.p. alpha; errors go to (y+1)%K w.p. bias (else uniform over the K-1 wrong values)."""
    def f(xa, xb):
        ys = np.arange(K)
        def one(x):
            wrong_b = np.where(x[..., None] == (ys + 1) % K, 1.0, 0.0)
            e = (1 - alpha) * (bias * wrong_b + (1 - bias) * (x[..., None] != ys) / (K - 1))
            return np.where(x[..., None] == ys, alpha, e)
        pa, pb = one(xa), one(xb)
        return (1 - kappa) * pa * pb + kappa * (xa == xb)[..., None] * pa
    return f
def noise(K=3):
    return lambda xa, xb: np.full(xa.shape + (K,), 1.0 / K ** 2)
def coll(rho, K=3, acc=None):
    """colluding pair: w.p. rho both copy a common c; c ~ U(K) (acc=None) or c = y w.p. acc (informative colluder)."""
    def f(xa, xb):
        ys = np.arange(K)
        if acc is None: pc = np.full(xa.shape + (K,), 1.0 / K)
        else: pc = np.where(xa[..., None] == ys, acc, (1 - acc) / (K - 1))
        return rho * (xa == xb)[..., None] * pc + (1 - rho) / K ** 2
    return f
def sample_role(spec, n, Q, Y, rng, K=3):
    kind = spec[0]
    if kind == "rel":
        _, alpha, bias, kappa = spec
        def one():
            ok = rng.random((n, Q)) < alpha; b = rng.random((n, Q)) < bias
            wr = np.where(b, (Y + 1) % K, (Y + rng.integers(1, K, (n, Q))) % K); return np.where(ok, Y, wr)
        a = one(); bb = one()
        if kappa > 0: bb = np.where(rng.random((n, Q)) < kappa, a, bb)
        return a, bb
    if kind == "noise": return rng.integers(0, K, (n, Q)), rng.integers(0, K, (n, Q))
    if kind == "coll":
        _, rho, acc = spec
        c = rng.integers(0, K, (n, Q)) if acc is None else np.where(rng.random((n, Q)) < acc, Y, (Y + rng.integers(1, K, (n, Q))) % K)
        cop = rng.random((n, Q)) < rho
        return np.where(cop, c, rng.integers(0, K, (n, Q))), np.where(cop, c, rng.integers(0, K, (n, Q)))
def lik_of(spec, K=3):
    if spec[0] == "rel": return rel(spec[1], K, spec[2], spec[3])
    if spec[0] == "noise": return noise(K)
    return coll(spec[1], K, spec[2])
class Gen:
    def __init__(self, configs, prior, K=3, Q=6):
        self.configs, self.prior, self.K, self.Q = configs, np.asarray(prior, float) / np.sum(prior), K, Q
    def sample(self, n, rng):
        K, Q = self.K, self.Q; ci = rng.choice(len(self.configs), n, p=self.prior); Y = rng.integers(0, K, (n, Q)); X = np.zeros((n, Q, 6), int)
        for i, cfg in enumerate(self.configs):
            s = ci == i
            if not s.any(): continue
            for (a, b), spec in zip(GROUPS, cfg):
                xa, xb = sample_role(spec, s.sum(), Q, Y[s], rng, K); X[s, :, a] = xa; X[s, :, b] = xb
        return dict(X=X, Y=Y, cfg=ci)
    def posterior(self, X):
        J = []
        for cfg in self.configs:
            l = np.ones(X.shape[:2] + (self.K,))
            for (a, b), spec in zip(GROUPS, cfg): l = l * lik_of(spec, self.K)(X[:, :, a], X[:, :, b])
            J.append(l / self.K)
        J = np.stack(J, 2)                                   # N,Q,C,K
        with np.errstate(divide="ignore"): lM = np.log(J.sum(-1))
        lp = np.log(self.prior)
        tot = lM.sum(1) + lp; m = tot.max(-1, keepdims=True); pc = np.exp(tot - m); pc /= pc.sum(-1, keepdims=True)
        loo = np.stack([lM[:, [k for k in range(lM.shape[1]) if k != q]].sum(1) for q in range(lM.shape[1])], 1) + lp  # direct leave-one-out (avoids inf - inf)
        m2 = loo.max(-1, keepdims=True); w = np.exp(loo - m2); w /= w.sum(-1, keepdims=True)
        py = (w[..., None] * J).sum(2); py /= py.sum(-1, keepdims=True)
        return dict(post_y=py, post_cfg=pc)
