"""XV task family (GATE variants).  Six source positions per question, K = 3, Q = 6 questions, Y ~ U(K).
Gate arity a in {2, 3}:  g = [x0 + ... + x_{a-1} = 0 mod 3], drawn with P(g=1) = 1/2 by choosing the LAST gate source given the others
(a = 2: x1 | x0;  a = 3: x2 | x0, x1).  For a = 2, position 2 is an irrelevant uniform source.
Evidence: positions 3, 4, 5.  g = 1: sources 3 and 4 report Y with prob beta (else uniform over the other K-1), source 5 is uniform noise.
          g = 0: source 5 reports Y with prob beta, sources 3 and 4 are uniform noise.
So the trust relation selects between an agreeing PAIR (3,4) and a SINGLE source 5.  Parameters: a, beta, nu.
nu = P(x3 == x4) when the pair is noise (g = 0): x3 uniform, x4 = x3 with prob nu, else uniform over the other K-1 values
(nu = 1/3 is independent noise; nu = 0 forces the noise pair to disagree, which makes pair agreement a strong trust cue).
Single-source marginals stay uniform for every nu.
Every single source is marginally uniform under both gate states; for a = 3 every PAIR of gate sources is too (3-wise relation)."""
import itertools
import numpy as np
K, Q = 3, 6
def sample(n, rng, a, beta, nu=1/3):
    Y = rng.integers(0, K, (n, Q)); X = np.zeros((n, Q, 6), int); g = rng.random((n, Q)) < 0.5
    for s in range(a - 1): X[:, :, s] = rng.integers(0, K, (n, Q))
    part = X[:, :, :a - 1].sum(-1) % K; last_open = (-part) % K; last_closed = (last_open + rng.integers(1, K, (n, Q))) % K
    X[:, :, a - 1] = np.where(g, last_open, last_closed)
    if a == 2: X[:, :, 2] = rng.integers(0, K, (n, Q))
    def rep(): ok = rng.random((n, Q)) < beta; return np.where(ok, Y, (Y + rng.integers(1, K, (n, Q))) % K)
    n3 = rng.integers(0, K, (n, Q)); n4 = np.where(rng.random((n, Q)) < nu, n3, (n3 + rng.integers(1, K, (n, Q))) % K)
    X[:, :, 3] = np.where(g, rep(), n3); X[:, :, 4] = np.where(g, rep(), n4)
    X[:, :, 5] = np.where(g, rng.integers(0, K, (n, Q)), rep())
    return dict(X=X, Y=Y, g=g)
def gate(X, a): return X[..., :a].sum(-1) % K == 0
def lik(X, S, beta):
    e = (1 - beta) / (K - 1); ys = np.arange(K); out = np.ones(X.shape[:-1] + (K,))
    for s in S: out = out * np.where(X[..., s, None] == ys, beta, e)
    return out
def bayes(X, a, beta):
    L = np.where(gate(X, a)[..., None], lik(X, (3, 4), beta), lik(X, (5,), beta)); return L / L.sum(-1, keepdims=True)
def noisepair(X, nu): return np.where(X[..., 3] == X[..., 4], nu, (1 - nu) / (K - 1)) / K
def gatefree(X, a, beta, nu=1/3):
    """exact posterior after marginalising g (the gate-free mixture)."""
    L = 0.5 * lik(X, (3, 4), beta) / K + 0.5 * lik(X, (5,), beta) * noisepair(X, nu)[..., None]
    return L / L.sum(-1, keepdims=True)
def gate_flip(X, a, rng):
    Xf = X.copy(); part = X[..., :a - 1].sum(-1) % K; g = gate(X, a)
    closed = ((-part) % K + rng.integers(1, K, part.shape)) % K; Xf[..., a - 1] = np.where(g, closed, (-part) % K); return Xf
def gate_keep_shift(X, a, rng):
    Xs = X.copy(); k = rng.integers(1, K, X.shape[:-1]); i, j = (0, 1) if a == 2 or rng.random() < 0.5 else (1, 2)
    Xs[..., i] = (X[..., i] + k) % K; Xs[..., j] = (X[..., j] - k) % K; return Xs
def swap45(X): Xs = X.copy(); Xs[..., 4], Xs[..., 5] = X[..., 5], X[..., 4]; return Xs
# ---- exact enumeration over all 3^6 inputs (one question)
XALL = np.array(list(itertools.product(range(K), repeat=6)))
def joint(a, beta, nu=1/3):
    """exact P(x) and P(y|x) (Bayes) for all 729 inputs."""
    e = (1 - beta) / (K - 1); px = np.zeros(len(XALL)); pxy = np.zeros((len(XALL), K))
    for i, x in enumerate(XALL):
        g = (x[:a].sum() % K) == 0
        pgate = (1 / K) ** (a - 1) * 0.5 * (1.0 if g else 0.5) * ((1 / K) if a == 2 else 1.0)     # free gate sources, last source, irrelevant x2
        for y in range(K):
            pe = 1.0
            if g:
                for s in (3, 4): pe *= (beta if x[s] == y else e)
            else: pe *= (nu if x[3] == x[4] else (1 - nu) / (K - 1)) / K
            pe *= ((1 / K) if g else (beta if x[5] == y else e))
            pxy[i, y] = pgate * pe / K
    px = pxy.sum(1); return px, pxy / px[:, None]
