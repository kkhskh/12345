"""XVI task family: the XV GATE variants with two independently adjustable evidence reliabilities, and an arity-4 extension.

S source positions per question (S = 6 is the neural format of XIV/XV; S = 7 is used only for the arity-4 mathematics), K = 3, Q = 6.
Gate sources 0..a-1:  g = [x0 + ... + x_{a-1} = 0 mod 3], P(g = 1) = 1/2, obtained by choosing the LAST gate source given the others.
Irrelevant uniform sources fill positions a .. S-4 (for S = 6, a = 2 this is XV's irrelevant x2).
Evidence positions (P1, P2, P0) = (S-3, S-2, S-1):
   g = 1: the PAIR P1, P2 each reports Y with probability b1 (else uniform over the other K-1 values); P0 is uniform noise.
   g = 0: the SINGLE source P0 reports Y with probability b0; the pair is noise with P(P1 == P2) = nu.
b1 = b0 = beta reproduces XV exactly (same random-number consumption, so XV seeds and streams carry over: verified in xvi_math_verify.py).
Every single source is marginally uniform for all (b1, b0, nu)."""
import itertools
import numpy as np
K, Q = 3, 6


def sample(n, rng, a, b1, b0, nu=1 / 3, S=6):
    """Random-number consumption is identical to XV's xvtask.sample for S = 6 (whatever b1, b0, nu)."""
    P1, P2, P0 = S - 3, S - 2, S - 1
    Y = rng.integers(0, K, (n, Q)); X = np.zeros((n, Q, S), int); g = rng.random((n, Q)) < 0.5
    for s in range(a - 1): X[:, :, s] = rng.integers(0, K, (n, Q))
    part = X[:, :, :a - 1].sum(-1) % K; last_open = (-part) % K; last_closed = (last_open + rng.integers(1, K, (n, Q))) % K
    X[:, :, a - 1] = np.where(g, last_open, last_closed)
    for s in range(a, S - 3): X[:, :, s] = rng.integers(0, K, (n, Q))
    def rep(b): ok = rng.random((n, Q)) < b; return np.where(ok, Y, (Y + rng.integers(1, K, (n, Q))) % K)
    n3 = rng.integers(0, K, (n, Q)); n4 = np.where(rng.random((n, Q)) < nu, n3, (n3 + rng.integers(1, K, (n, Q))) % K)
    X[:, :, P1] = np.where(g, rep(b1), n3); X[:, :, P2] = np.where(g, rep(b1), n4)
    X[:, :, P0] = np.where(g, rng.integers(0, K, (n, Q)), rep(b0))
    return dict(X=X, Y=Y, g=g)


def gate(X, a): return X[..., :a].sum(-1) % K == 0


def lik(X, Ss, b):
    e = (1 - b) / (K - 1); ys = np.arange(K); out = np.ones(X.shape[:-1] + (K,))
    for s in Ss: out = out * np.where(X[..., s, None] == ys, b, e)
    return out


def bayes(X, a, b1, b0):
    S = X.shape[-1]; L = np.where(gate(X, a)[..., None], lik(X, (S - 3, S - 2), b1), lik(X, (S - 1,), b0)); return L / L.sum(-1, keepdims=True)


def noisepair(X, nu): S = X.shape[-1]; return np.where(X[..., S - 3] == X[..., S - 2], nu, (1 - nu) / (K - 1)) / K


def gatefree(X, a, b1, b0, nu=1 / 3):
    """exact posterior after marginalising g (the gate-free mixture)."""
    S = X.shape[-1]
    L = 0.5 * lik(X, (S - 3, S - 2), b1) / K + 0.5 * lik(X, (S - 1,), b0) * noisepair(X, nu)[..., None]
    return L / L.sum(-1, keepdims=True)


def gate_flip(X, a, rng):
    """redraw the last gate source so that g switches (on-support; uniform over the valid values); everything else kept."""
    Xf = X.copy(); part = X[..., :a - 1].sum(-1) % K; g = gate(X, a)
    closed = ((-part) % K + rng.integers(1, K, part.shape)) % K; Xf[..., a - 1] = np.where(g, closed, (-part) % K); return Xf


def swap_pair_single(X):
    """swap the second pair source with the single source (XV's swap(4,5) for S = 6)."""
    S = X.shape[-1]; Xs = X.copy(); Xs[..., S - 2], Xs[..., S - 1] = X[..., S - 1], X[..., S - 2]; return Xs


def agree(X): S = X.shape[-1]; return X[..., S - 3] == X[..., S - 2]


def xall(S): return np.array(list(itertools.product(range(K), repeat=S)))


def joint(a, b1, b0, nu=1 / 3, S=6):
    """exact P(x) and P(y | x) (Bayes) for all K^S inputs of one question (vectorised)."""
    XA = xall(S); P1, P2, P0 = S - 3, S - 2, S - 1; g = (XA[:, :a].sum(1) % K) == 0
    pgate = (1 / K) ** (a - 1) * 0.5 * np.where(g, 1.0, 0.5) * (1 / K) ** (S - 3 - a)
    ys = np.arange(K)
    e1, e0 = (1 - b1) / (K - 1), (1 - b0) / (K - 1)
    pair = np.where(XA[:, P1, None] == ys, b1, e1) * np.where(XA[:, P2, None] == ys, b1, e1) * (1 / K)
    single = np.where(XA[:, P0, None] == ys, b0, e0) * (np.where(XA[:, P1] == XA[:, P2], nu, (1 - nu) / (K - 1)) / K)[:, None]
    pxy = pgate[:, None] * np.where(g[:, None], pair, single) / K
    px = pxy.sum(1); return px, pxy / px[:, None]
