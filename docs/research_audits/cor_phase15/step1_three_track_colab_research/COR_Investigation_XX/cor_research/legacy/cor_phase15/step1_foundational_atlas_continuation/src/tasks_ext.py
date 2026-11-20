"""New tasks for the continuation (same token layout as the latent-reliability Transformer: Q questions x 6 sources).

PRETRAINING TASKS (PP3 follow-up, mandate §4)
  P-local : T1-style sources, but the reliable group is resampled independently PER QUESTION.  Target: answer Y_q.
            Within-question pair parsing / agreement is useful; cross-question sharing is useless (π_{-q} = prior).
  P-group : T1 episodes (shared R).  Target at readout q: for each group g, whether g's pair agrees in question
            (q+1) mod Q (3 binary).  Requires cross-question reading of ONE other question's pair agreement; no counting,
            no answer.
  P-count : T1 episodes.  Target at readout q: the leave-one-out agreement count n_g^{-q} for each group (3 x 6-way).
            Requires agreement detection + aggregation over all other questions; no reliability->answer routing.
BEYOND-RELIABILITY TASKS (mandate §11)
  GATE    : within-question constraint-gated evidence.  Per question, sources 0,1 satisfy x0 + x1 = 0 (mod K) ("gate
            open") with prob 1/2.  Open: sources 2,3 report Y w.p. beta, 4,5 uniform noise.  Closed: 2,3 noise, 4,5 report
            Y w.p. beta.  The relation (a modular constraint between two non-answer sources) decides which evidence pair
            to trust.  Nothing is shared across questions.
  TRANS   : cross-question latent transport with two hops.  Latent permutations sigma, tau of [K], shared by the episode.
            Channel A (sources 0,1) report Y, channel B (2,3) report sigma(Y), channel C (4,5) report tau(sigma(Y)),
            each source correct w.p. beta, else uniform over the other K-1 values.  In every question exactly one of
            A or C is MASKED (token value K): B is always present.  Decoding a C-only question needs (tau sigma)^{-1},
            which is identified only by chaining A-B co-occurrences (sigma) and B-C co-occurrences (tau) across questions.
Exact Bayes is given for GATE (per question) and TRANS (enumeration over the 36 (sigma,tau) pairs).
"""
from __future__ import annotations
import itertools, os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_computation_atlas", "src"))
import gen_family as G

K, Q = 3, 6


# ------------------------------------------------------------------ pretraining tasks
def sample_plocal(n, rng):
    d = G.sample(G.Fam(pc=0.0, Q=1), n * Q, rng)
    return dict(X=d["X"].reshape(n, Q, 6), Y=d["Y"].reshape(n, Q))


def agree(X):
    return np.stack([(X[:, :, a] == X[:, :, b]) for a, b in G.GROUPS], -1).astype(int)       # N,Q,3


def sample_pgroup(n, rng):
    d = G.sample(G.Fam(pc=0.0), n, rng); A = agree(d["X"])
    T = np.roll(A, -1, axis=1)                                                                 # target at q: agreement in q+1
    return dict(X=d["X"], Y=d["Y"], T=T)


def sample_pcount(n, rng):
    d = G.sample(G.Fam(pc=0.0), n, rng); A = agree(d["X"])
    return dict(X=d["X"], Y=d["Y"], T=A.sum(1, keepdims=True) - A)                            # N,Q,3 in 0..5


# ------------------------------------------------------------------ GATE
def sample_gate(n, rng, beta=0.9):
    Y = rng.integers(0, K, (n, Q)); X = np.zeros((n, Q, 6), int)
    openg = rng.random((n, Q)) < 0.5
    x0 = rng.integers(0, K, (n, Q)); x1_open = (-x0) % K
    x1_closed = (x1_open + rng.integers(1, K, (n, Q))) % K
    X[:, :, 0], X[:, :, 1] = x0, np.where(openg, x1_open, x1_closed)
    def rep(m):
        ok = rng.random((n, Q)) < beta; return np.where(ok, Y, (Y + rng.integers(1, K, (n, Q))) % K)
    for s in (2, 3):
        X[:, :, s] = np.where(openg, rep(None), rng.integers(0, K, (n, Q)))
    for s in (4, 5):
        X[:, :, s] = np.where(openg, rng.integers(0, K, (n, Q)), rep(None))
    return dict(X=X, Y=Y, open=openg)


def bayes_gate(X, beta=0.9):
    e = (1 - beta) / (K - 1); ys = np.arange(K)
    openg = (X[:, :, 0] + X[:, :, 1]) % K == 0
    def lik(srcs):
        out = np.ones(X.shape[:2] + (K,))
        for s in srcs: out = out * np.where(X[:, :, s, None] == ys, beta, e)
        return out
    L = np.where(openg[..., None], lik((2, 3)), lik((4, 5)))
    return L / L.sum(-1, keepdims=True)


def bayes_gate_no_relation(X, beta=0.9):
    """Best rule that does not use the gate relation: mixture over 'pair 2,3 reliable' and 'pair 4,5 reliable'."""
    e = (1 - beta) / (K - 1); ys = np.arange(K)
    def lik(srcs):
        out = np.ones(X.shape[:2] + (K,))
        for s in srcs: out = out * np.where(X[:, :, s, None] == ys, beta, e)
        return out
    L = 0.5 * lik((2, 3)) + 0.5 * lik((4, 5)); return L / L.sum(-1, keepdims=True)


# ------------------------------------------------------------------ TRANS
PERMS = [np.array(p) for p in itertools.permutations(range(K))]
MASK = K


def sample_trans(n, rng, beta=0.9):
    """Question types (uniform): AB (A,B present; C masked), BC (B,C present; A masked), C (only C present)."""
    Y = rng.integers(0, K, (n, Q)); si = rng.integers(0, len(PERMS), n); ti = rng.integers(0, len(PERMS), n)
    S = np.stack([PERMS[i] for i in si]); Tm = np.stack([PERMS[i] for i in ti])
    vA = Y; vB = np.take_along_axis(S, Y, 1); vC = np.take_along_axis(Tm, vB, 1)
    def noisy(v):
        ok = rng.random(v.shape) < beta; return np.where(ok, v, (v + rng.integers(1, K, v.shape)) % K)
    typ = rng.integers(0, 3, (n, Q))                                   # 0 AB, 1 BC, 2 C-only
    X = np.zeros((n, Q, 6), int)
    X[:, :, 0] = np.where(typ == 0, noisy(vA), MASK); X[:, :, 1] = np.where(typ == 0, noisy(vA), MASK)
    X[:, :, 2] = np.where(typ <= 1, noisy(vB), MASK); X[:, :, 3] = np.where(typ <= 1, noisy(vB), MASK)
    X[:, :, 4] = np.where(typ >= 1, noisy(vC), MASK); X[:, :, 5] = np.where(typ >= 1, noisy(vC), MASK)
    return dict(X=X, Y=Y, sigma=si, tau=ti, typ=typ)


def trans_joint(X, beta=0.9):
    """J (N,Q,36,K) = P(X_q, Y_q=y | sigma, tau)."""
    e = (1 - beta) / (K - 1); ys = np.arange(K)
    def f(x, v):                                       # x (N,Q), v (K,) value for each y -> (N,Q,K); masked -> 1
        val = np.where(x[..., None] == v[None, None, :], beta, e)
        return np.where(x[..., None] == MASK, 1.0, val)
    out = []
    for s in PERMS:
        for t in PERMS:
            vA, vB, vC = ys, s[ys], t[s[ys]]
            J = (1.0 / K) * f(X[:, :, 0], vA) * f(X[:, :, 1], vA) * f(X[:, :, 2], vB) * f(X[:, :, 3], vB) * f(X[:, :, 4], vC) * f(X[:, :, 5], vC)
            out.append(J)
    return np.stack(out, 2)


def bayes_trans(X, beta=0.9):
    J = trans_joint(X, beta); M = J.sum(-1); lM = np.log(M)
    res = {}
    loo = np.stack([lM[:, [k for k in range(Q) if k != q]].sum(1) for q in range(Q)], 1)       # N,Q,36
    w = np.exp(loo - loo.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    py = (w[..., None] * J).sum(2); res["post_y"] = py / py.sum(-1, keepdims=True)
    own = J.sum(2); res["own_y"] = own / own.sum(-1, keepdims=True)
    # one-hop-only reference: knows sigma posterior from A-B pairs but ignores composite tau (treats C as noise)
    return res


def bayes_trans_onehop(X, beta=0.9):
    """No-composition reference: infers each map only from DIRECT co-occurrence with A (sigma from AB questions; the
    composite tau*sigma would need A-C co-occurrence, which never happens).  Implemented by removing the BC questions'
    C reports (so tau is never learned): C-only questions are then undecodable beyond the prior."""
    Xn = X.copy(); bc = (X[:, :, 0] == MASK) & (X[:, :, 2] != MASK)
    Xn[:, :, 4] = np.where(bc, MASK, X[:, :, 4]); Xn[:, :, 5] = np.where(bc, MASK, X[:, :, 5])
    return bayes_trans(Xn, beta)
