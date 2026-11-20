"""Exact generators and Bayes references (mandate §1-§3). Pure numpy, enumeration over the latent R.

Single-question environments
  E0  independent sources, known reliabilities q_i: log posterior odds are additive in source contributions.
  E1  hidden shared reliability: sources in groups; latent R says which group is reliable (others uniform).

Episode environment (multi-question, shared R) used for training tasks T0-T5
  K candidates (3; extension 5); Q questions per episode; S = 6 sources in 3 groups {0,1},{2,3},{4,5}.
  Y_q ~ Uniform(K) independently per question.
  reliable source : reports Y_q w.p. alpha, otherwise uniform over the other K-1 candidates
  noise source    : uniform over K
  colluding group (T4 only): both sources report the same c_q ~ Uniform(K), independent of Y_q
  T0  known reliability: group 0 always reliable (R fixed, not latent)              -> additive per-source evidence
  T1  R ~ Uniform{0,1,2} = index of the reliable group; other groups noise; shared by all Q questions
  T2  = T1 generator; outputs: per-question candidate distribution + per-question confidence (Bayes max posterior)
  T3  = T1 generator + auxiliary supervision of R (positive control)
  T4  R = (g, h), g != h uniform: group g reliable, group h colluding, third group noise
  T5  aggregate-sufficient control: all 6 sources reliable with alpha5, no latent -> per-question vote histogram sufficient
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

GROUPS = [(0, 1), (2, 3), (4, 5)]


@dataclass(frozen=True)
class Env:
    task: str = "T1"
    K: int = 3
    Q: int = 4
    alpha: float = 0.8
    alpha5: float = 0.5

    @property
    def R_values(self):
        if self.task in ("T1", "T2", "T3"):
            return [(g, None) for g in range(3)]
        if self.task == "T4":
            return [(g, h) for g in range(3) for h in range(3) if g != h]
        return [None]          # T0, T5: no latent


def p_report(env: Env, role: str, x, y):
    """P(source report x | Y = y, role). x, y int arrays broadcastable."""
    K = env.K
    if role == "rel":
        return np.where(x == y, env.alpha, (1 - env.alpha) / (K - 1))
    if role == "rel5":
        return np.where(x == y, env.alpha5, (1 - env.alpha5) / (K - 1))
    if role == "noise":
        return np.full(np.broadcast(x, y).shape, 1.0 / K)
    raise ValueError(role)


def roles(env: Env, R):
    """Role of each of the 6 sources under latent R."""
    if env.task == "T0":
        return ["rel", "rel", "noise", "noise", "noise", "noise"]
    if env.task == "T5":
        return ["rel5"] * 6
    g, h = R
    out = []
    for s in range(6):
        gi = s // 2
        out.append("rel" if gi == g else ("collude" if gi == h else "noise"))
    return out


def q_lik(env: Env, X, R):
    """P(X_q | Y_q = y, R) for every y: X (N, 6) int -> (N, K). Colluding pair handled jointly."""
    K = env.K
    N = X.shape[0]
    ys = np.arange(K)[None, :]
    L = np.ones((N, K))
    rl = roles(env, R)
    done = set()
    for s in range(6):
        if s in done:
            continue
        if rl[s] == "collude":
            a, b = GROUPS[s // 2]
            L *= np.where(X[:, a] == X[:, b], 1.0 / K, 0.0)[:, None]
            done |= {a, b}
        else:
            L *= p_report(env, rl[s], X[:, s:s + 1], ys)
            done.add(s)
    return L


def episode_posterior(env: Env, X):
    """X: (N, Q, 6). Returns dict with
       post_y  (N, Q, K)  exact P(Y_q | whole episode)
       post_R  (N, nR)    exact P(R | whole episode)
       pi_minus(N, Q, nR) P(R | all questions except q)        -- the reusable cross-question reliability summary
       own_y   (N, Q, K)  P(Y_q | X_q only)                     -- no cross-question information
    """
    K, Q = env.K, env.Q
    N = X.shape[0]
    Rs = env.R_values
    nR = len(Rs)
    prior_y = 1.0 / K
    # per question, per R: joint P(X_q, Y_q=y | R) and marginal P(X_q | R)
    J = np.zeros((N, Q, nR, K))
    for r, R in enumerate(Rs):
        for q in range(Q):
            J[:, q, r, :] = prior_y * q_lik(env, X[:, q, :], R)
    M = J.sum(-1)                                   # (N, Q, nR)
    logM = np.log(np.maximum(M, 1e-300))
    tot = logM.sum(1)                               # (N, nR)
    pR = np.exp(tot - tot.max(1, keepdims=True)); pR /= pR.sum(1, keepdims=True)
    lm = tot[:, None, :] - logM                     # log prod over q' != q
    pim = np.exp(lm - lm.max(2, keepdims=True)); pim /= pim.sum(2, keepdims=True)
    post_y = np.einsum("nqr,nqrk->nqk", pim, J); post_y /= post_y.sum(-1, keepdims=True)
    own = J.sum(2) / nR; own /= own.sum(-1, keepdims=True)
    return dict(post_y=post_y, post_R=pR, pi_minus=pim, own_y=own, M=M)


def sample_episodes(env: Env, n, rng):
    K, Q = env.K, env.Q
    Rs = env.R_values
    ridx = rng.integers(0, len(Rs), n)
    Y = rng.integers(0, K, (n, Q))
    X = np.zeros((n, Q, 6), int)
    for i in range(n):
        R = Rs[ridx[i]]
        rl = roles(env, R)
        for q in range(Q):
            y = Y[i, q]
            done = set()
            for s in range(6):
                if s in done:
                    continue
                if rl[s] == "collude":
                    a, b = GROUPS[s // 2]; c = rng.integers(0, K); X[i, q, a] = X[i, q, b] = c; done |= {a, b}
                elif rl[s] in ("rel", "rel5"):
                    al = env.alpha if rl[s] == "rel" else env.alpha5
                    X[i, q, s] = y if rng.random() < al else rng.choice([k for k in range(K) if k != y]); done.add(s)
                else:
                    X[i, q, s] = rng.integers(0, K); done.add(s)
    return dict(X=X, Y=Y, ridx=ridx)


def sample_fast(env: Env, n, rng, ridx=None):
    """Vectorised sampler (same distribution as sample_episodes). If ridx is given, R is fixed per episode."""
    K, Q = env.K, env.Q
    Rs = env.R_values
    ridx = rng.integers(0, len(Rs), n) if ridx is None else np.asarray(ridx)
    Y = rng.integers(0, K, (n, Q))
    X = rng.integers(0, K, (n, Q, 6))                       # noise default
    off = rng.integers(1, K, (n, Q, 6))                     # wrong-answer offsets
    u = rng.random((n, Q, 6))
    col = rng.integers(0, K, (n, Q))
    for s in range(6):
        gi = s // 2
        if env.task == "T0":
            rel = np.full(n, gi == 0); coll = np.zeros(n, bool); al = env.alpha
        elif env.task == "T5":
            rel = np.ones(n, bool); coll = np.zeros(n, bool); al = env.alpha5
        else:
            g = np.array([Rs[r][0] for r in ridx]); h = np.array([-1 if Rs[r][1] is None else Rs[r][1] for r in ridx])
            rel = g == gi; coll = h == gi; al = env.alpha
        rep = np.where(u[:, :, s] < al, Y, (Y + off[:, :, s]) % K)
        X[:, :, s] = np.where(rel[:, None], rep, X[:, :, s])
        X[:, :, s] = np.where(coll[:, None], col, X[:, :, s])
    return dict(X=X, Y=Y, ridx=ridx)


# ----------------------------------------------------------------------------- §1 single-question checks


def e1_example(prior=(0.025, 0.025, 0.95), acc=0.88):
    """Mandate's 4-source example: groups A=(s0,s1), B=(s2,s3); R in {A reliable, B reliable} with prob 1/2;
    reliable source reports Y w.p. acc else uniform over the other two; unreliable reports uniformly."""
    K = 3
    pr = np.array(prior)
    def rel(x, y):
        return acc if x == y else (1 - acc) / (K - 1)
    def post(A, B):
        out = np.zeros(K)
        for y in range(K):
            la = rel(A[0], y) * rel(A[1], y) * (1 / K) ** 2      # R = A reliable
            lb = (1 / K) ** 2 * rel(B[0], y) * rel(B[1], y)      # R = B reliable
            out[y] = pr[y] * 0.5 * (la + lb)
        return out / out.sum()
    return post((0, 0), (1, 1)), post((0, 1), (0, 1))


def e0_additivity_check(rng, n=2000, S=6, K=2):
    """E0: binary Y, independent sources with known q_i: log odds = log prior odds + sum_i w_i (2 x_i - 1)."""
    qs = rng.uniform(0.55, 0.95, S)
    X = rng.integers(0, 2, (n, S))
    lp1 = np.sum(np.log(np.where(X == 1, qs, 1 - qs)), 1)
    lp0 = np.sum(np.log(np.where(X == 0, qs, 1 - qs)), 1)
    exact = lp1 - lp0
    additive = np.sum(np.log(qs / (1 - qs)) * (2 * X - 1), 1)
    return float(np.abs(exact - additive).max())
