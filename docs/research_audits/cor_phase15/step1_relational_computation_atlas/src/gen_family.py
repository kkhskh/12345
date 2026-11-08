"""Generalised latent-reliability generator family and exact Bayes (atlas, Direction C/D/E).

Episode: Q questions, K candidates, 6 sources in 3 pairs G0={0,1}, G1={2,3}, G2={4,5} (before an optional per-episode
source permutation that hides the pairing).  A configuration R = (g, h) says group g is reliable and group h (or None)
is a colluding/copying group; remaining groups are noise.  Prior: with prob pc an episode has a colluding group
(uniform over the 6 ordered (g,h)); otherwise uniform over the 3 configs (g, None).
Roles (per question, per pair (xa, xb), given Y=y):
  rel   : xa, xb independent, each = y w.p. alpha, else uniform over the K-1 others
  noise : xa, xb independent uniform
  coll  : w.p. rho both equal a common c ~ U(K) (independent of y); else independent uniform
The original package's T1 is (pc=0), T4 is (pc=1, rho=1).
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np

GROUPS = [(0, 1), (2, 3), (4, 5)]


@dataclass(frozen=True)
class Fam:
    K: int = 3
    Q: int = 6
    alpha: float = 0.8
    rho: float = 1.0
    pc: float = 1.0
    perm: bool = False           # hide the pairing by a random per-episode source permutation

    @property
    def configs(self):
        c = []
        if self.pc < 1:
            c += [(g, None) for g in range(3)]
        if self.pc > 0:
            c += [(g, h) for g in range(3) for h in range(3) if g != h]
        return c

    @property
    def prior(self):
        return np.array([(1 - self.pc) / 3 if h is None else self.pc / 6 for g, h in self.configs])


def pair_lik(f: Fam, xa, xb, role):
    """P(xa, xb | y, role) for every y: returns (..., K)."""
    K = f.K; e = (1 - f.alpha) / (K - 1); ys = np.arange(K)
    if role == "rel":
        pa = np.where(xa[..., None] == ys, f.alpha, e); pb = np.where(xb[..., None] == ys, f.alpha, e); return pa * pb
    if role == "noise":
        return np.full(xa.shape + (K,), 1.0 / K ** 2)
    if role == "coll":
        v = f.rho / K * (xa == xb) + (1 - f.rho) / K ** 2
        return np.broadcast_to(v[..., None], xa.shape + (K,))
    raise ValueError(role)


def roles_of(cfg):
    g, h = cfg
    return ["rel" if k == g else ("coll" if k == h else "noise") for k in range(3)]


def joint(f: Fam, X, pairing=GROUPS):
    """X (N,Q,6) -> J (N,Q,C,K) = P(X_q, Y_q=y | config) under the given pairing of source positions."""
    out = []
    for cfg in f.configs:
        J = np.full(X.shape[:2] + (f.K,), 1.0 / f.K)
        for (a, b), r in zip(pairing, roles_of(cfg)):
            J = J * pair_lik(f, X[..., a], X[..., b], r)
        out.append(J)
    return np.stack(out, 2)


def posterior(f: Fam, X, pairing=GROUPS, prior=None):
    """Exact posterior with KNOWN pairing. Returns dict(post_y, post_cfg, pi_minus, own_y, logM)."""
    pr = f.prior if prior is None else prior
    J = joint(f, X, pairing); M = J.sum(-1)                               # N,Q,C
    with np.errstate(divide="ignore"):
        lM = np.log(M)
    tot = lM.sum(1) + np.log(np.maximum(pr, 1e-300))                      # N,C
    def norm(l):
        m = np.max(l, -1, keepdims=True); m = np.where(np.isfinite(m), m, 0); w = np.exp(l - m); s = w.sum(-1, keepdims=True)
        return w / np.where(s > 0, s, np.nan)
    post_c = norm(tot)
    Qn = X.shape[1]                                                       # leave-one-out sums computed directly (no inf - inf)
    loo = np.stack([lM[:, [k for k in range(Qn) if k != q]].sum(1) for q in range(Qn)], 1) + np.log(np.maximum(pr, 1e-300))
    pim = norm(loo)
    py = (pim[..., None] * J).sum(2); py = py / py.sum(-1, keepdims=True)
    own = (pr[None, None, :, None] * J).sum(2); own = own / own.sum(-1, keepdims=True)
    return dict(post_y=py, post_cfg=post_c, pi_minus=pim, own_y=own, logM=lM)


ALL_PAIRINGS = []
def _pairings(items):
    if not items:
        yield []; return
    a = items[0]
    for i in range(1, len(items)):
        b = items[i]; rest = items[1:i] + items[i + 1:]
        for p in _pairings(rest):
            yield [(a, b)] + p
ALL_PAIRINGS = [tuple(p) for p in _pairings(list(range(6)))]          # 15 pairings


def posterior_unknown(f: Fam, X):
    """Exact posterior when the pairing is unknown (uniform over the 15 pairings, and over labelled role assignment)."""
    Js, pri = [], []
    for P in ALL_PAIRINGS:
        Js.append(joint(f, X, list(P))); pri.append(f.prior / len(ALL_PAIRINGS))
    J = np.concatenate(Js, 2); pr = np.concatenate(pri)                  # N,Q,15C,K
    M = J.sum(-1)
    with np.errstate(divide="ignore"):
        lM = np.log(M)
    tot = lM.sum(1) + np.log(pr)
    Qn = X.shape[1]
    loo = np.stack([lM[:, [k for k in range(Qn) if k != q]].sum(1) for q in range(Qn)], 1) + np.log(pr)
    m = loo.max(-1, keepdims=True); w = np.exp(loo - m); w = w / w.sum(-1, keepdims=True)
    py = (w[..., None] * J).sum(2); py = py / py.sum(-1, keepdims=True)
    m2 = tot.max(-1, keepdims=True); pc = np.exp(tot - m2); pc /= pc.sum(-1, keepdims=True)
    pairing_post = pc.reshape(len(X), len(ALL_PAIRINGS), -1).sum(-1)
    return dict(post_y=py, pairing_post=pairing_post)


def sample(f: Fam, n, rng, cfg_idx=None):
    K, Q = f.K, f.Q; cf = f.configs
    ci = rng.choice(len(cf), n, p=f.prior) if cfg_idx is None else np.asarray(cfg_idx)
    Y = rng.integers(0, K, (n, Q)); X = np.zeros((n, Q, 6), int)
    for i, cfg in enumerate(cf):
        sel = ci == i
        if not sel.any():
            continue
        m = sel.sum()
        for (a, b), r in zip(GROUPS, roles_of(cfg)):
            if r == "rel":
                for s in (a, b):
                    ok = rng.random((m, Q)) < f.alpha; wrong = (Y[sel] + rng.integers(1, K, (m, Q))) % K
                    X[sel, :, s] = np.where(ok, Y[sel], wrong)
            elif r == "noise":
                X[sel, :, a] = rng.integers(0, K, (m, Q)); X[sel, :, b] = rng.integers(0, K, (m, Q))
            else:
                cop = rng.random((m, Q)) < f.rho; c = rng.integers(0, K, (m, Q))
                X[sel, :, a] = np.where(cop, c, rng.integers(0, K, (m, Q))); X[sel, :, b] = np.where(cop, c, rng.integers(0, K, (m, Q)))
    perms = None
    if f.perm:
        perms = np.argsort(rng.random((n, 6)), -1)
        X = np.take_along_axis(X, perms[:, None, :], -1)
    return dict(X=X, Y=Y, cfg=ci, perms=perms)


def agree_probs(f: Fam):
    """Per-question within-pair agreement probability for each role."""
    e = (1 - f.alpha) / (f.K - 1)
    return dict(rel=f.alpha ** 2 + (f.K - 1) * e ** 2, noise=1.0 / f.K, coll=f.rho + (1 - f.rho) / f.K)
