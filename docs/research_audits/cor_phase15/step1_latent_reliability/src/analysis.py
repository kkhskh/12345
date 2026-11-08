"""Identification (§7) and causal tests (§8) for the episode Transformer.

Nothing here reads model outputs to *define* a mediator: subspaces are fitted to Bayes quantities (D1) or to
input-level contrast pairs (D2). Model outputs are used only to *evaluate* interventions.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import bayes as B
import nets as N

CH = 1000  # chunk size


# ----------------------------------------------------------------------------- activations
def sites(p):
    L = len(p["layers"])
    return [f"resid_pre{l}" for l in range(L)] + [f"resid_post{L - 1}"]


@partial(jax.jit, static_argnames=("names",))
def _acts(p, X, names):
    rec = {}

    def hk(n, a):
        if n in names:
            rec[n] = a
        return a
    out = N.forward(p, X, hk)
    return rec, out


def acts(p, X, names):
    names = tuple(names)
    recs, outs = [], []
    for i in range(0, len(X), CH):
        r, o = _acts(p, jnp.asarray(X[i:i + CH]), names)
        recs.append({k: np.asarray(v) for k, v in r.items()}); outs.append({k: np.asarray(v) for k, v in o.items()})
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]}
    return cat(recs), cat(outs)


def probs(o):
    a = o["answer"]; a = a - a.max(-1, keepdims=True); e = np.exp(a); return e / e.sum(-1, keepdims=True)


@partial(jax.jit, static_argnames=("site",))
def _patch(p, X, site, mask, P, src):
    """At `site`, replace the component in span(P) at positions mask (T,) by that of src (N,T,d)."""
    def hk(n, a):
        if n == site:
            delta = (src - a) @ P
            return a + delta * mask[None, :, None]
        return a
    return N.forward(p, X, hk)


@partial(jax.jit, static_argnames=("site",))
def _patch_pattern(p, X, site, qmask, hmask, src):
    """At attention-pattern site (N,H,T,T): rows qmask (T,), heads hmask (H,) take src's pattern."""
    def hk(n, a):
        if n == site:
            m = hmask[None, :, None, None] * qmask[None, None, :, None]
            return a * (1 - m) + src * m
        return a
    return N.forward(p, X, hk)


def patch(p, X, site, mask, P, src):
    outs = []
    for i in range(0, len(X), CH):
        o = _patch(p, jnp.asarray(X[i:i + CH]), site, jnp.asarray(mask, jnp.float32), jnp.asarray(P, jnp.float32),
                   jnp.asarray(src[i:i + CH]))
        outs.append(np.asarray(jax.nn.softmax(o["answer"], -1)))
    return np.concatenate(outs)


def patch_full(p, X, site, mask, P, src):
    """Like patch but also returns confidence."""
    A, C = [], []
    for i in range(0, len(X), CH):
        o = _patch(p, jnp.asarray(X[i:i + CH]), site, jnp.asarray(mask, jnp.float32), jnp.asarray(P, jnp.float32),
                   jnp.asarray(src[i:i + CH]))
        A.append(np.asarray(jax.nn.softmax(o["answer"], -1))); C.append(np.asarray(o["conf"]))
    return np.concatenate(A), np.concatenate(C)


def patch_pattern(p, X, site, qmask, hmask, src):
    outs = []
    for i in range(0, len(X), CH):
        o = _patch_pattern(p, jnp.asarray(X[i:i + CH]), site, jnp.asarray(qmask, jnp.float32),
                           jnp.asarray(hmask, jnp.float32), jnp.asarray(src[i:i + CH]))
        outs.append(np.asarray(jax.nn.softmax(o["answer"], -1)))
    return np.concatenate(outs)


def rd_pos(env, q):
    return 6 * env.Q + q


def ep_pos(env):
    return 7 * env.Q


def T_of(env):
    return 7 * env.Q + 1


# ----------------------------------------------------------------------------- pair constructions
def own_R_post(env, Xq):
    """P(R | X_q) for a single question's reports (uniform R prior). Xq: (N,6)."""
    M = np.stack([B.q_lik(env, Xq, R).sum(-1) for R in env.R_values], -1)
    return M / M.sum(-1, keepdims=True)


def pairs(env, X, q, kind, rng):
    """Return X' (same shape as X). Kinds:
       cond   : X'_q = X_q; X'_{-q} freshly sampled from P(X_{-q} | X_q) (sample R' ~ P(R|X_q), then questions). On-manifold.
       perm   : X'_q = X_q; each other question's six reports permuted across sources (per-question histograms preserved).
       splice : X'_q = X_q; X'_{-q} from an independent natural episode (off-manifold for T4).
       target : X'_{-q} = X_{-q}; X'_q = (X_q + shift) mod K, shift in 1..K-1 (agreement pattern, hence all counts, preserved)."""
    n = len(X); Xp = X.copy()
    if kind == "cond":
        pr = own_R_post(env, X[:, q]); cum = pr.cumsum(-1)
        r = (rng.random((n, 1)) > cum).sum(-1)
        fresh = B.sample_fast(env, n, rng, ridx=r)["X"]
        oth = [k for k in range(env.Q) if k != q]; Xp[:, oth] = fresh[:, oth]
    elif kind == "perm":
        for k in range(env.Q):
            if k == q:
                continue
            perm = np.argsort(rng.random((n, 6)), -1)
            Xp[:, k] = np.take_along_axis(X[:, k], perm, -1)
    elif kind == "splice":
        fresh = B.sample_fast(env, n, rng)["X"]
        oth = [k for k in range(env.Q) if k != q]; Xp[:, oth] = fresh[:, oth]
    elif kind == "target":
        sh = rng.integers(1, env.K, (n, 1)); Xp[:, q] = (X[:, q] + sh) % env.K
    else:
        raise ValueError(kind)
    return Xp


def log_lik(env, X):
    """log P(X) under the generator (uniform R prior), for on-/off-manifold classification."""
    post = B.episode_posterior(env, X)
    lm = np.log(np.maximum(post["M"], 1e-300)).sum(1)          # (N, nR)
    mx = lm.max(1)
    return mx + np.log(np.exp(lm - mx[:, None]).mean(1))


# ----------------------------------------------------------------------------- Bayes targets for probes
def logit_target(pi, eps=1e-6):
    l = np.log(np.maximum(pi, eps)); return l - l.mean(-1, keepdims=True)


def hist_feats(env, X, q):
    """Aggregate surrogate: per-question global vote histograms of the other questions (no source identity)."""
    oth = [k for k in range(env.Q) if k != q]
    h = np.stack([(X[:, oth] == k).sum(-1) for k in range(env.K)], -1)      # N, Q-1, K
    return h.reshape(len(X), -1).astype(float)


def count_feats(env, X, q):
    """Relational statistic: leave-one-out per-group within-pair agreement counts."""
    A = np.stack([X[:, :, a] == X[:, :, b] for a, b in B.GROUPS], -1).astype(float)   # N,Q,3
    return A.sum(1) - A[:, q]


# ----------------------------------------------------------------------------- linear probes
def ridge(Z, T, lam=1e-2):
    mu, tm = Z.mean(0), T.mean(0)
    Zc, Tc = Z - mu, T - tm
    W = np.linalg.solve(Zc.T @ Zc + lam * len(Z) * np.eye(Z.shape[1]), Zc.T @ Tc)
    return W, mu, tm


def r2(Z, T, W, mu, tm):
    P = (Z - mu) @ W + tm
    return float(1 - ((T - P) ** 2).sum() / max(((T - T.mean(0)) ** 2).sum(), 1e-12))


def basis_from_W(W, k):
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, :k]


def proj(U):
    return U @ U.T


def random_basis(d, k, rng):
    Qm, _ = np.linalg.qr(rng.normal(size=(d, k))); return Qm


def contrast_basis(D, k):
    D = D - D.mean(0)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:k].T


# ----------------------------------------------------------------------------- metrics
def kl(p, q):
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)


def interchange_metrics(pE, pEp, pPatch):
    """pE: model(base) at q; pEp: model(source); pPatch: model(base with patch from source)."""
    inf = pE.argmax(-1) != pEp.argmax(-1)
    iia = float((pPatch.argmax(-1) == pEp.argmax(-1))[inf].mean()) if inf.any() else float("nan")
    base_kl = kl(pEp, pE); rec = 1 - kl(pEp, pPatch) / np.maximum(base_kl, 1e-9)
    w = base_kl > 1e-3
    return dict(iia=iia, n_inf=int(inf.sum()), kl_recovered=float(np.median(rec[w])) if w.any() else float("nan"),
                kl_recovered_mean=float(np.clip(rec[w], -2, 2).mean()) if w.any() else float("nan"),
                change_rate=float((pPatch.argmax(-1) != pE.argmax(-1)).mean()))
