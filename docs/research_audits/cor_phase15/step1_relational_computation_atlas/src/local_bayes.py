"""Factorised ('local') reliability reference, used to test the post hoc hypothesis that came out of PP2.

The exact Bayes posterior couples the groups through the configuration constraint: exactly one group is reliable and
(in T4) exactly one colludes. The *local* reference drops that coupling. Each group's role posterior is computed from
its own leave-one-out agreement count alone, with a fixed per-group role prior. The candidate posterior is then a
product over groups of role-mixtures:
    P_loc(y | X) ∝ Π_g Σ_r w_g(r) f_r(x_{q,g} | y),    w_g(r) ∝ π(r) · a_r^{n_g} (1 - a_r)^{M - n_g}
where a_r is the role's per-question agreement probability (training parameters) and M = Q - 1.
"""
import numpy as np
import gen_family as G


def local_post(train: G.Fam, X, role_prior=None):
    roles = ["rel", "coll", "noise"] if train.pc > 0 else ["rel", "noise"]
    if role_prior is None:
        role_prior = {"rel": 1 / 3, "coll": 1 / 3, "noise": 1 / 3} if train.pc > 0 else {"rel": 1 / 3, "noise": 2 / 3}
    ap = G.agree_probs(train); Q = X.shape[1]; M = Q - 1
    A = np.stack([(X[:, :, a] == X[:, :, b]) for a, b in G.GROUPS], -1).astype(int)       # N,Q,3
    loo = A.sum(1, keepdims=True) - A                                                      # N,Q,3
    out = np.ones(X.shape[:2] + (train.K,))
    for g, (a, b) in enumerate(G.GROUPS):
        n = loo[:, :, g]
        lw = {r: np.log(role_prior[r]) + n * np.log(max(ap[r], 1e-12)) + (M - n) * np.log(max(1 - ap[r], 1e-12)) for r in roles}
        m = np.max(np.stack(list(lw.values())), 0)
        w = {r: np.exp(lw[r] - m) for r in roles}; s = sum(w.values())
        mix = sum((w[r] / s)[..., None] * G.pair_lik(train, X[..., a], X[..., b], r) for r in roles)
        out = out * mix
    return out / out.sum(-1, keepdims=True)
