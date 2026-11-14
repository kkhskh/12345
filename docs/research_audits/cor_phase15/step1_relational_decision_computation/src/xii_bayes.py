"""Exact posterior for the ACTUAL genlib sampler with C >= 1 colluders.
genlib.sample draws ONE copied value per question and gives it to every colluding source (col is shared across colluder groups).
genlib.posterior_brute instead multiplies per-group colluder factors  beta*eq_g/K + (1-beta)/K^2,  i.e. it is the posterior of an
'independent copying' model.  The two coincide for C = 1 and differ for C >= 2.
Shared-copy likelihood of the colluder set H at one question:  beta * 1[all 2|H| sources equal] / K  +  (1 - beta) / K^(2|H|)."""
import numpy as np
import genlib as GL


def posterior_shared(env, X, q):
    n = len(X); idx = np.arange(n); K = env.K
    rel, _, noi = GL._group_lik(env, X); relm = rel.mean(-1)
    a_ = X.reshape(*X.shape[:2], env.G, 2)
    numer = []
    for g, hs in env.roles:
        hs = list(hs); others = [h for h in range(env.G) if h != g and h not in hs]
        srcs = a_[:, :, hs, :].reshape(n, X.shape[1], -1)                                   # (n,Q,2|H|)
        alleq = (srcs == srcs[..., :1]).all(-1)
        colH = env.beta * alleq / K + (1 - env.beta) / K ** (2 * len(hs))
        lq = np.log(relm[:, :, g]) + np.log(colH) + len(others) * np.log(noi)
        mask = np.ones_like(lq, bool); mask[idx, q] = False; tot = np.where(mask, lq, 0.0).sum(1)
        lyq = np.log(rel[idx, q, g, :]) + np.log(colH[idx, q])[:, None] + len(others) * np.log(noi)
        numer.append(tot[:, None] + lyq)
    L = np.stack(numer); m = L.max((0, 2), keepdims=True); P = np.exp(L - m).sum(0)
    return P / P.sum(-1, keepdims=True)
