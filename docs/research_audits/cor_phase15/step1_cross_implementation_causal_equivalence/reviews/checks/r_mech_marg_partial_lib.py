"""Shared helper for reviewer checks: count encoder fitted jointly with own-question covariates (count block returned).
Identical to fit_partial in r_mech_marg_partial.py."""
import numpy as np
import eqv as E


def cov_feats(Xs, k):            # own-question covariates of question k: one-hot of its 6 reports + 3 agreements
    return np.c_[np.eye(3)[Xs[:, k]].reshape(len(Xs), -1), E.agree(Xs)[:, k]]


def fit_partial(kind, Xs, A):
    Fd = E.site_features(kind, Xs); W = {}
    for t, F in Fd.items():
        k = t - 36 if kind == "P_full" else t // 6
        Z = np.c_[np.ones(len(Xs)), F, cov_feats(Xs, k)]
        W[t] = np.linalg.lstsq(Z, A[:, t], rcond=None)[0][1:1 + F.shape[1]]
    return W
