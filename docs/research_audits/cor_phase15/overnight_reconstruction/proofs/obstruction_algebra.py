"""The O/D algebra, kept only so the preserved negative results stay checked.

Nothing here is a causal bridge. O and D are quadratic forms of one coboundary.
"""

from __future__ import annotations

import numpy as np


def two_node_operator(rho):
    """d on (s_u, s_v): block row [rho | -I]."""
    rho = np.asarray(rho, dtype=float)
    r = rho.shape[0]
    d = np.zeros((r, 2 * r))
    d[:, :r] = rho
    d[:, r:] = -np.eye(r)
    return d


def obstruction_pair(s, rho, M, W):
    s = np.asarray(s, dtype=float).reshape(-1)
    d = two_node_operator(rho)
    ds = d @ s
    O = float(ds @ W @ ds)
    gram = d @ np.linalg.inv(M) @ d.T
    D = float(ds @ np.linalg.pinv(gram) @ ds)
    return O, D, ds


def spectrum_bounds(s, rho, M, W):
    """lambda_min^+ D <= O <= lambda_max D on the whole space, via the two quadratics.

    Both forms are ds^T A ds. The comparison used here is the generalized
    eigenvalue relation on im(d): O = ds^T W ds and D = ds^T G^+ ds.
    """
    O, D, ds = obstruction_pair(s, rho, M, W)
    return O, D
