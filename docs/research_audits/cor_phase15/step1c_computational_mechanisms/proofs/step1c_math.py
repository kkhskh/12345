"""Synthetic definitions for Step 1C.

Every object here is a finite map used to check a stated identity or a
counterexample. Nothing in this module is a pretrained Transformer, and a
passing check is not evidence that a trained model implements the map.
"""

from __future__ import annotations

import numpy as np


def four_term(values):
    """values is the tuple (F(u,v), F(u,0), F(0,v), F(0,0))."""
    f_uv, f_u0, f_0v, f_00 = values
    return f_uv - f_u0 - f_0v + f_00


def interaction_at_zero(F, u, v):
    """Four-term interaction of F at baseline (0, 0)."""
    return four_term((F(u, v), F(u, 0.0), F(0.0, v), F(0.0, 0.0)))


def mixed_partial_at_zero(F, step=1e-4):
    """Central mixed difference of F at (0, 0)."""
    s = step
    return (
        F(s, s) - F(s, -s) - F(-s, s) + F(-s, -s)
    ) / (4.0 * s * s)


def obstruction_O(ds, W):
    """O = (ds)^T W (ds). M is not an argument."""
    ds = np.atleast_1d(np.asarray(ds, dtype=float))
    W = np.asarray(W, dtype=float)
    return float(ds @ W @ ds)


def distance_D(s, basis, M):
    """Minimise ||s - B c||_M^2 over c.

    ``basis`` has columns spanning ker d. M must be symmetric positive
    definite on that subspace. This is the definition of D, not O.
    """
    s = np.asarray(s, dtype=float)
    B = np.asarray(basis, dtype=float)
    M = np.asarray(M, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    BtM = B.T @ M
    gram = BtM @ B
    coeff = np.linalg.solve(gram, BtM @ s)
    gap = s - B @ coeff
    return float(gap @ M @ gap)


def layer_norm(h, eps=0.0, gamma=None, beta=None):
    """Population LayerNorm. Mean subtraction, then rms of the centered vector.

    eps=0 is the exact map used in the sign counterexample. It is undefined
    on constant vectors. gamma and beta are the learned affine, applied after
    normalization. This is not a free change of coordinates.
    """
    h = np.asarray(h, dtype=float)
    centered = h - h.mean()
    scale = np.sqrt(np.mean(centered * centered) + eps)
    z = centered / scale
    if gamma is not None:
        z = np.asarray(gamma, dtype=float) * z
    if beta is not None:
        z = z + np.asarray(beta, dtype=float)
    return z


def rms_norm(h, eps=0.0, gamma=None):
    """Root-mean-square normalization. No mean subtraction."""
    h = np.asarray(h, dtype=float)
    scale = np.sqrt(np.mean(h * h) + eps)
    z = h / scale
    if gamma is not None:
        z = np.asarray(gamma, dtype=float) * z
    return z


def layer_norm_jacobian(h, eps=0.0, step=1e-6):
    """Finite-difference Jacobian of plain LayerNorm at h."""
    h = np.asarray(h, dtype=float)
    d = h.shape[0]
    columns = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = step
        columns.append((layer_norm(h + e, eps=eps) - layer_norm(h - e, eps=eps)) / (2 * step))
    return np.column_stack(columns)


def softmax_rows(scores):
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    safe = np.where(finite, scores, -np.inf)
    row_max = np.max(np.where(finite, safe, -np.inf), axis=-1, keepdims=True)
    shifted = np.where(finite, safe - row_max, -np.inf)
    exp = np.exp(shifted)
    exp = np.where(finite, exp, 0.0)
    return exp / exp.sum(axis=-1, keepdims=True)


def attention(X, W_Q, W_K, W_V, W_O, causal=True):
    """Standard attention.

    Q = X W_Q, K = X W_K, V = X W_V,
    scores = Q K^T / sqrt(d_k) plus an optional causal mask,
    Y = softmax(scores) V W_O.

    Returns Y, the attention matrix, and V. Intervening on X moves Q, K,
    and V together. That is a different intervention from editing V while
    holding the attention matrix fixed.
    """
    X = np.asarray(X, dtype=float)
    Q = X @ np.asarray(W_Q, dtype=float)
    K = X @ np.asarray(W_K, dtype=float)
    V = X @ np.asarray(W_V, dtype=float)
    dk = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(dk)
    if causal:
        n = scores.shape[0]
        scores = scores.copy()
        scores[np.triu(np.ones((n, n), dtype=bool), k=1)] = -np.inf
    A = softmax_rows(scores)
    Y = (A @ V) @ np.asarray(W_O, dtype=float)
    return Y, A, V


def mix_values(A, V, W_O):
    """Attention output with the pattern held fixed. Linear in V."""
    return (np.asarray(A, dtype=float) @ np.asarray(V, dtype=float)) @ np.asarray(W_O, dtype=float)


def relu(z):
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0)


def mlp(h, W1, b1, W2, b2):
    """W2 relu(W1 h + b1) + b2."""
    h = np.asarray(h, dtype=float)
    hidden = relu(np.asarray(W1, dtype=float) @ h + np.asarray(b1, dtype=float))
    return np.asarray(W2, dtype=float) @ hidden + np.asarray(b2, dtype=float)


def linear_residual(h, P_u, P_v, A, rho):
    """r(h) = P_v A h - rho P_u h.

    The measurement operator is P_v A - rho P_u. A vector in ker P_u is in
    that kernel only if P_v A also kills it.
    """
    h = np.asarray(h, dtype=float)
    return (
        np.asarray(P_v, dtype=float) @ np.asarray(A, dtype=float) @ h
        - np.asarray(rho, dtype=float) @ np.asarray(P_u, dtype=float) @ h
    )
