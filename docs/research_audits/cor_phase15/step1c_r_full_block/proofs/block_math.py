"""One pre-norm block and the LayerNorm facts needed to correct Step 1C.

Nothing here is a pretrained model. Weights are fixed numbers. A finite-difference
match is a check of this implementation, not a theorem.
"""

from __future__ import annotations

import numpy as np


def centering_matrix(d):
    return np.eye(d) - np.ones((d, d)) / d


def layer_norm(h, gamma, beta, eps):
    """Per-vector LayerNorm. h has shape (..., d). eps is the variance offset."""
    h = np.asarray(h, dtype=float)
    z = h - h.mean(axis=-1, keepdims=True)
    v = np.mean(z * z, axis=-1, keepdims=True)
    out = z / np.sqrt(v + eps)
    out = np.asarray(gamma, dtype=float) * out + np.asarray(beta, dtype=float)
    return out


def layer_norm_jacobian(h, gamma, eps):
    """Analytic Jacobian of LN_eps at one vector.

    J = diag(gamma) @ (C / sqrt(v+eps) - z z^T / (d (v+eps)^{3/2})),
    where z = C h and v = z·z / d.
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    d = h.shape[0]
    C = centering_matrix(d)
    z = C @ h
    v = float(z @ z / d)
    s = np.sqrt(v + eps)
    bracket = C / s - np.outer(z, z) / (d * s**3)
    return np.diag(gamma) @ bracket


def softmax_rows(scores):
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    safe = np.where(finite, scores, -np.inf)
    row_max = np.max(np.where(finite, safe, -np.inf), axis=-1, keepdims=True)
    shifted = np.where(finite, safe - row_max, -np.inf)
    exp = np.exp(np.where(finite, shifted, -np.inf))
    exp = np.where(finite, exp, 0.0)
    return exp / exp.sum(axis=-1, keepdims=True)


def _attn_parts(X, params):
    """Return per-head Q, K, V, scores, patterns. X is (T, d), already normalized."""
    T = X.shape[0]
    heads = []
    W_Q, W_K, W_V = params["W_Q"], params["W_K"], params["W_V"]
    n_heads = W_Q.shape[0]
    dk = W_Q.shape[-1]
    for h in range(n_heads):
        Q = X @ W_Q[h]
        K = X @ W_K[h]
        V = X @ W_V[h]
        S = (Q @ K.T) / np.sqrt(dk)
        S = S.copy()
        S[np.triu(np.ones((T, T), dtype=bool), k=1)] = -np.inf
        P = softmax_rows(S)
        heads.append((Q, K, V, S, P))
    return heads


def attention(X, params):
    """Causal multi-head attention. Returns Y of shape (T, d) and the patterns."""
    parts = _attn_parts(X, params)
    dk = params["W_Q"].shape[-1]
    pieces = [P @ V for (_Q, _K, V, _S, P) in parts]
    concat = np.concatenate(pieces, axis=-1)
    Y = concat @ params["W_O"] + params["b_O"]
    patterns = [P for (_Q, _K, _V, _S, P) in parts]
    return Y, patterns, dk


def mlp(x, params):
    pre = x @ params["W1"].T + params["b1"]
    hidden = np.tanh(pre)
    return hidden @ params["W2"].T + params["b2"], hidden, pre


def block(H, params):
    """H1 = H + Attn(LN1(H)), H2 = H1 + MLP(LN2(H1))."""
    X = layer_norm(H, params["gamma1"], params["beta1"], params["eps"])
    A, patterns, _dk = attention(X, params)
    H1 = H + A
    U = layer_norm(H1, params["gamma2"], params["beta2"], params["eps"])
    M, hidden, pre = mlp(U, params)
    H2 = H1 + M
    return H2, {"X": X, "A": A, "patterns": patterns, "H1": H1, "U": U, "M": M, "hidden": hidden}


def margin(H, params):
    H2, cache = block(H, params)
    t = params["read_token"]
    return float(params["w_read"] @ H2[t]), H2, cache


def intervene(H, token, delta):
    out = np.array(H, dtype=float, copy=True)
    out[token] = out[token] + np.asarray(delta, dtype=float)
    return out


def response(H, params, token, delta):
    base, _, _ = margin(H, params)
    moved, _, _ = margin(intervene(H, token, delta), params)
    return moved - base


def _ln_jvp(h_row, dh_row, gamma, eps):
    return layer_norm_jacobian(h_row, gamma, eps) @ dh_row


def attention_jvp(X, dX, params, value_only=False):
    """Directional derivative of attention. value_only clamps every pattern."""
    parts = _attn_parts(X, params)
    dk = params["W_Q"].shape[-1]
    scale = np.sqrt(dk)
    d_pieces = []
    for h, (Q, K, V, S, P) in enumerate(parts):
        if value_only:
            dQ = np.zeros_like(Q)
            dK = np.zeros_like(K)
        else:
            dQ = dX @ params["W_Q"][h]
            dK = dX @ params["W_K"][h]
        dV = dX @ params["W_V"][h]
        raw = (dQ @ K.T + Q @ dK.T) / scale
        legal = np.isfinite(S)
        dS = np.where(legal, raw, 0.0)
        dP = np.zeros_like(P)
        for i in range(P.shape[0]):
            p = P[i]
            ds = dS[i]
            dP[i] = p * ds - p * float(p @ ds)
        d_head = dP @ V + P @ dV
        d_pieces.append(d_head)
    d_concat = np.concatenate(d_pieces, axis=-1)
    return d_concat @ params["W_O"]


def mlp_jvp(x, dx, params, hidden):
    # d tanh(pre) = (1 - tanh^2) * d pre, and hidden = tanh(pre).
    d_pre = dx @ params["W1"].T
    d_hidden = (1.0 - hidden * hidden) * d_pre
    return d_hidden @ params["W2"].T


def block_jvp(H, dH, params, value_only=False):
    """Chain-rule directional derivative of H2."""
    H2, cache = block(H, params)
    dX = np.vstack([
        _ln_jvp(H[i], dH[i], params["gamma1"], params["eps"])
        for i in range(H.shape[0])
    ])
    dA = attention_jvp(cache["X"], dX, params, value_only=value_only)
    dH1 = dH + dA
    dU = np.vstack([
        _ln_jvp(cache["H1"][i], dH1[i], params["gamma2"], params["eps"])
        for i in range(H.shape[0])
    ])
    dM = mlp_jvp(cache["U"], dU, params, cache["hidden"])
    dH2 = dH1 + dM
    return dH2, H2


def margin_jvp(H, dH, params, value_only=False):
    dH2, _H2 = block_jvp(H, dH, params, value_only=value_only)
    t = params["read_token"]
    return float(params["w_read"] @ dH2[t])


def token_direction(T, d, token, delta):
    dH = np.zeros((T, d))
    dH[token] = np.asarray(delta, dtype=float)
    return dH


def make_params(seed, T=3, d=4, n_heads=2, d_head=2, d_ff=5, eps=1e-5, scale=0.4):
    rng = np.random.default_rng(seed)
    def n(shape):
        return rng.normal(size=shape) * scale
    params = {
        "eps": float(eps),
        "T": T,
        "d": d,
        "gamma1": np.ones(d) + 0.05 * n(d),
        "beta1": 0.05 * n(d),
        "gamma2": np.ones(d) + 0.05 * n(d),
        "beta2": 0.05 * n(d),
        "W_Q": n((n_heads, d, d_head)),
        "W_K": n((n_heads, d, d_head)),
        "W_V": n((n_heads, d, d_head)),
        "W_O": n((n_heads * d_head, d)),
        "b_O": 0.05 * n(d),
        "W1": n((d_ff, d)),
        "b1": 0.05 * n(d_ff),
        "W2": n((d, d_ff)),
        "b2": 0.05 * n(d),
        "w_read": n(d),
        "read_token": T - 1,
    }
    H = rng.normal(size=(T, d))
    return params, H


def additive_margin(b, c1, c2):
    return b + c1 + c2
