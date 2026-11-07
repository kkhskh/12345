"""One pre-norm residual block.

The architecture is the block fixed in step1c_r_full_block: T=3, d=4, two heads,
tanh MLP, epsilon 1e-5, causal mask, readout on the last token. This file is a
self-contained implementation of that specification. It is not a pretrained model.

Intervention semantics, fixed before any bound:

    do(H_t <- H_t + delta) replaces one residual token and recomputes the block.
    m(H) = w_read · H_2[read_token]
    R(H, t, delta) = m(do(H_t <- H_t + delta)) - m(H)

Queries, keys, and values are not clamped. A value-only derivative is a different
map and is labeled as such.
"""

from __future__ import annotations

import numpy as np


def centering_matrix(d):
    return np.eye(d) - np.ones((d, d)) / d


def layer_norm(h, gamma, beta, eps):
    """Per-vector LayerNorm. h has shape (..., d)."""
    h = np.asarray(h, dtype=float)
    z = h - h.mean(axis=-1, keepdims=True)
    v = np.mean(z * z, axis=-1, keepdims=True)
    out = z / np.sqrt(v + eps)
    return np.asarray(gamma, dtype=float) * out + np.asarray(beta, dtype=float)


def layer_norm_jacobian(h, gamma, eps):
    """J = diag(gamma) @ (C / s - z z^T / (d s^3)), s = sqrt(v+eps)."""
    h = np.asarray(h, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    d = h.shape[0]
    C = centering_matrix(d)
    z = C @ h
    v = float(z @ z / d)
    s = np.sqrt(v + eps)
    bracket = C / s - np.outer(z, z) / (d * s**3)
    return np.diag(gamma) @ bracket


def layer_norm_bracket_spectrum(h, eps):
    """Eigenvalues of the symmetric bracket, computed from the closed form.

    For eps>0 the eigenvalues are 0, and 1/s with multiplicity d-1 when z=0.
    When z≠0 they are 0, eps/s^3, and 1/s with multiplicity d-2.
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    d = h.shape[0]
    C = centering_matrix(d)
    z = C @ h
    v = float(z @ z / d)
    s = np.sqrt(v + eps)
    if v == 0.0:
        values = np.array([0.0] + [1.0 / s] * (d - 1))
    else:
        values = np.array([0.0, eps / s**3] + [1.0 / s] * (d - 2))
    return values, s, v


def softmax_rows(scores):
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    safe = np.where(finite, scores, -np.inf)
    row_max = np.max(np.where(finite, safe, -np.inf), axis=-1, keepdims=True)
    # An all-masked row cannot occur under a causal mask with a self-loop.
    shifted = np.where(finite, safe - row_max, -np.inf)
    exp = np.exp(np.where(finite, shifted, -np.inf))
    exp = np.where(finite, exp, 0.0)
    return exp / exp.sum(axis=-1, keepdims=True)


def _attn_parts(X, params):
    T = X.shape[0]
    heads = []
    W_Q, W_K, W_V = params["W_Q"], params["W_K"], params["W_V"]
    dk = W_Q.shape[-1]
    for h in range(W_Q.shape[0]):
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
    parts = _attn_parts(X, params)
    pieces = [P @ V for (_Q, _K, V, _S, P) in parts]
    concat = np.concatenate(pieces, axis=-1)
    Y = concat @ params["W_O"] + params["b_O"]
    patterns = [P for (_Q, _K, _V, _S, P) in parts]
    return Y, patterns


def mlp(x, params):
    pre = x @ params["W1"].T + params["b1"]
    hidden = np.tanh(pre)
    return hidden @ params["W2"].T + params["b2"], hidden, pre


def block(H, params):
    """H1 = H + Attn(LN1(H)), H2 = H1 + MLP(LN2(H1))."""
    X = layer_norm(H, params["gamma1"], params["beta1"], params["eps"])
    A, patterns = attention(X, params)
    H1 = H + A
    U = layer_norm(H1, params["gamma2"], params["beta2"], params["eps"])
    M, hidden, pre = mlp(U, params)
    H2 = H1 + M
    cache = {"X": X, "A": A, "patterns": patterns, "H1": H1, "U": U, "M": M, "hidden": hidden}
    return H2, cache


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
        d_pieces.append(dP @ V + P @ dV)
    return np.concatenate(d_pieces, axis=-1) @ params["W_O"]


def mlp_jvp(dx, params, hidden):
    d_pre = dx @ params["W1"].T
    d_hidden = (1.0 - hidden * hidden) * d_pre
    return d_hidden @ params["W2"].T


def block_jvp(H, dH, params, value_only=False):
    """Chain-rule directional derivative of H2. This is a derivative, not R."""
    _H2, cache = block(H, params)
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
    dM = mlp_jvp(dU, params, cache["hidden"])
    return dH1 + dM


def margin_jvp(H, dH, params, value_only=False):
    dH2 = block_jvp(H, dH, params, value_only=value_only)
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


def centered(delta):
    delta = np.asarray(delta, dtype=float)
    return delta - delta.mean()


def constant_component(delta):
    delta = np.asarray(delta, dtype=float)
    return float(delta.mean())


def skip_prediction(params, token, delta):
    """Exact response of a pure constant, and the skip dot product in general.

    For a constant vector a*1 this equals R for every branch weight. For a
    non-constant vector it is only the skip term.
    """
    if token != params["read_token"]:
        return 0.0
    return float(params["w_read"] @ np.asarray(delta, dtype=float))
