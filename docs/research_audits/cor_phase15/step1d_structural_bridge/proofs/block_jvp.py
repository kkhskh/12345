"""Analytic directional derivatives of implementation A.

The split into skip, attention, and MLP is the chain rule on
H2 = H + A(LN(H)) + M(LN(H + A(LN(H)))).
Their sum is the derivative of m. It is not the finite response.
"""

from __future__ import annotations

import numpy as np

from proofs.block_forward import block


def _centering(width):
    return np.eye(width) - np.ones((width, width)) / width


def layer_norm_jacobian(h, gamma, eps):
    h = np.asarray(h, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    width = h.shape[0]
    centered_map = _centering(width)
    z = centered_map @ h
    variance = float(z @ z / width)
    scale = np.sqrt(variance + eps)
    bracket = centered_map / scale - np.outer(z, z) / (width * scale**3)
    return np.diag(gamma) @ bracket


def _ln_jvp(row, d_row, gamma, eps):
    return layer_norm_jacobian(row, gamma, eps) @ d_row


def attention_jvp(X, dX, params):
    tokens = X.shape[0]
    width = params["W_Q"].shape[-1]
    scale = np.sqrt(width)
    pieces = []
    for head in range(params["W_Q"].shape[0]):
        query = X @ params["W_Q"][head]
        key = X @ params["W_K"][head]
        value = X @ params["W_V"][head]
        scores = (query @ key.T) / scale
        scores = scores.copy()
        scores[np.triu(np.ones((tokens, tokens), dtype=bool), k=1)] = -np.inf
        pattern = _softmax_from_scores(scores)
        d_query = dX @ params["W_Q"][head]
        d_key = dX @ params["W_K"][head]
        d_value = dX @ params["W_V"][head]
        raw = (d_query @ key.T + query @ d_key.T) / scale
        legal = np.isfinite(scores)
        d_scores = np.where(legal, raw, 0.0)
        d_pattern = np.zeros_like(pattern)
        for row in range(tokens):
            weights = pattern[row]
            d_row = d_scores[row]
            d_pattern[row] = weights * d_row - weights * float(weights @ d_row)
        pieces.append(d_pattern @ value + pattern @ d_value)
    d_concat = np.concatenate(pieces, axis=-1)
    return d_concat @ params["W_O"], d_concat


def _softmax_from_scores(scores):
    finite = np.isfinite(scores)
    row_max = np.max(np.where(finite, scores, -np.inf), axis=-1, keepdims=True)
    shifted = np.where(finite, scores - row_max, -np.inf)
    exp_scores = np.where(finite, np.exp(shifted), 0.0)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def mlp_jvp(dU, params, hidden):
    d_pre = dU @ params["W1"].T
    d_hidden = (1.0 - hidden * hidden) * d_pre
    d_mlp = d_hidden @ params["W2"].T
    return d_mlp, d_hidden


def component_jvps(H, dH, params):
    """Return read-token derivatives of skip, attention, and MLP, plus hidden JVP."""
    _output, cache = block(H, params)
    dX = np.vstack([
        _ln_jvp(H[i], dH[i], params["gamma1"], params["eps"])
        for i in range(H.shape[0])
    ])
    dA, d_concat = attention_jvp(cache["X"], dX, params)
    dH1 = dH + dA
    dU = np.vstack([
        _ln_jvp(cache["H1"][i], dH1[i], params["gamma2"], params["eps"])
        for i in range(H.shape[0])
    ])
    dM, d_hidden = mlp_jvp(dU, params, cache["hidden"])
    token = params["read_token"]
    w = params["w_read"]
    d_skip = float(w @ dH[token])
    d_attn = float(w @ dA[token])
    d_mlp = float(w @ dM[token])
    return {
        "skip": d_skip,
        "attention": d_attn,
        "mlp": d_mlp,
        "total": d_skip + d_attn + d_mlp,
        "d_hidden_read": d_hidden[token].copy(),
        "dA": dA,
        "dM": dM,
    }


def token_gradient(H, params, token):
    width = H.shape[1]
    gradient = np.zeros(width)
    for coord in range(width):
        direction = np.zeros((H.shape[0], width))
        direction[token, coord] = 1.0
        gradient[coord] = component_jvps(H, direction, params)["total"]
    return gradient


def branch_cancellation_weights(H, params, token, delta):
    """Rank-one update of W2 that makes the branch derivative along delta vanish.

    The update is computed from the base-point hidden derivative. It does not
    use the finite response. After the update, the attention and MLP
    directional derivatives sum to zero along this delta. The skip derivative
    is unchanged.
    """
    direction = np.zeros_like(H)
    direction[token] = np.asarray(delta, dtype=float)
    parts = component_jvps(H, direction, params)
    hidden_dir = parts["d_hidden_read"]
    hidden_norm = float(hidden_dir @ hidden_dir)
    w = params["w_read"]
    w_norm = float(w @ w)
    if hidden_norm < 1e-18 or w_norm == 0.0:
        raise RuntimeError("cancellation update is undefined at this point")
    target = -(parts["attention"] + parts["mlp"])
    coefficient = target / (hidden_norm * w_norm)
    updated = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in params.items()}
    updated["W2"] = updated["W2"] + coefficient * np.outer(w, hidden_dir)
    return updated, parts
