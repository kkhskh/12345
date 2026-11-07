"""Pre-norm block, implementation A.

Vectorized NumPy. The architecture is the overnight block: T=3, d=4, two heads,
head dimension 2, tanh MLP, epsilon 1e-5, causal mask, raw readout on the last token.

    X = LN(H)
    H1 = H + Attn(X)
    U = LN(H1)
    H2 = H1 + MLP(U)
    m(H) = w · H2[r]

Intervention, fixed before any bound:

    do(H_t <- H_t + delta) replaces one residual token and recomputes the block.
    R(H, t, delta) = m(do(...)) - m(H).

This file does not compute Lipschitz certificates.
"""

from __future__ import annotations

import numpy as np


def layer_norm(h, gamma, beta, eps):
    h = np.asarray(h, dtype=float)
    centered = h - np.mean(h, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    normal = centered / np.sqrt(variance + eps)
    return np.asarray(gamma, dtype=float) * normal + np.asarray(beta, dtype=float)


def softmax_rows(scores):
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    row_max = np.max(np.where(finite, scores, -np.inf), axis=-1, keepdims=True)
    shifted = np.where(finite, scores - row_max, -np.inf)
    exp_scores = np.exp(np.where(finite, shifted, -np.inf))
    exp_scores = np.where(finite, exp_scores, 0.0)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def attention_from_normalized(X, params):
    """Causal multi-head attention on an already normalized token matrix."""
    tokens = X.shape[0]
    W_Q, W_K, W_V = params["W_Q"], params["W_K"], params["W_V"]
    width = W_Q.shape[-1]
    pieces = []
    patterns = []
    for head in range(W_Q.shape[0]):
        query = X @ W_Q[head]
        key = X @ W_K[head]
        value = X @ W_V[head]
        scores = (query @ key.T) / np.sqrt(width)
        scores = scores.copy()
        scores[np.triu(np.ones((tokens, tokens), dtype=bool), k=1)] = -np.inf
        pattern = softmax_rows(scores)
        pieces.append(pattern @ value)
        patterns.append(pattern)
    concatenated = np.concatenate(pieces, axis=-1)
    output = concatenated @ params["W_O"] + params["b_O"]
    return output, patterns, concatenated


def mlp_from_normalized(U, params):
    preactivation = U @ params["W1"].T + params["b1"]
    hidden = np.tanh(preactivation)
    output = hidden @ params["W2"].T + params["b2"]
    return output, hidden, preactivation


def block(H, params):
    normalized = layer_norm(H, params["gamma1"], params["beta1"], params["eps"])
    attention, patterns, concatenated = attention_from_normalized(normalized, params)
    residual = H + attention
    second = layer_norm(residual, params["gamma2"], params["beta2"], params["eps"])
    mlp, hidden, preactivation = mlp_from_normalized(second, params)
    output = residual + mlp
    cache = {
        "X": normalized,
        "A": attention,
        "patterns": patterns,
        "concat": concatenated,
        "H1": residual,
        "U": second,
        "M": mlp,
        "hidden": hidden,
        "preactivation": preactivation,
    }
    return output, cache


def margin(H, params):
    output, cache = block(H, params)
    token = params["read_token"]
    return float(params["w_read"] @ output[token]), output, cache


def intervene(H, token, delta):
    edited = np.array(H, dtype=float, copy=True)
    edited[token] = edited[token] + np.asarray(delta, dtype=float)
    return edited


def response(H, params, token, delta):
    base, _, _ = margin(H, params)
    moved, _, _ = margin(intervene(H, token, delta), params)
    return moved - base


def structural_residual(H, params):
    """Recomputed trace versus the two residual assignments. Exact on this block."""
    output, cache = block(H, params)
    first = cache["H1"] - (H + cache["A"])
    second = output - (cache["H1"] + cache["M"])
    return float(np.max(np.abs(first))), float(np.max(np.abs(second)))


def make_params(seed, T=3, d=4, n_heads=2, d_head=2, d_ff=5, eps=1e-5, scale=0.4):
    """Same draw as the overnight `make_params` under NumPy 1.26.4 `default_rng`."""
    rng = np.random.default_rng(seed)

    def draw(shape):
        return rng.normal(size=shape) * scale

    params = {
        "eps": float(eps),
        "T": T,
        "d": d,
        "gamma1": np.ones(d) + 0.05 * draw(d),
        "beta1": 0.05 * draw(d),
        "gamma2": np.ones(d) + 0.05 * draw(d),
        "beta2": 0.05 * draw(d),
        "W_Q": draw((n_heads, d, d_head)),
        "W_K": draw((n_heads, d, d_head)),
        "W_V": draw((n_heads, d, d_head)),
        "W_O": draw((n_heads * d_head, d)),
        "b_O": 0.05 * draw(d),
        "W1": draw((d_ff, d)),
        "b1": 0.05 * draw(d_ff),
        "W2": draw((d, d_ff)),
        "b2": 0.05 * draw(d),
        "w_read": draw(d),
        "read_token": T - 1,
    }
    state = rng.normal(size=(T, d))
    return params, state


def scale_branch_outputs(params, alpha):
    """Scale the branch output maps W_O and W2. Other weights stay fixed."""
    scaled = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in params.items()}
    scaled["W_O"] = alpha * scaled["W_O"]
    scaled["W2"] = alpha * scaled["W2"]
    return scaled


def project_outputs_off_readout(params):
    """Enforce W_O w = 0 and W2.T w = 0. Q, K, V are untouched."""
    out = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in params.items()}
    w = out["w_read"]
    denom = float(w @ w)
    if denom == 0.0:
        return out
    # Y = concat @ W_O, so w·Y = concat·(W_O w). Kill W_O w.
    out["W_O"] = out["W_O"] - np.outer(out["W_O"] @ w, w) / denom
    # M = hidden @ W2.T, so w·M = hidden·(W2.T w). Kill W2.T w.
    out["W2"] = out["W2"] - np.outer(w, w @ out["W2"]) / denom
    return out


def orthogonal_unit(w, seed_vector):
    """Unit vector in w⊥ obtained from a fixed seed vector. None if the projection vanishes."""
    w = np.asarray(w, dtype=float)
    seed = np.asarray(seed_vector, dtype=float)
    denom = float(w @ w)
    if denom == 0.0:
        return None
    projected = seed - w * float(w @ seed) / denom
    norm = float(np.linalg.norm(projected))
    if norm == 0.0:
        return None
    return projected / norm


def stacked_qkv(params):
    """Map R^d -> R^{3 n_heads d_head} given by every Q, K, and V matrix."""
    blocks = []
    for tensor in (params["W_Q"], params["W_K"], params["W_V"]):
        for head in range(tensor.shape[0]):
            blocks.append(tensor[head].T)
    return np.vstack(blocks)


def qkv_smallest_singular_value(params):
    matrix = stacked_qkv(params)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular[-1])
