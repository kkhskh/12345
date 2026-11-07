"""Vectorized pre-norm block used by the Step 1E joint-reachability package.

The structural model is a frozen causal block, not a product of independent
pathway balls. With query and key weights equal to zero the attention pattern
is the causal uniform pattern. The value map is a scalar multiple of the
identity, and the MLP is rank one. Those choices are hypotheses of the
theorems; the forward map itself accepts any weights of the recorded shapes.
"""

from __future__ import annotations

import numpy as np


def layer_norm(h, gamma, beta, eps):
    """LayerNorm over the last axis. Variance divides by the width, not width-1."""
    h = np.asarray(h, dtype=float)
    mean = h.mean(axis=-1, keepdims=True)
    centered = h - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    scaled = centered / np.sqrt(variance + eps)
    return gamma * scaled + beta


def softmax_rows(scores):
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError("softmax row has no finite score")
    shifted = np.where(finite, scores, -np.inf)
    peak = np.max(shifted, axis=-1, keepdims=True)
    expo = np.exp(np.where(finite, scores - peak, -np.inf))
    total = np.sum(expo, axis=-1, keepdims=True)
    return expo / total


def causal_mask(tokens):
    return np.triu(np.ones((tokens, tokens), dtype=bool), k=1)


def value_matrix(params):
    """Matrix V such that a fixed-pattern increment uses ΔX @ V."""
    parts = [params["W_V"][head] for head in range(params["W_V"].shape[0])]
    return np.concatenate(parts, axis=1) @ params["W_O"]


def attention(X, params):
    tokens = X.shape[0]
    width = params["W_Q"].shape[-1]
    mask = causal_mask(tokens)
    heads = []
    weights = []
    for head in range(params["W_Q"].shape[0]):
        query = X @ params["W_Q"][head]
        key = X @ params["W_K"][head]
        value = X @ params["W_V"][head]
        scores = (query @ key.T) / np.sqrt(width)
        scores = scores.copy()
        scores[mask] = -np.inf
        pattern = softmax_rows(scores)
        heads.append(pattern @ value)
        weights.append(pattern)
    concat = np.concatenate(heads, axis=-1)
    output = concat @ params["W_O"] + params["b_O"]
    return output, weights


def mlp(U, params):
    hidden = np.tanh(U @ params["W1"].T + params["b1"])
    return hidden @ params["W2"].T + params["b2"]


def forward(H, params):
    residual = np.asarray(H, dtype=float)
    normalized = layer_norm(residual, params["gamma1"], params["beta1"], params["eps"])
    attended, weights = attention(normalized, params)
    hidden = residual + attended
    second = layer_norm(hidden, params["gamma2"], params["beta2"], params["eps"])
    mixed = mlp(second, params)
    output = hidden + mixed
    return {
        "X": normalized,
        "A": attended,
        "H1": hidden,
        "U": second,
        "M": mixed,
        "H2": output,
        "weights": weights,
    }


def intervene(H, token, delta):
    shifted = np.array(H, dtype=float, copy=True)
    shifted[token] = shifted[token] + np.asarray(delta, dtype=float)
    return shifted


def pathway_increments(H, token, delta, params):
    """Attention and MLP increments from one recomputed intervention."""
    base = forward(H, params)
    moved = forward(intervene(H, token, delta), params)
    return moved["A"] - base["A"], moved["M"] - base["M"], base, moved


def raw_readout(state, params):
    row = int(params["read_token"])
    return float(params["w_read"] @ state[row])


def normalized_readout(state, params):
    row = int(params["read_token"])
    normalized = layer_norm(
        state[row], params["gamma_f"], params["beta_f"], params["eps"]
    )
    return float(params["w_read"] @ normalized)


def raw_response(H, token, delta, params):
    base = forward(H, params)
    moved = forward(intervene(H, token, delta), params)
    return raw_readout(moved["H2"], params) - raw_readout(base["H2"], params)


def normalized_response(H, token, delta, params):
    base = forward(H, params)
    moved = forward(intervene(H, token, delta), params)
    return normalized_readout(moved["H2"], params) - normalized_readout(base["H2"], params)


def mean_zero_axis(width=2):
    if width != 2:
        raise ValueError("the invariant-line axis is constructed in width 2")
    return np.array([1.0, -1.0]) / np.sqrt(2.0)


def structural_params(mu=0.5, alpha=1.0, s_gain=1.0, eps=1e-5, tokens=3, heads=2):
    """Frozen block whose last-token and earlier-token increments reduce to the line.

    Query and key weights are zero, so the pattern is uniform on the causal past.
    Two heads each carry an identity value map. Their output blocks are equal,
    and the product of the concatenated value map with W_O is mu * tokens * I.
    Multiplied by the causal weight 1/tokens, a single-token LayerNorm change
    therefore contributes mu times that change at every later token, including
    the edited token itself.
    """
    if heads != 2 or tokens < 2:
        raise ValueError("the structural block uses two heads and at least two tokens")
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    width = 2
    head_width = 2
    axis = mean_zero_axis(width)
    readout = axis.copy()
    scale = mu * tokens / 2.0
    eye = np.eye(width)
    value = np.stack([eye, eye], axis=0)
    output = np.vstack([scale * eye, scale * eye])
    zeros = np.zeros((heads, width, head_width))
    return {
        "T": tokens,
        "d": width,
        "n_heads": heads,
        "d_head": head_width,
        "d_ff": 1,
        "eps": float(eps),
        "read_token": tokens - 1,
        "gamma1": np.ones(width),
        "beta1": np.zeros(width),
        "gamma2": np.ones(width),
        "beta2": np.zeros(width),
        "gamma_f": np.ones(width),
        "beta_f": np.zeros(width),
        "W_Q": zeros.copy(),
        "W_K": zeros.copy(),
        "W_V": value,
        "W_O": output,
        "b_O": np.zeros(width),
        "W1": (-alpha * readout).reshape(1, width),
        "b1": np.zeros(1),
        "W2": (s_gain * readout).reshape(width, 1),
        "b2": np.zeros(width),
        "w_read": readout,
        "mu": float(mu),
        "alpha": float(alpha),
        "s_gain": float(s_gain),
        "axis": axis,
    }


def zero_state(params, coordinate=0.0):
    state = np.zeros((params["T"], params["d"]))
    if coordinate != 0.0:
        state[:] = coordinate * params["axis"]
    return state


def line_delta(params, step):
    return float(step) * params["axis"]
