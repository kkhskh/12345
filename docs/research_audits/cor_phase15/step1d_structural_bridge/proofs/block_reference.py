"""Pre-norm block, implementation B.

Explicit token and head loops, sums written out, no call into block_forward.
Derivatives, when requested, are central finite differences of this forward map.
The analytic Jacobian lives in block_jvp.py and must not be used here.
"""

from __future__ import annotations

import numpy as np


def layer_norm_rows(H, gamma, beta, eps):
    H = np.asarray(H, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    tokens, width = H.shape
    out = np.zeros_like(H)
    for token in range(tokens):
        total = 0.0
        for coord in range(width):
            total += H[token, coord]
        mean = total / width
        second = 0.0
        for coord in range(width):
            diff = H[token, coord] - mean
            second += diff * diff
        scale = 1.0 / np.sqrt(second / width + eps)
        for coord in range(width):
            out[token, coord] = gamma[coord] * (H[token, coord] - mean) * scale + beta[coord]
    return out


def _softmax(scores):
    finite = [value for value in scores if np.isfinite(value)]
    peak = max(finite)
    weights = []
    for value in scores:
        if np.isfinite(value):
            weights.append(float(np.exp(value - peak)))
        else:
            weights.append(0.0)
    total = sum(weights)
    return [weight / total for weight in weights]


def block_parts(H, params):
    H = np.asarray(H, dtype=float)
    tokens, _width = H.shape
    X = layer_norm_rows(H, params["gamma1"], params["beta1"], params["eps"])
    n_heads = params["W_Q"].shape[0]
    head_width = params["W_Q"].shape[-1]
    attention = np.zeros_like(H)
    head_outputs = []
    for head in range(n_heads):
        query = X @ params["W_Q"][head]
        key = X @ params["W_K"][head]
        value = X @ params["W_V"][head]
        mixed = np.zeros((tokens, head_width))
        for row in range(tokens):
            scores = []
            for col in range(tokens):
                if col <= row:
                    dot = 0.0
                    for coord in range(head_width):
                        dot += query[row, coord] * key[col, coord]
                    scores.append(dot / np.sqrt(head_width))
                else:
                    scores.append(-np.inf)
            weights = _softmax(scores)
            for col in range(tokens):
                for coord in range(head_width):
                    mixed[row, coord] += weights[col] * value[col, coord]
        head_outputs.append(mixed)
    concatenated = np.concatenate(head_outputs, axis=-1)
    attention = concatenated @ params["W_O"] + params["b_O"]
    residual = H + attention
    second = layer_norm_rows(residual, params["gamma2"], params["beta2"], params["eps"])
    preactivation = second @ params["W1"].T + params["b1"]
    hidden = np.tanh(preactivation)
    mlp = hidden @ params["W2"].T + params["b2"]
    output = residual + mlp
    return output, attention, mlp, hidden, concatenated


def margin(H, params):
    output, attention, mlp, hidden, concatenated = block_parts(H, params)
    token = params["read_token"]
    return float(np.dot(params["w_read"], output[token])), output, attention, mlp, hidden, concatenated


def response(H, params, token, delta):
    base, _, _, _, _, _ = margin(H, params)
    edited = np.array(H, dtype=float, copy=True)
    edited[token] = edited[token] + np.asarray(delta, dtype=float)
    moved, _, _, _, _, _ = margin(edited, params)
    return moved - base


def token_gradient(H, params, token, step=1e-6):
    """Central differences. Does not call the analytic JVP."""
    width = H.shape[1]
    gradient = np.zeros(width)
    for coord in range(width):
        direction = np.zeros(width)
        direction[coord] = 1.0
        forward = response(H, params, token, step * direction)
        backward = response(H, params, token, -step * direction)
        gradient[coord] = (forward - backward) / (2.0 * step)
    return gradient
