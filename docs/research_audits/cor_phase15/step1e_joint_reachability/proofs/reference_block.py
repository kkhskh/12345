"""Loop implementation of the same block, with finite-difference gradients.

This file does not import the vectorized forward. Tests compare the two
routes on identical weights. PyTorch is not installed in this environment,
so this is the second numerical route.
"""

from __future__ import annotations

import math

import numpy as np


def layer_norm_token(row, gamma, beta, eps):
    width = row.shape[0]
    mean = 0.0
    for value in row:
        mean += float(value)
    mean /= width
    variance = 0.0
    centered = np.empty(width)
    for index, value in enumerate(row):
        centered[index] = float(value) - mean
        variance += centered[index] * centered[index]
    variance /= width
    scale = math.sqrt(variance + eps)
    output = np.empty(width)
    for index in range(width):
        output[index] = float(gamma[index]) * centered[index] / scale + float(beta[index])
    return output


def layer_norm(H, gamma, beta, eps):
    return np.vstack([layer_norm_token(row, gamma, beta, eps) for row in H])


def attention(X, params):
    tokens, _width = X.shape
    heads = params["W_Q"].shape[0]
    head_width = params["W_Q"].shape[-1]
    output = np.zeros((tokens, params["d"]))
    patterns = []
    concat_rows = []
    for token in range(tokens):
        concat_rows.append([])
    for head in range(heads):
        query = X @ params["W_Q"][head]
        key = X @ params["W_K"][head]
        value = X @ params["W_V"][head]
        pattern = np.zeros((tokens, tokens))
        for token in range(tokens):
            scores = []
            for source in range(token + 1):
                dot = 0.0
                for coord in range(head_width):
                    dot += float(query[token, coord]) * float(key[source, coord])
                scores.append(dot / math.sqrt(head_width))
            peak = max(scores)
            weights = [math.exp(score - peak) for score in scores]
            total = sum(weights)
            for source, weight in enumerate(weights):
                pattern[token, source] = weight / total
            mixed = np.zeros(head_width)
            for source in range(token + 1):
                mixed += pattern[token, source] * value[source]
            concat_rows[token].append(mixed)
        patterns.append(pattern)
    concat = np.zeros((tokens, heads * head_width))
    for token in range(tokens):
        cursor = 0
        for piece in concat_rows[token]:
            concat[token, cursor : cursor + head_width] = piece
            cursor += head_width
    for token in range(tokens):
        output[token] = concat[token] @ params["W_O"] + params["b_O"]
    return output, patterns


def mlp(U, params):
    hidden = np.tanh(U @ params["W1"].T + params["b1"])
    return hidden @ params["W2"].T + params["b2"]


def forward(H, params):
    residual = np.asarray(H, dtype=float)
    normalized = layer_norm(residual, params["gamma1"], params["beta1"], params["eps"])
    attended, patterns = attention(normalized, params)
    hidden = residual + attended
    second = layer_norm(hidden, params["gamma2"], params["beta2"], params["eps"])
    mixed = mlp(second, params)
    return {
        "X": normalized,
        "A": attended,
        "H1": hidden,
        "U": second,
        "M": mixed,
        "H2": hidden + mixed,
        "weights": patterns,
    }


def raw_response(H, token, delta, params):
    base = forward(H, params)
    shifted = np.array(H, dtype=float, copy=True)
    shifted[int(token)] = shifted[int(token)] + np.asarray(delta, dtype=float)
    moved = forward(shifted, params)
    row = int(params["read_token"])
    return float(params["w_read"] @ (moved["H2"][row] - base["H2"][row]))


def directional_derivative(H, token, direction, params, step):
    """Central difference of the raw readout along one direction."""
    direction = np.asarray(direction, dtype=float)
    forward_step = raw_response(H, token, step * direction, params)
    backward_step = raw_response(H, token, -step * direction, params)
    return (forward_step - backward_step) / (2.0 * step)


def coordinate_gradient(H, token, params, step):
    width = H.shape[1]
    gradient = np.zeros(width)
    for coord in range(width):
        direction = np.zeros(width)
        direction[coord] = 1.0
        gradient[coord] = directional_derivative(H, token, direction, params, step)
    return gradient
