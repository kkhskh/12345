"""Constructions that separate the joint set, its product relaxation, and a norm ball.

None of these functions is a certificate. They build the counterexamples
named in the register.
"""

from __future__ import annotations

import numpy as np

from .block import pathway_increments, structural_params, value_matrix


def singular_value_block(mu=1.0, eps=1e-5):
    """Width-3 fixed-pattern block whose value map keeps only one mean-zero axis.

    Two edits can share that attention component and still feed different
    vectors to the MLP.
    """
    tokens, width, heads, head_width = 3, 3, 2, 2
    axis_a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    axis_b = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    projector = np.outer(axis_a, axis_a)
    # Two heads implement the rank-one value map μ T P, then the pattern
    # weight 1/T brings the increment down to μ ΔX @ P.
    scale = mu * tokens / 2.0
    value = np.zeros((heads, width, head_width))
    # Pad the 3-dimensional projector through a 4-wide concatenation by
    # placing the same factor in two heads of width 2. The product
    # W_Vcat @ W_O = 2 scale P = mu T P.
    factor = np.zeros((width, head_width))
    factor[:, 0] = axis_a
    value[0] = factor
    value[1] = factor
    output = np.zeros((heads * head_width, width))
    output[0] = scale * axis_a
    output[head_width] = scale * axis_a
    params = structural_params(mu=mu, eps=eps)
    params["d"] = width
    params["gamma1"] = np.ones(width)
    params["beta1"] = np.zeros(width)
    params["gamma2"] = np.ones(width)
    params["beta2"] = np.zeros(width)
    params["gamma_f"] = np.ones(width)
    params["beta_f"] = np.zeros(width)
    params["W_Q"] = np.zeros((heads, width, head_width))
    params["W_K"] = np.zeros((heads, width, head_width))
    params["W_V"] = value
    params["W_O"] = output
    params["b_O"] = np.zeros(width)
    params["W1"] = axis_b.reshape(1, width)
    params["b1"] = np.zeros(1)
    params["W2"] = axis_b.reshape(width, 1)
    params["b2"] = np.zeros(width)
    params["w_read"] = axis_a.copy()
    params["axis_a"] = axis_a
    params["axis_b"] = axis_b
    params["projector"] = projector
    return params


def same_attention_different_mlp(radius=0.4, angle=0.7):
    """Two width-3 edits with equal attention increments and opposite MLP increments.

    The two edits are reflections through axis_a. They have the same norm, so
    their LayerNorm images are reflections. The value map keeps only axis_a,
    and the MLP reads axis_b.
    """
    params = singular_value_block()
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    first = radius * (cosine * params["axis_a"] + sine * params["axis_b"])
    second = radius * (cosine * params["axis_a"] - sine * params["axis_b"])
    base = np.zeros((params["T"], params["d"]))
    token = params["read_token"]
    a1, b1, _base, _moved = pathway_increments(base, token, first, params)
    a2, b2, _base, _moved = pathway_increments(base, token, second, params)
    return {
        "first": first,
        "second": second,
        "a1": a1[token],
        "a2": a2[token],
        "b1": b1[token],
        "b2": b2[token],
    }


def product_ball_ceiling(attention_radius, mlp_radius, attention_direction, mlp_direction):
    """Set C maximum of w·a + w·b over independent Euclidean balls.

    This is the Step 1D product-ball ceiling. It is not the jointly reachable set.
    """
    return attention_radius * float(np.linalg.norm(attention_direction)) + mlp_radius * float(
        np.linalg.norm(mlp_direction)
    )


def moving_pattern_params(seed=1, scale=0.3):
    """Structural block with nonzero query and key weights. The line formula does not apply."""
    rng = np.random.default_rng(seed)
    params = structural_params()
    params["W_Q"] = scale * rng.normal(size=params["W_Q"].shape)
    params["W_K"] = scale * rng.normal(size=params["W_K"].shape)
    return params


def fiber_pair(params, token, delta, shift):
    constant = shift * np.ones(params["d"])
    base = np.zeros((params["T"], params["d"]))
    a1, b1, _b, _m = pathway_increments(base, token, delta, params)
    a2, b2, _b, _m = pathway_increments(base, token, delta + constant, params)
    return a1, b1, a2, b2


def value_map_is_isotropic(params):
    mapped = value_matrix(params)
    target = params["mu"] * params["T"] * np.eye(params["d"])
    return float(np.linalg.norm(mapped - target))
