"""Checks for the Step 1C-R corrections and one pre-norm block.

Run from the package root:

    python -m pytest tests -q

A pass means this file and proofs/block_math.py agree on these maps.
It is not a theorem and it is not a trained Transformer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))

from block_math import (  # noqa: E402
    attention,
    attention_jvp,
    block,
    centering_matrix,
    intervene,
    layer_norm,
    layer_norm_jacobian,
    make_params,
    margin,
    margin_jvp,
    response,
    token_direction,
)


def _fd_jacobian(h, gamma, beta, eps, step=1e-6):
    h = np.asarray(h, dtype=float)
    cols = []
    for i in range(h.shape[0]):
        e = np.zeros_like(h)
        e[i] = step
        forward = layer_norm(h + e, gamma, beta, eps)
        backward = layer_norm(h - e, gamma, beta, eps)
        cols.append((forward - backward) / (2 * step))
    return np.column_stack(cols)


def test_sympy_matches_the_layer_norm_jacobian_formula():
    sympy = pytest.importorskip("sympy")
    d = 3
    symbols = sympy.symbols(f"h0:{d}")
    eps = sympy.symbols("eps", positive=True)
    hv = sympy.Matrix(symbols)
    mean = sum(hv) / d
    z = hv - mean * sympy.ones(d, 1)
    v = z.dot(z) / d
    out = z / sympy.sqrt(v + eps)
    got = sympy.simplify(out.jacobian(hv))
    eye = sympy.eye(d) - sympy.ones(d) / d
    s = sympy.sqrt(v + eps)
    formula = sympy.simplify(eye / s - (z * z.T) / (d * s**3))
    assert sympy.simplify(got - formula) == sympy.zeros(d)


def test_analytic_layer_norm_jacobian_matches_finite_differences():
    h = np.array([0.4, -1.2, 0.2, 0.7, -0.3])
    gamma = np.array([1.0, 0.5, 1.5, 0.8, 1.2])
    beta = np.array([0.1, -0.2, 0.0, 0.3, -0.1])
    for eps in (0.0, 1e-5, 1e-2):
        analytic = layer_norm_jacobian(h, gamma, eps)
        numeric = _fd_jacobian(h, gamma, beta, eps)
        assert analytic == pytest.approx(numeric, abs=1e-7)


def test_positive_epsilon_does_not_kill_the_centered_direction():
    h = np.array([1.0, 0.2, -0.4, 0.7])
    gamma = np.ones(4)
    C = centering_matrix(4)
    z = C @ h
    v = float(z @ z / 4)
    eps = 1e-5
    s = np.sqrt(v + eps)
    J = layer_norm_jacobian(h, gamma, eps)
    assert np.linalg.norm(J @ np.ones(4)) < 1e-12
    assert J @ z == pytest.approx(eps * z / s**3, abs=1e-12)
    assert np.linalg.norm(J @ z) > 1e-6
    J0 = layer_norm_jacobian(h, gamma, 0.0)
    assert np.linalg.norm(J0 @ np.ones(4)) < 1e-12
    assert np.linalg.norm(J0 @ z) < 1e-12


def test_a_loose_singular_value_cutoff_misreads_small_epsilon():
    h = np.array([1.0, 0.2, -0.4, 0.7])
    J = layer_norm_jacobian(h, np.ones(4), 1e-5)
    singular = np.linalg.svd(J, compute_uv=False)
    assert singular[-1] < 1e-12
    assert singular[-2] > 1e-6
    assert np.linalg.matrix_rank(J, tol=1e-8) == 3
    assert np.linalg.matrix_rank(J, tol=1e-3) == 2


def test_one_zero_gamma_need_not_drop_rank_and_two_zeros_do():
    h = np.array([1.0, 0.2, -0.4, 0.7])
    # The bracket's image is the complement of the all-ones vector. That
    # complement meets a coordinate axis only at 0, so one zero gain need
    # not reduce the rank.
    one_zero = layer_norm_jacobian(h, np.array([0.0, 1.0, 1.0, 1.0]), 1e-5)
    assert np.linalg.matrix_rank(one_zero, tol=1e-8) == 3
    two_zeros = layer_norm_jacobian(h, np.array([0.0, 0.0, 1.0, 1.0]), 1e-5)
    assert np.linalg.matrix_rank(two_zeros, tol=1e-8) == 2


def test_shift_is_exact_and_scale_is_not_when_epsilon_is_positive():
    params, H = make_params(1, eps=1e-5)
    a = 0.7
    H2, _cache = block(H, params)
    shifted = H.copy()
    shifted[1] = shifted[1] + a * np.ones(params["d"])
    H2_shifted, _cache = block(shifted, params)
    expect = np.zeros_like(H)
    expect[1] = a * np.ones(params["d"])
    assert H2_shifted - H2 == pytest.approx(expect, abs=1e-12)
    assert response(H, params, 1, a * np.ones(params["d"])) == pytest.approx(0.0, abs=1e-12)
    read = params["read_token"]
    got = response(H, params, read, a * np.ones(params["d"]))
    assert got == pytest.approx(a * float(np.sum(params["w_read"])), abs=1e-12)
    scaled = H.copy()
    scaled[1] = 2.0 * scaled[1]
    H2_scaled, _cache = block(scaled, params)
    skip_only = np.zeros_like(H)
    skip_only[1] = scaled[1] - H[1]
    assert np.max(np.abs((H2_scaled - H2) - skip_only)) > 1e-2


def test_zero_bias_addition_cannot_make_both_deletions_necessary():
    c1, c2 = 1.0, -0.5
    assert c1 + c2 == pytest.approx(0.5)
    assert c1 > 0
    assert c2 < 0


def test_biased_addition_makes_both_necessary_with_zero_interaction():
    b, c1, c2 = -1.5, 1.0, 1.0
    assert b + c1 + c2 == pytest.approx(0.5)
    assert b + c2 == pytest.approx(-0.5)
    assert b + c1 == pytest.approx(-0.5)

    def additive(u, v):
        return b + u + v

    interaction = additive(1.0, 1.0) - additive(1.0, 0.0) - additive(0.0, 1.0) + additive(0.0, 0.0)
    assert interaction == pytest.approx(0.0)


def test_block_directional_derivative_matches_finite_difference():
    params, H = make_params(0)
    delta = np.array([0.2, -0.1, 0.05, 0.3])
    dH = token_direction(params["T"], params["d"], 1, delta)
    analytic = margin_jvp(H, dH, params)
    step = 1e-6
    numeric = response(H, params, 1, step * delta) / step
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6)


def test_value_only_linearization_can_have_the_wrong_sign():
    params, H = make_params(16)
    delta = np.array([
        0.01206307677054339,
        -0.8987843131414323,
        0.00171581962876215,
        -0.22913906964580683,
    ])
    dH = token_direction(params["T"], params["d"], 1, delta)
    total = margin_jvp(H, dH, params, value_only=False)
    value_only = margin_jvp(H, dH, params, value_only=True)
    assert total == pytest.approx(0.02019538349844552, rel=1e-8)
    assert value_only == pytest.approx(-0.008580691798845158, rel=1e-8)
    assert total * value_only < 0


def test_zero_first_order_response_need_not_be_a_zero_finite_response():
    params, H = make_params(0)
    d = params["d"]
    gradient = np.array([
        margin_jvp(H, token_direction(params["T"], d, 1, np.eye(d)[i]), params)
        for i in range(d)
    ])
    direction = np.array([gradient[1], -gradient[0], 0.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    linear = margin_jvp(H, token_direction(params["T"], d, 1, direction), params)
    finite = response(H, params, 1, 0.2 * direction)
    assert linear == pytest.approx(0.0, abs=1e-9)
    assert finite == pytest.approx(-0.6292659314667218, rel=1e-8)
    small = response(H, params, 1, 1e-3 * direction)
    assert small / 1e-6 == pytest.approx(-14.04747949363, rel=1e-6)


def test_same_projection_opposite_intervention_signs():
    params, H = make_params(16)
    other = H.copy()
    other[0, 3] += 2.0
    assert np.allclose(np.delete(H, 3, axis=1), np.delete(other, 3, axis=1))
    delta = np.array([0.4, -1.2, 0.3, 0.5])
    first = response(H, params, 1, delta)
    second = response(other, params, 1, delta)
    assert first < -0.1
    assert second > 0.1


def test_head_derivatives_add_and_token_edits_need_not():
    params, H = make_params(4)
    dH = token_direction(params["T"], params["d"], 1, np.array([0.2, -0.4, 0.1, 0.3]))
    _H2, cache = block(H, params)
    dX = np.zeros_like(H)
    for i in range(H.shape[0]):
        dX[i] = layer_norm_jacobian(H[i], params["gamma1"], params["eps"]) @ dH[i]
    both = attention_jvp(cache["X"], dX, params)
    pieces = []
    width = params["W_Q"].shape[-1]
    n_heads = params["W_Q"].shape[0]
    for head in range(n_heads):
        altered = {}
        for key, value in params.items():
            altered[key] = value.copy() if isinstance(value, np.ndarray) else value
        mask = np.ones_like(params["W_O"])
        mask[head * width:(head + 1) * width] = 0.0
        altered["W_O"] = params["W_O"] * mask
        pieces.append(attention_jvp(cache["X"], dX, altered))
    assert both == pytest.approx(sum(pieces), abs=1e-10)
    e1 = np.array([0.3, -0.2, 0.4, 0.1])
    e2 = np.array([-0.2, 0.5, 0.1, -0.4])
    m0, _, _ = margin(H, params)
    m1, _, _ = margin(intervene(H, 1, e1), params)
    m2, _, _ = margin(intervene(H, 2, e2), params)
    m12, _, _ = margin(intervene(intervene(H, 1, e1), 2, e2), params)
    interaction = (m12 - m0) - (m1 - m0) - (m2 - m0)
    assert interaction == pytest.approx(-0.0628083292402164, rel=1e-8)
    delta = np.array([0.2, -0.4, 0.1, 0.3])
    joint = response(H, params, 1, delta)
    singles = []
    for head in range(n_heads):
        altered = {}
        for key, value in params.items():
            altered[key] = value.copy() if isinstance(value, np.ndarray) else value
        mask = np.ones_like(params["W_O"])
        mask[head * width:(head + 1) * width] = 0.0
        altered["W_O"] = params["W_O"] * mask
        singles.append(response(H, altered, 1, delta))
    assert joint - sum(singles) == pytest.approx(-0.007621108909372204, rel=1e-8)


def test_attention_directional_derivative_matches_finite_differences():
    params, H = make_params(3)
    _H2, cache = block(H, params)
    delta = np.array([0.15, -0.25, 0.05, 0.1])
    dH = token_direction(params["T"], params["d"], 1, delta)
    dX = np.vstack([
        layer_norm_jacobian(H[i], params["gamma1"], params["eps"]) @ dH[i]
        for i in range(H.shape[0])
    ])
    analytic = attention_jvp(cache["X"], dX, params)
    step = 1e-6
    X_pos = cache["X"] + step * dX
    X_neg = cache["X"] - step * dX
    forward, _patterns, _dk = attention(X_pos, params)
    backward, _patterns, _dk = attention(X_neg, params)
    numeric = (forward - backward) / (2 * step)
    assert analytic == pytest.approx(numeric, abs=1e-6)


def test_rank_one_transport_can_disagree_when_the_margin_does_not():
    params, H = make_params(1, eps=1e-5)
    base, _H2, cache = margin(H, params)
    shifted = H.copy()
    shifted[1] = shifted[1] + 0.7 * np.ones(params["d"])
    moved, _H2s, cache_s = margin(shifted, params)
    assert moved == pytest.approx(base, abs=1e-12)
    source = H[0]
    target = cache["H1"][0]
    rho = np.outer(target, source) / float(source @ source)

    def disagreement(state, H1):
        return np.linalg.norm(H1 - (rho @ state.T).T)

    base_gap = disagreement(H, cache["H1"])
    shifted_gap = disagreement(shifted, cache_s["H1"])
    assert base_gap == pytest.approx(1.257891910391803, rel=1e-8)
    assert shifted_gap == pytest.approx(1.7969065901504973, rel=1e-8)
    assert abs(shifted_gap - base_gap) > 0.5


def test_exact_equations_hold_while_margins_differ_across_inputs():
    first_params, first_H = make_params(0)
    second_params, second_H = make_params(2)
    first, _H2, first_cache = margin(first_H, first_params)
    second, _H2b, second_cache = margin(second_H, second_params)
    assert first > 0.4
    assert second < -1.0
    again, _patterns, _dk = attention(first_cache["X"], first_params)
    assert again == pytest.approx(first_cache["A"], abs=1e-12)
    again_b, _patterns, _dk = attention(second_cache["X"], second_params)
    assert again_b == pytest.approx(second_cache["A"], abs=1e-12)


def test_a_constant_component_factors_out_of_every_finite_edit():
    params, H = make_params(9)
    width = params["d"]
    edits = [
        (0, np.array([0.2, -0.4, 0.1, 0.3])),
        (1, np.array([-0.5, 0.2, 0.4, -0.1])),
        (2, np.array([0.3, 0.1, -0.6, 0.2])),
    ]
    constant = 0.8 * np.ones(width)
    base = response_of(H, params, edits)
    for token in range(params["T"]):
        both = response_of(H, params, edits + [(token, constant)])
        kappa = 0.8 * float(np.sum(params["w_read"])) if token == params["read_token"] else 0.0
        assert both == pytest.approx(base + kappa, abs=1e-9)
    centered = np.array([0.4, -0.2, 0.5, -0.7])
    centered = centered - centered.mean()
    same = response_of(H, params, [(1, centered), (1, constant)])
    only_centered = response_of(H, params, [(1, centered)])
    assert same == pytest.approx(only_centered, abs=1e-9)


def response_of(H, params, edits):
    state = H
    for token, delta in edits:
        state = intervene(state, token, delta)
    base, _, _ = margin(H, params)
    moved, _, _ = margin(state, params)
    return moved - base


def test_recomputed_attention_matches_the_structural_equation():
    params, H = make_params(2)
    _H2, cache = block(H, params)
    again, _patterns, _dk = attention(cache["X"], params)
    assert again == pytest.approx(cache["A"], abs=1e-12)
