"""Independent checks for the Step 1D certificates.

Route A is the vectorized forward and the analytic JVP.
Route B is the loop forward and central differences.
Route C is SymPy, which never calls the block.
The certificate module never calls a forward or a derivative.
"""

from __future__ import annotations

import numpy as np
import pytest

from proofs.block_forward import (
    block,
    layer_norm,
    make_params,
    margin,
    orthogonal_unit,
    project_outputs_off_readout,
    qkv_smallest_singular_value,
    response,
    scale_branch_outputs,
    structural_residual,
)
from proofs.block_jvp import branch_cancellation_weights, component_jvps, token_gradient
from proofs.block_reference import margin as margin_b
from proofs.block_reference import response as response_b
from proofs.block_reference import token_gradient as token_gradient_b
from proofs.certificates import (
    aligned_certificate,
    directional_certificate,
    final_readout_certificate,
    isotropic_certificate,
    output_scale_rate_threshold,
)
from proofs.sympy_identities import (
    layer_norm_scale_identity,
    softmax_two_point_norm,
    tanh_derivative_bound,
)


SEED_VECTOR = np.array([0.0, 1.0, 0.0, 0.0])


def _delta(params):
    return orthogonal_unit(params["w_read"], SEED_VECTOR)


def test_certificate_module_does_not_evaluate_the_response():
    source = open("proofs/certificates.py", encoding="utf-8").read()
    assert "block_forward" not in source
    assert "block_jvp" not in source
    assert "def response" not in source


def test_two_forwards_and_two_gradients_agree():
    params, state = make_params(0)
    analytic, _, _ = margin(state, params)
    looped, *_ = margin_b(state, params)
    assert analytic == pytest.approx(0.4644721976523223, rel=0, abs=1e-15)
    assert looped == pytest.approx(analytic, rel=0, abs=1e-12)
    delta = np.array([0.2, -0.4, 0.1, 0.05])
    assert response(state, params, 1, delta) == pytest.approx(
        response_b(state, params, 1, delta), rel=0, abs=1e-10
    )
    analytic_g = token_gradient(state, params, 2)
    finite_g = token_gradient_b(state, params, 2)
    assert np.linalg.norm(analytic_g - finite_g) < 1e-8


def test_recomputed_trace_has_zero_structural_residual_and_opposite_margins():
    params, state = make_params(0)
    assert structural_residual(state, params) == (0.0, 0.0)
    base, _, _ = margin(state, params)
    low, high = 0.0, 20.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        shifted = state.copy()
        shifted[2] = state[2] - mid * params["w_read"]
        value, _, _ = margin(shifted, params)
        if value > -base:
            low = mid
        else:
            high = mid
    shifted = state.copy()
    shifted[2] = state[2] - high * params["w_read"]
    other, _, _ = margin(shifted, params)
    assert structural_residual(shifted, params) == (0.0, 0.0)
    assert other == pytest.approx(-base, rel=0, abs=1e-12)
    assert base > 0 > other


def test_directional_certificate_dominates_the_response_across_regimes():
    params, state = make_params(0)
    rng = np.random.default_rng(4)
    deltas = [
        _delta(params),
        params["w_read"] / np.linalg.norm(params["w_read"]),
        np.array([0.2, -0.5, 0.1, 0.4]),
        np.ones(4),
    ]
    for alpha in (0.0, 1e-6, 1e-4, 1e-2, 1.0, 10.0):
        scaled = scale_branch_outputs(params, alpha)
        for token in (0, 1, 2):
            for delta in deltas:
                bound, _, _ = directional_certificate(scaled, delta, token)
                isotropic, _ = isotropic_certificate(scaled, np.linalg.norm(delta))
                aligned, _, _ = aligned_certificate(scaled, delta, token)
                got = abs(response(state, scaled, token, delta))
                assert got <= bound + 1e-8
                assert aligned <= bound + 1e-8
                assert bound <= isotropic + 1e-6
    for _ in range(4):
        state = rng.normal(size=(3, 4))
        delta = orthogonal_unit(params["w_read"], rng.normal(size=4))
        bound, _, _ = directional_certificate(params, delta, 2)
        assert abs(response(state, params, 2, delta)) <= bound + 1e-8


def test_small_output_scale_beats_the_cauchy_number_and_unit_scale_does_not():
    params, state = make_params(0)
    assert qkv_smallest_singular_value(params) > 0.5
    delta = _delta(params)
    assert abs(float(params["w_read"] @ delta)) < 1e-12
    assert not np.allclose(delta, np.full(4, delta.mean()))
    small = scale_branch_outputs(params, 1e-6)
    bound, _, rate = directional_certificate(small, delta, 2)
    gradient = token_gradient(state, small, 2)
    cauchy = float(np.linalg.norm(gradient) * np.linalg.norm(delta))
    finite = response(state, small, 2, delta)
    assert rate < 0.5
    assert bound < cauchy
    assert abs(finite) <= bound
    # Locked reconstruction. These are this construction's values, not the
    # inaccessible audit's printed floats.
    assert bound == pytest.approx(0.004051326443971923, rel=1e-12)
    assert cauchy == pytest.approx(0.8764350281115764, rel=1e-12)
    assert finite == pytest.approx(-1.3267974285913198e-07, rel=1e-9)
    full = scale_branch_outputs(params, 1.0)
    full_bound, _, full_rate = directional_certificate(full, delta, 2)
    full_gradient = token_gradient(state, full, 2)
    full_cauchy = float(np.linalg.norm(full_gradient))
    assert full_rate > 1.0
    assert full_bound > full_cauchy
    _, cache = block(state, full)
    edited = state.copy()
    edited[2] = edited[2] + delta
    _, moved = block(edited, full)
    assert np.linalg.norm(moved["A"] - cache["A"]) > 1e-2
    assert np.linalg.norm(moved["M"] - cache["M"]) > 1e-2


def test_rate_threshold_is_the_quadratic_root_and_separates_the_comparison():
    params, state = make_params(0)
    alpha_star = output_scale_rate_threshold(params, target=0.5)
    assert alpha_star == pytest.approx(0.00010387929173812708, rel=1e-12)
    delta = _delta(params)
    below = scale_branch_outputs(params, 0.5 * alpha_star)
    above = scale_branch_outputs(params, 2.0 * alpha_star)
    bound, _, rate = directional_certificate(below, delta, 2)
    gradient = token_gradient(state, below, 2)
    assert rate < 0.5
    assert np.linalg.norm(gradient - params["w_read"]) <= np.linalg.norm(params["w_read"]) * rate + 1e-8
    assert bound < np.linalg.norm(gradient)
    high_bound, _, high_rate = directional_certificate(above, delta, 2)
    assert high_rate > 0.5
    assert high_bound > np.linalg.norm(token_gradient(state, above, 2))


def test_output_kernel_is_exact_for_the_raw_readout_and_not_for_final_layernorm():
    params, state = make_params(0)
    killed = project_outputs_off_readout(params)
    assert qkv_smallest_singular_value(killed) == pytest.approx(
        qkv_smallest_singular_value(params), rel=0, abs=1e-12
    )
    assert np.linalg.norm(killed["W_O"] @ killed["w_read"]) < 1e-12
    assert np.linalg.norm(killed["W2"].T @ killed["w_read"]) < 1e-12
    delta = 2.5 * _delta(killed)
    got = response(state, killed, 2, delta)
    assert got == pytest.approx(float(killed["w_read"] @ delta), abs=1e-12)
    _, base = block(state, killed)
    edited = state.copy()
    edited[2] = edited[2] + _delta(killed)
    _, moved = block(edited, killed)
    assert np.linalg.norm(moved["A"] - base["A"]) > 0.1
    assert np.linalg.norm(moved["M"] - base["M"]) > 0.05
    gamma = np.ones(4)
    beta = np.zeros(4)

    def final_margin(matrix):
        output, _ = block(matrix, killed)
        normalized = layer_norm(output[2], gamma, beta, 1e-5)
        return float(killed["w_read"] @ normalized)

    final_gap = final_margin(edited) - final_margin(state)
    assert abs(final_gap) > 0.1
    generic, _ = final_readout_certificate(killed["w_read"], gamma, beta, 4)
    assert abs(final_gap) <= generic


def test_zero_branch_final_layernorm_moves_under_an_orthogonal_edit():
    params, state = make_params(0)
    scaled = scale_branch_outputs(params, 0.0)
    delta = _delta(scaled)
    assert response(state, scaled, 2, delta) == pytest.approx(0.0, abs=1e-15)
    gamma = np.ones(4)
    beta = np.zeros(4)

    def final_margin(matrix):
        output, _ = block(matrix, scaled)
        return float(scaled["w_read"] @ layer_norm(output[2], gamma, beta, 1e-5))

    edited = state.copy()
    edited[2] = edited[2] + delta
    gap = final_margin(edited) - final_margin(state)
    assert gap == pytest.approx(-0.32024254867419905, rel=1e-12)
    generic, radius = final_readout_certificate(scaled["w_read"], gamma, beta, 4)
    assert radius == pytest.approx(2.0)
    assert abs(gap) <= generic


def test_first_order_cancellation_does_not_control_the_finite_response():
    params, state = make_params(0)
    delta = _delta(params)
    updated, _before = branch_cancellation_weights(state, params, 2, delta)
    direction = np.zeros_like(state)
    direction[2] = delta
    after = component_jvps(state, direction, updated)
    assert abs(after["attention"] + after["mlp"]) < 1e-12
    assert abs(after["attention"]) > 1e-3
    assert abs(after["mlp"]) > 1e-3
    finite = response(state, updated, 2, 2.0 * delta)
    assert abs(finite) == pytest.approx(0.16901883579206622, rel=1e-10)
    assert abs(finite) > 0.1
    assert abs(after["total"] * 2.0) < 1e-12


def test_sign_flip_preserves_separate_bounds_and_changes_the_finite_response():
    params, state = make_params(0)
    delta = _delta(params)
    reflected = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in params.items()}
    w = reflected["w_read"]
    reflected["W2"] = reflected["W2"] - 2.0 * np.outer(w, w @ reflected["W2"]) / float(w @ w)
    bound, _, _ = directional_certificate(params, delta, 2)
    other, _, _ = directional_certificate(reflected, delta, 2)
    aligned, _, _ = aligned_certificate(params, delta, 2)
    aligned_other, _, _ = aligned_certificate(reflected, delta, 2)
    assert other == pytest.approx(bound, rel=0, abs=1e-9)
    assert aligned_other == pytest.approx(aligned, rel=0, abs=1e-9)
    first = response(state, params, 2, delta)
    second = response(state, reflected, 2, delta)
    assert abs(first - second) > 0.03


def test_component_derivatives_sum_to_the_total_derivative():
    params, state = make_params(0)
    delta = _delta(params)
    direction = np.zeros_like(state)
    direction[2] = delta
    parts = component_jvps(state, direction, params)
    assert parts["total"] == pytest.approx(parts["skip"] + parts["attention"] + parts["mlp"], abs=1e-12)
    gradient = token_gradient(state, params, 2)
    assert parts["total"] == pytest.approx(float(gradient @ delta), abs=1e-12)


def test_sympy_routes_do_not_use_the_numeric_block():
    scale = layer_norm_scale_identity()
    assert scale["positive_solutions"] == ["1"]
    assert "eps" in scale["factored"] and "lam" in scale["factored"]
    softmax = softmax_two_point_norm()
    assert softmax["equals_one_half"] is True
    tanh = tanh_derivative_bound()
    assert tanh["gap_from_one_minus_tanh_sq"] == "0"


def test_attention_and_layernorm_lipschitz_samples_stay_under_the_constants():
    from proofs.block_forward import attention_from_normalized
    from proofs.certificates import attention_factors, layer_norm_lipschitz, mlp_lipschitz

    params, _state = make_params(0)
    attention, _concat, radius, _heads = attention_factors(params)
    rng = np.random.default_rng(9)
    worst = 0.0
    for _ in range(25):
        left = rng.normal(size=(3, 4))
        right = rng.normal(size=(3, 4))
        for matrix in (left, right):
            for token in range(3):
                norm = np.linalg.norm(matrix[token])
                if norm > radius:
                    matrix[token] *= radius / norm
        out_left, _, _ = attention_from_normalized(left, params)
        out_right, _, _ = attention_from_normalized(right, params)
        gap = np.linalg.norm(out_right - out_left)
        step = np.linalg.norm(right - left)
        if step > 0:
            worst = max(worst, gap / step)
    assert worst <= attention
    first = layer_norm_lipschitz(params["gamma1"], params["eps"])
    worst_ln = 0.0
    for _ in range(20):
        left = rng.normal(size=4)
        right = left + rng.normal(size=4)
        gap = np.linalg.norm(
            layer_norm(right, params["gamma1"], params["beta1"], params["eps"])
            - layer_norm(left, params["gamma1"], params["beta1"], params["eps"])
        )
        worst_ln = max(worst_ln, gap / np.linalg.norm(right - left))
    assert worst_ln <= first
    mlp_bound = mlp_lipschitz(params)
    from proofs.block_forward import mlp_from_normalized

    worst_mlp = 0.0
    for _ in range(20):
        left = rng.normal(size=(3, 4))
        right = rng.normal(size=(3, 4))
        out_left, _, _ = mlp_from_normalized(left, params)
        out_right, _, _ = mlp_from_normalized(right, params)
        worst_mlp = max(
            worst_mlp,
            np.linalg.norm(out_right - out_left) / np.linalg.norm(right - left),
        )
    assert worst_mlp <= mlp_bound
