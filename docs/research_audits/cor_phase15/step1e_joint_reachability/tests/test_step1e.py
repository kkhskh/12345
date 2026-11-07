"""Implementation checks for the joint-reachability theorems.

The strict bounds are proved in the notes. These tests check that both block
implementations realize the scalar reduction, that the special-gain samples
stay under the proved bound, and that the counterexamples have the stated
geometry. Sampling is not the proof.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proofs.block import (  # noqa: E402
    forward,
    line_delta,
    normalized_response,
    pathway_increments,
    raw_response,
    structural_params,
    value_matrix,
    zero_state,
)
from proofs.certificate import (  # noqa: E402
    joint_branch_certificate,
    separate_branch_certificate,
    special_gain_constants,
)
from proofs.examples import (  # noqa: E402
    fiber_pair,
    moving_pattern_params,
    product_ball_ceiling,
    same_attention_different_mlp,
    singular_value_block,
    value_map_is_isotropic,
)
from proofs.line_model import (  # noqa: E402
    attention_derivative,
    attention_readout,
    branch_readout,
    full_raw_response,
    lam,
    mlp_derivative,
    separate_marginal_upper,
)
from proofs.reference_block import (  # noqa: E402
    directional_derivative,
    forward as reference_forward,
    raw_response as reference_response,
)
from proofs.support import linear_ball_supports, positively_parallel, triangle_gap_identity  # noqa: E402
from proofs.sympy_identities import run as run_sympy  # noqa: E402


def _record(params, y0=0.0, y1=0.0):
    return {
        "mu": params["mu"],
        "alpha": params["alpha"],
        "s_gain": params["s_gain"],
        "eps": params["eps"],
        "y0": y0,
        "y1": y1,
    }


def test_implementations_agree_on_the_structural_block():
    params = structural_params()
    state = zero_state(params, coordinate=0.3)
    left = forward(state, params)
    right = reference_forward(state, params)
    for key in ("X", "A", "H1", "U", "M", "H2"):
        assert np.allclose(left[key], right[key], atol=1e-12)
    delta = line_delta(params, 0.2)
    assert raw_response(state, 2, delta, params) == pytest_approx(
        reference_response(state, 2, delta, params)
    )


def pytest_approx(value, tol=1e-10):
    class _Comparator:
        def __eq__(self, other):
            return abs(other - value) <= tol

        def __repr__(self):
            return f"approx({value})"

    return _Comparator()


def test_causal_pattern_is_uniform_and_both_heads_are_used():
    params = structural_params()
    state = zero_state(params, coordinate=0.2)
    result = forward(state, params)
    read = params["read_token"]
    for pattern in result["weights"]:
        assert np.allclose(pattern[read, : read + 1], 1.0 / params["T"])
        assert np.all(pattern[read, read + 1 :] == 0.0)
    assert value_map_is_isotropic(params) < 1e-12
    crippled = structural_params()
    crippled["W_O"][: params["d"], :] = 0.0
    full = forward(state, params)["A"]
    half = forward(state, crippled)["A"]
    assert np.linalg.norm(full - half) > 1e-8
    assert np.allclose(half, 0.5 * full, atol=1e-10)


def test_last_token_edit_does_not_move_earlier_rows():
    params = structural_params()
    state = zero_state(params)
    attention, mlp, _base, _moved = pathway_increments(
        state, params["read_token"], line_delta(params, 0.4), params
    )
    assert np.linalg.norm(attention[:2]) < 1e-12
    assert np.linalg.norm(mlp[:2]) < 1e-12


def test_scalar_reduction_matches_both_intervention_sites():
    params = structural_params(mu=0.5, alpha=1.0, s_gain=1.0)
    state = zero_state(params)
    for step in (1e-4, 0.01, 0.2, 1.0):
        for token, direct in ((params["read_token"], True), (0, False)):
            attention, mlp, _base, _moved = pathway_increments(
                state, token, line_delta(params, step), params
            )
            row = params["read_token"]
            got_a = float(params["w_read"] @ attention[row])
            got_b = float(params["w_read"] @ mlp[row])
            exp_a, exp_b, _sum = branch_readout(
                step, params["mu"], params["alpha"], params["s_gain"], 0.0, 0.0, params["eps"], direct
            )
            assert abs(got_a - exp_a) < 1e-10
            assert abs(got_b - exp_b) < 1e-10
            got_r = raw_response(state, token, line_delta(params, step), params)
            exp_r = full_raw_response(
                step,
                params["mu"],
                params["alpha"],
                params["s_gain"],
                0.0,
                0.0,
                params["eps"],
                direct,
                1.0,
            )
            assert abs(got_r - exp_r) < 1e-10


def test_sign_theorem_on_several_gains_and_bases():
    eps = 1e-5
    for mu, alpha, s_gain, y0, y1, end in (
        (0.5, 1.0, 1.0, 0.0, 0.0, 1.0),
        (2.0, 1.0, 2.0, 0.0, 0.0, 2.0),
        (0.5, 1.0, 0.3, 0.0, 0.0, 1.0),
        (1.0, 0.5, 1.5, -0.4, 0.2, 0.8),
        (0.4, 1.0, 0.4, 2.0, 2.0, 1.0),
    ):
        marginal = separate_marginal_upper(end, mu, alpha, s_gain, y0, y1, eps, True)
        joint_values = []
        previous_a, previous_b = 0.0, 0.0
        for step in np.linspace(0.0, end, 25)[1:]:
            attention, mlp, branch = branch_readout(step, mu, alpha, s_gain, y0, y1, eps, True)
            assert attention > previous_a
            assert mlp < previous_b
            assert attention_derivative(step, mu, y0, eps) > 0.0
            assert mlp_derivative(step, mu, alpha, s_gain, y0, y1, eps, True) < 0.0
            joint_values.append(branch)
            previous_a, previous_b = attention, mlp
        assert max(joint_values) < marginal - 1e-12


def test_special_gain_joint_maximum_is_zero_and_beats_the_marginal_sum():
    params = structural_params()
    constants = special_gain_constants()
    assert constants["lambda_threshold"] and constants["cubic_room"] and constants["tanh_gap"]
    record = _record(params)
    assert joint_branch_certificate(record) == 0.0
    attention_end = attention_readout(1.0, params["mu"], 0.0, params["eps"])
    separate = separate_branch_certificate(record, attention_end)
    assert separate > 0.7
    worst = 0.0
    for step in np.logspace(-8, 0, 40):
        _a, _b, branch = branch_readout(
            step, params["mu"], params["alpha"], params["s_gain"], 0.0, 0.0, params["eps"], True
        )
        worst = max(worst, branch)
        assert branch <= 1e-12
    assert worst <= 1e-12
    assert separate - joint_branch_certificate(record) == pytest_approx(attention_end, tol=1e-15)
    # The full raw response keeps the skip. Its product relaxation adds the
    # same positive attention marginal, so the gap remains exactly that marginal.
    full_values = [
        full_raw_response(step, 0.5, 1.0, 1.0, 0.0, 0.0, 1e-5, True, 1.0)
        for step in np.logspace(-6, 0, 30)
    ]
    assert max(full_values) <= 1.0 + 1e-12
    assert max(full_values) < 1.0 + separate


def test_negative_mlp_gain_attains_the_product_corner():
    end = 1.0
    attention, mlp, branch = branch_readout(end, 0.5, 1.0, -1.0, 0.0, 0.0, 1e-5, True)
    assert attention > 0.0 and mlp > 0.0
    assert branch == pytest_approx(attention + mlp, tol=1e-15)


def test_zero_attention_gain_makes_the_joint_set_a_product():
    _attention, mlp, branch = branch_readout(1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1e-5, True)
    assert _attention == 0.0
    assert branch == pytest_approx(mlp)


def test_linear_gap_is_strict_exactly_off_the_nonnegative_ray():
    rng = np.random.default_rng(2)
    first = rng.normal(size=(4, 4))
    second = rng.normal(size=(4, 4))
    readout = rng.normal(size=4)
    report = linear_ball_supports(first, second, readout, radius=1.7)
    assert report["gap"] > 0.0
    assert report["aligned"] is False
    aligned = linear_ball_supports(first, 2.0 * first, readout, radius=1.7)
    assert aligned["aligned"] is True
    assert aligned["gap"] < 1e-10
    zero = linear_ball_supports(first, np.zeros((4, 4)), readout, radius=1.7)
    assert zero["aligned"] is True
    assert zero["gap"] < 1e-10
    identity = triangle_gap_identity(report["attention_gradient"], report["mlp_gradient"])
    assert identity["gap"] == pytest_approx(report["gap"] / 1.7, tol=1e-10)
    assert identity["inner_defect"] > 0.0


def test_product_ball_is_not_the_joint_support():
    params = structural_params()
    state = zero_state(params)
    attention, mlp, _base, _moved = pathway_increments(
        state, params["read_token"], line_delta(params, 1.0), params
    )
    row = params["read_token"]
    # Independent balls large enough to contain the observed increments.
    ball = product_ball_ceiling(
        np.linalg.norm(attention[row]) + 0.1,
        np.linalg.norm(mlp[row]) + 0.1,
        params["w_read"],
        params["w_read"],
    )
    joint = float(params["w_read"] @ (attention[row] + mlp[row]))
    assert ball > joint + 0.5


def test_constant_fiber_does_not_change_pathway_increments():
    params = structural_params()
    params["d_ff"] = 2
    params["W1"] = np.array([[0.4, -0.2], [0.1, 0.7]])
    params["b1"] = np.zeros(2)
    params["W2"] = np.array([[0.3, -0.5], [0.2, 0.4]])
    params["b2"] = np.zeros(2)
    delta = line_delta(params, 0.3) + np.array([0.2, 0.4])
    a1, b1, a2, b2 = fiber_pair(params, params["read_token"], delta, shift=1.3)
    assert np.allclose(a1, a2, atol=1e-10)
    assert np.allclose(b1, b2, atol=1e-10)
    # The same frozen block, read by a functional that sees the constant.
    # Pathway increments match; the raw responses differ by the skip alone.
    params["w_read"] = np.ones(2) / np.sqrt(2.0)
    plain = raw_response(zero_state(params), params["read_token"], delta, params)
    shifted = raw_response(
        zero_state(params),
        params["read_token"],
        delta + 1.3 * np.ones(2),
        params,
    )
    skip_gap = float(params["w_read"] @ (1.3 * np.ones(2)))
    assert shifted - plain == pytest_approx(skip_gap, tol=1e-8)


def test_same_attention_increment_can_have_two_mlp_increments():
    found = same_attention_different_mlp()
    assert np.linalg.norm(found["a1"] - found["a2"]) < 1e-10
    assert np.linalg.norm(found["a1"]) > 1e-3
    assert np.linalg.norm(found["b1"] - found["b2"]) > 1e-3
    assert float(found["b1"] @ found["b2"]) < 0.0


def test_moving_pattern_leaves_the_scalar_formula():
    params = moving_pattern_params()
    state = zero_state(params)
    attention, _mlp, _base, _moved = pathway_increments(
        state, params["read_token"], line_delta(params, 0.25), params
    )
    expected = attention_readout(0.25, params["mu"], 0.0, params["eps"])
    got = float(structural_params()["w_read"] @ attention[params["read_token"]])
    assert abs(got - expected) > 1e-4


def test_final_normalization_is_not_the_raw_sum():
    params = structural_params()
    state = zero_state(params)
    step = 0.01
    raw = raw_response(state, params["read_token"], line_delta(params, step), params)
    normalized = normalized_response(state, params["read_token"], line_delta(params, step), params)
    assert raw < -0.2
    assert normalized == pytest_approx(lam(raw, params["eps"]), tol=1e-10)
    assert normalized < raw - 0.5


def test_earlier_token_skip_is_zero_while_the_branch_response_is_not():
    params = structural_params()
    state = zero_state(params)
    step = 1.0
    attention, mlp, branch = branch_readout(
        step, params["mu"], params["alpha"], params["s_gain"], 0.0, 0.0, params["eps"], False
    )
    assert attention > 0.5
    assert mlp < -0.5
    assert branch < -0.1
    got = raw_response(state, 0, line_delta(params, step), params)
    assert got == pytest_approx(branch, tol=1e-10)


def test_base_derivative_sign_does_not_survive_the_segment():
    slope = 1.0 + attention_derivative(0.0, 0.5, 0.0, 1e-5)
    slope += mlp_derivative(0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1e-5, True)
    assert slope < -1000.0
    end = full_raw_response(1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1e-5, True, 1.0)
    assert end > 0.5
    params = structural_params()
    numerical = directional_derivative(
        zero_state(params), params["read_token"], params["axis"], params, step=1e-7
    )
    assert numerical < -1000.0
    assert abs(numerical - slope) / abs(slope) < 1e-3


def test_saturated_base_keeps_a_strict_but_small_gap():
    marginal = separate_marginal_upper(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1e-5, True)
    _a, _b, branch = branch_readout(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1e-5, True)
    assert 0.0 < marginal < 1e-4
    assert branch < marginal
    assert marginal - branch < 1e-5


def test_off_axis_edit_is_outside_the_line_certificate():
    params = structural_params()
    state = zero_state(params)
    delta = np.array([1.0, 1.2])
    attention, mlp, _base, _moved = pathway_increments(state, params["read_token"], delta, params)
    attention_part = float(params["w_read"] @ attention[-1])
    mlp_part = float(params["w_read"] @ mlp[-1])
    # On this vector the branch sum is positive, so the special-gain bound 0
    # is false off the invariant line. The line theorem does not claim it.
    assert attention_part < 0.0
    assert mlp_part > 0.0
    assert attention_part + mlp_part > 0.1


def test_sympy_identities():
    report = run_sympy()
    assert report == {"lambda_derivative": True, "tanh_cubic": True, "triangle": True}


def test_certificate_source_does_not_call_the_block():
    source = (ROOT / "proofs" / "certificate.py").read_text()
    for banned in ("pathway_increments", "raw_response", "import block", "reference_block"):
        assert banned not in source


def test_singular_value_map_matches_its_projector():
    params = singular_value_block()
    mapped = value_matrix(params)
    assert np.allclose(mapped, params["mu"] * params["T"] * params["projector"])


def test_readout_orthogonal_to_the_line_kills_the_branch_gap():
    params = structural_params()
    params["w_read"] = np.ones(2) / np.sqrt(2.0)
    state = zero_state(params)
    attention, mlp, _base, _moved = pathway_increments(
        state, params["read_token"], line_delta(params, 1.0), params
    )
    assert abs(float(params["w_read"] @ attention[-1])) < 1e-10
    assert abs(float(params["w_read"] @ mlp[-1])) < 1e-10


def test_strong_gains_remain_strict_and_order_one():
    marginal = separate_marginal_upper(2.0, 2.0, 1.0, 2.0, 0.0, 0.0, 1e-5, True)
    values = [
        branch_readout(step, 2.0, 1.0, 2.0, 0.0, 0.0, 1e-5, True)[2]
        for step in np.linspace(0.0, 2.0, 40)
    ]
    assert marginal > 2.0
    assert max(values) > 1.0
    assert max(values) < marginal - 1.0


def test_positive_parallel_detector_accepts_a_zero_vector():
    assert positively_parallel(np.zeros(3), np.array([1.0, 0.0, 0.0]))
