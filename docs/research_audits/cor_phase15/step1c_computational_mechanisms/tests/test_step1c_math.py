"""Executable checks for the Step 1C corrections and the block derivations.

A passing test shows that this implementation matches the stated finite case.
It is not a proof of a general theorem, and it is not a Transformer experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))

from step1c_math import (  # noqa: E402
    attention,
    distance_D,
    interaction_at_zero,
    layer_norm,
    layer_norm_jacobian,
    linear_residual,
    mix_values,
    mlp,
    mixed_partial_at_zero,
    obstruction_O,
    relu,
    rms_norm,
)

TOL = 1e-9


def test_error1_separable_square_has_zero_interaction():
    def F(u, v):
        return u * u + v * v

    for u in (-2.0, -0.5, 0.0, 0.3, 1.5):
        for v in (-1.0, 0.0, 0.7, 2.0):
            assert interaction_at_zero(F, u, v) == pytest.approx(0.0, abs=TOL)
    assert mixed_partial_at_zero(F) == pytest.approx(0.0, abs=1e-6)


def test_error1_cubic_vanishes_at_one_pair_and_not_another():
    def F(u, v):
        return (u + v) ** 3

    # I(u,v) = 3uv(u+v). Nonlinear, and not separable.
    assert interaction_at_zero(F, 1.0, -1.0) == pytest.approx(0.0, abs=TOL)
    assert interaction_at_zero(F, 1.0, 1.0) == pytest.approx(6.0, abs=TOL)
    # The mixed partial at the baseline is 0, so this pair does not by itself
    # separate "nonzero Hessian" from "zero finite interaction".
    assert mixed_partial_at_zero(F) == pytest.approx(0.0, abs=1e-5)


def test_error1_nonzero_baseline_hessian_does_not_force_every_finite_I():
    def F(u, v):
        return u * v - (u * u) * v - u * (v * v)

    assert mixed_partial_at_zero(F) == pytest.approx(1.0, abs=1e-5)
    assert interaction_at_zero(F, 2.0, -1.0) == pytest.approx(0.0, abs=TOL)
    assert interaction_at_zero(F, 1.0, 1.0) == pytest.approx(-1.0, abs=TOL)


def test_error1_channels_that_do_not_mix_stay_separable():
    def F(u, v):
        return np.tanh(u) + u * u * u + np.cos(v)

    assert interaction_at_zero(F, 0.4, -1.2) == pytest.approx(0.0, abs=TOL)


def test_error2_changing_only_M_changes_D_and_not_O():
    s = np.array([1.0, 0.0])
    d = np.array([1.0, -1.0])
    ds = np.array([d @ s])
    W = np.array([[1.0]])
    ker = np.array([1.0, 1.0])
    O = obstruction_O(ds, W)
    assert O == pytest.approx(1.0)
    D_identity = distance_D(s, ker, np.eye(2))
    D_other = distance_D(s, ker, np.diag([10.0, 0.1]))
    assert D_identity == pytest.approx(0.5)
    assert D_other == pytest.approx(10.0 / 101.0)
    assert obstruction_O(ds, W) == pytest.approx(O)
    # Changing only W changes O and leaves D alone.
    assert obstruction_O(ds, np.array([[4.0]])) == pytest.approx(4.0)
    assert distance_D(s, ker, np.eye(2)) == pytest.approx(D_identity)


def test_error3_fixed_readout_need_not_keep_its_sign_under_layernorm():
    h = np.array([1.0, 0.0])
    w = np.array([1.0, 2.0])
    before = float(w @ h)
    after = float(w @ layer_norm(h, eps=0.0))
    assert before == pytest.approx(1.0)
    assert after == pytest.approx(-1.0)
    assert before * after < 0
    # A small eps does not restore the sign.
    assert float(w @ layer_norm(h, eps=1e-5)) < 0


def test_error3_plain_rmsnorm_preserves_linear_sign_and_gamma_need_not():
    h = np.array([1.0, 3.0])
    w = np.array([1.0, -0.2])
    before = float(w @ h)
    after = float(w @ rms_norm(h, eps=0.0))
    assert before > 0 and after > 0
    assert np.sign(before) == np.sign(after)
    h_gamma = np.array([1.0, 1.0])
    # w·h = 0.8, and γ ⊙ h / rms reverses it.
    assert float(w @ h_gamma) == pytest.approx(0.8)
    flipped = float(w @ rms_norm(h_gamma, gamma=np.array([-1.0, 1.0])))
    assert flipped == pytest.approx(-1.2)


def test_error3_layernorm_is_not_an_invertible_coordinate_change():
    h = np.array([1.0, -2.0, 0.5])
    assert np.allclose(layer_norm(h), layer_norm(3.0 * h + 4.0))
    # Dual reparameterization preserves the pairing. LayerNorm is not that map.
    A = np.array([[2.0, 0.5], [0.0, 0.5]])
    h2 = np.array([1.0, 0.0])
    w2 = np.array([0.3, -0.7])
    w_dual = np.linalg.solve(A.T, w2)
    assert float(w2 @ h2) == pytest.approx(float(w_dual @ (A @ h2)))
    # The same fixed w, applied after LayerNorm, is a different number.
    # LN([1, 0]) = [1, -1], so the pairing moves from 0.3 to 1.0.
    assert float(w2 @ layer_norm(h2)) == pytest.approx(1.0)
    assert float(w2 @ h2) != pytest.approx(float(w2 @ layer_norm(h2)))


def test_error3_jacobian_of_layernorm_is_singular():
    h = np.array([1.0, -0.25, 0.5])
    J = layer_norm_jacobian(h)
    ones = np.ones(3)
    centered = h - h.mean()
    assert np.linalg.norm(J @ ones) < 1e-6
    assert np.linalg.norm(J @ centered) < 1e-6
    # d=3 still has a transverse direction, so J is not the zero map.
    assert np.linalg.norm(J) > 1e-3


def test_readout_orthogonal_to_ones_keeps_its_sign_under_plain_layernorm():
    # w·LN(h) = (w·h)/σ when 1·w = 0, γ = 1, β = 0, and σ > 0.
    h = np.array([1.0, 0.0])
    w = np.array([1.0, -1.0])
    assert float(np.ones(2) @ w) == pytest.approx(0.0)
    before = float(w @ h)
    after = float(w @ layer_norm(h, eps=0.0))
    assert before == pytest.approx(1.0)
    assert after == pytest.approx(2.0)
    assert np.sign(before) == np.sign(after)


def test_error3_learned_affine_can_flip_a_sign_that_plain_layernorm_kept():
    h = np.array([1.0, 0.0])
    w = np.array([1.0, 0.0])
    plain = float(w @ layer_norm(h))
    learned = float(w @ layer_norm(h, gamma=np.array([-2.0, 1.0]), beta=np.array([0.0, 0.0])))
    assert float(w @ h) > 0
    assert plain > 0
    assert learned < 0


def test_error4_kernel_of_one_projection_need_not_preserve_the_residual():
    # r(h) = P_v h. δ is in ker P_u and moves r.
    P_u = np.array([[1.0, 0.0]])
    P_v = np.array([[0.0, 1.0]])
    A = np.eye(2)
    rho = np.array([[0.0]])
    h = np.array([0.4, -0.2])
    delta = np.array([0.0, 1.0])
    assert float(np.asarray(P_u @ delta).reshape(-1)[0]) == pytest.approx(0.0)
    before = linear_residual(h, P_u, P_v, A, rho)
    after = linear_residual(h + delta, P_u, P_v, A, rho)
    assert float(np.asarray(after - before).reshape(-1)[0]) == pytest.approx(1.0)


def test_error4_the_full_measurement_kernel_does_leave_the_residual_fixed():
    P_u = np.array([[1.0, 0.0]])
    P_v = np.array([[0.0, 1.0]])
    A = np.eye(2)
    rho = np.zeros((1, 1))
    # Measurement operator is P_v A - rho P_u = [[0, 1]].
    measurement = P_v @ A - rho @ P_u
    delta = np.array([3.0, 0.0])
    assert np.allclose(measurement @ delta, 0.0)
    h = np.array([0.4, -0.2])
    assert np.allclose(
        linear_residual(h + delta, P_u, P_v, A, rho),
        linear_residual(h, P_u, P_v, A, rho),
    )


def test_n1_narrow_counterexample_still_stands():
    # Same Q, opposite linear responses. The statistic is the full measurement.
    Q = np.array([[1.0, 0.0]])
    h = np.array([0.2, 0.0])
    delta = np.array([0.0, 1.0])
    assert np.allclose(Q @ (h + delta), Q @ h)
    w = np.array([0.0, 1.0])
    assert float(w @ delta) > 0
    assert float((-w) @ delta) < 0


def test_value_only_attention_interaction_is_exactly_zero():
    A = np.array([[1.0, 0.0], [0.5, 0.5]])
    V = np.array([[1.0], [1.0]])
    W_O = np.array([[1.0]])

    def y(dv0, dv1):
        Vp = V.copy()
        Vp[0, 0] += dv0
        Vp[1, 0] += dv1
        return float(mix_values(A, Vp, W_O)[1, 0])

    assert interaction_at_zero(y, 1.0, 1.0) == pytest.approx(0.0, abs=TOL)
    assert interaction_at_zero(y, -0.3, 2.0) == pytest.approx(0.0, abs=TOL)


def test_residual_intervention_moves_qkv_together_and_can_interact():
    W = np.array([[1.0]])

    def token1(dx0, dx1):
        X = np.array([[1.0 + dx0], [1.0 + dx1]])
        Y, A, _V = attention(X, W, W, W, W, causal=True)
        assert A[0, 1] == pytest.approx(0.0)
        return float(Y[1, 0])

    both = token1(1.0, 1.0)
    only0 = token1(1.0, 0.0)
    only1 = token1(0.0, 1.0)
    base = token1(0.0, 0.0)
    assert interaction_at_zero(token1, 1.0, 1.0) == pytest.approx(both - only0 - only1 + base)
    assert abs(both - only0 - only1 + base) > 0.1


def test_query_bias_keeps_scores_dependent_on_keys_when_W_Q_is_zero():
    # Q = b_Q, not X W_Q. This is not the map in test_zero_query_map_makes_residual_edits_additive.
    def token1(dx0, dx1):
        x0 = 0.2 + dx0
        x1 = 0.7 + dx1
        logits = np.array([1.0 * x0, 1.0 * x1])
        weights = np.exp(logits - logits.max())
        weights = weights / weights.sum()
        return float(weights[0] * x0 + weights[1] * x1)

    assert abs(interaction_at_zero(token1, 0.5, -0.25)) > 1e-3


def test_constant_pattern_through_layernorm_is_not_additive_inside_one_token():
    def mixed(du, dv):
        h = np.array([1.0, 0.2, -0.4])
        step = np.zeros(3)
        step[0] = du
        step[1] = dv
        return layer_norm(h + step, eps=0.0)

    base = mixed(0.0, 0.0)
    both = mixed(0.3, 0.5)
    only_u = mixed(0.3, 0.0)
    only_v = mixed(0.0, 0.5)
    assert np.linalg.norm(both - only_u - only_v + base) > 1e-3


def test_constant_pattern_layernorm_is_additive_across_tokens():
    def token_output(d0, d1):
        # A is constant. Each token is normalized on its own.
        a = np.array([0.25, 0.75])
        x0 = layer_norm(np.array([0.2 + d0, -0.1]), eps=0.0)
        x1 = layer_norm(np.array([0.5, 0.4 + d1]), eps=0.0)
        return float(a[0] * x0[0] + a[1] * x1[0])

    assert interaction_at_zero(token_output, 0.3, -0.2) == pytest.approx(0.0, abs=TOL)


def test_zero_query_map_makes_residual_edits_additive():
    W_Q = np.array([[0.0]])
    W_K = np.array([[1.0]])
    W_V = np.array([[1.0]])
    W_O = np.array([[1.0]])

    def token1(dx0, dx1):
        X = np.array([[0.2 + dx0], [0.7 + dx1]])
        Y, A, _V = attention(X, W_Q, W_K, W_V, W_O, causal=True)
        # Scores do not depend on X. Token 1's pattern is uniform on the past.
        assert A[1, 0] == pytest.approx(0.5)
        assert A[1, 1] == pytest.approx(0.5)
        return float(Y[1, 0])

    assert interaction_at_zero(token1, 0.5, -0.25) == pytest.approx(0.0, abs=TOL)


def test_mlp_relu_is_not_automatically_an_interaction():
    # Separate channels: nonlinear, four-term interaction 0.
    def separate(u, v):
        return float(relu(np.array([u])).reshape(-1)[0] + relu(np.array([v])).reshape(-1)[0])

    assert interaction_at_zero(separate, 1.0, -2.0) == pytest.approx(0.0)

    # Shared preactivation. Positive orthant is linear, so I(1,1)=0.
    # The pair (1,-2) crosses the kink.
    def shared(u, v):
        hidden = relu(np.array([u + v]))
        return float(hidden[0])

    assert interaction_at_zero(shared, 1.0, 1.0) == pytest.approx(0.0)
    assert interaction_at_zero(shared, 1.0, -2.0) == pytest.approx(-1.0)


def test_same_projected_residual_different_attention():
    # P keeps coordinate 0. W_K reads coordinate 1, which P discards.
    # Coordinate 0 is the projected coordinate and also the value feature.
    # Coordinate 1 is discarded by P and read by Q and K.
    # The two states agree on every value, and disagree on where attention lands.
    P = np.array([[1.0, 0.0]])
    W_Q = np.array([[0.0], [1.0]])
    W_K = np.array([[0.0], [1.0]])
    W_V = np.array([[1.0], [0.0]])
    W_O = np.array([[1.0]])
    Xa = np.array([[1.0, 0.0], [0.0, 0.0]])
    Xb = np.array([[1.0, 0.0], [0.0, 4.0]])
    assert np.allclose(Xa @ P.T, Xb @ P.T)
    Ya, Aa, Va = attention(Xa, W_Q, W_K, W_V, W_O, causal=True)
    Yb, Ab, Vb = attention(Xb, W_Q, W_K, W_V, W_O, causal=True)
    assert np.allclose(Va, Vb)
    assert not np.allclose(Aa[1], Ab[1])
    assert float(Ya[1, 0]) == pytest.approx(0.5)
    assert float(Yb[1, 0]) == pytest.approx(0.0, abs=1e-6)


def test_a_relation_collapses_pairs_that_the_tuple_separates():
    def relation(zu, zv):
        return zv - zu

    assert relation(0.0, 0.0) == relation(1.0, 1.0)
    # The downstream value is zv. The relation does not determine it.
    assert 0.0 != 1.0


def test_recomputed_structural_residual_is_identically_zero():
    def f(parent):
        return np.tanh(parent)

    def residual_after_recompute(parent, delta):
        parent_do = parent + delta
        child = f(parent_do)
        return child - f(parent_do)

    for delta in (-1.5, 0.0, 0.4, 2.0):
        assert residual_after_recompute(np.array([0.3, -0.7]), delta) == pytest.approx(0.0)
    # The response is not constant on that same family.
    def response(delta):
        return float(np.tanh(0.3 + delta) - np.tanh(0.3))

    assert response(0.5) != pytest.approx(response(-0.5))


def test_manufactured_compatibility_zeroes_one_pair():
    z_u = np.array([2.0, 0.0])
    z_v = np.array([0.0, 5.0])
    # One-point least-squares transport. It is fit to this pair.
    rho = np.outer(z_v, z_u) / float(z_u @ z_u)
    assert np.allclose(rho @ z_u, z_v)
    # A second pair with the same z_u direction and a different z_v is not fit.
    other = np.array([1.0, 1.0])
    assert not np.allclose(rho @ z_u, other)


def test_affine_linearization_keeps_the_offset():
    def f(h):
        return 5.0 * h[0] + 3.0

    h0 = np.array([2.0])
    delta = np.array([0.1])
    approx = f(h0) + 5.0 * delta[0]
    # The derivative applied to the absolute state drops the constant term.
    absolute = 5.0 * (h0[0] + delta[0])
    assert approx == pytest.approx(float(f(h0 + delta)))
    assert absolute != pytest.approx(float(f(h0 + delta)))

    def square(h):
        return h[0] ** 2 + 3.0

    # At δ=1 the remainder of the square is δ^2, so the first-order formula is short by 1.
    step = np.array([1.0])
    first_order = square(h0) + (2.0 * h0[0]) * step[0]
    assert float(square(h0 + step)) == pytest.approx(first_order + 1.0)


def test_three_intervention_semantics_differ():
    def branch(h):
        return np.array([np.tanh(h[0])])

    h = np.array([0.2])
    delta = np.array([0.5])
    total = (h + delta) + branch(h + delta)
    skip_only = (h + delta) + branch(h)
    imposed = np.array([7.0])
    assert not np.allclose(total, skip_only)
    assert not np.allclose(total, imposed)


def test_vector_agreement_is_not_readout_redundancy():
    w = np.array([1.0, 0.0])
    p1 = np.array([1.0, 100.0])
    p2 = np.array([1.0, -100.0])
    # Large disagreement, equal readout contributions.
    assert np.linalg.norm(p1 - p2) > 10
    assert float(w @ p1) == pytest.approx(float(w @ p2))
    # The same vector copied twice is an agreement, and both copies can be wrong.
    q = np.array([-2.0, 0.0])
    assert float(w @ q) == pytest.approx(float(w @ q.copy()))
    assert float(w @ q) < 0


def test_single_ablation_robustness_needs_the_remaining_inner_product():
    def margin(c1, c2):
        return c1 + c2

    # Either path alone keeps the sign.
    assert margin(1.0, 0.0) > 0 and margin(0.0, 1.0) > 0
    # Agreement on the wrong side is stable and incorrect.
    assert margin(-1.0, -1.0) < 0
    assert margin(-1.0, 0.0) < 0
    # Opposite contributions: ablating one reverses the decision.
    assert margin(1.0, -0.5) > 0
    assert margin(0.0, -0.5) < 0


def test_function_preserving_rescale_changes_an_internal_score():
    W1 = np.array([[1.5, -0.25]])
    W2 = np.array([[0.8], [0.1]])
    x = np.array([0.4, -0.2])
    h = relu(W1 @ x)
    y = W2 @ h
    h_scaled = relu((2.0 * W1) @ x)
    y_scaled = (W2 / 2.0) @ h_scaled
    assert np.allclose(y, y_scaled)
    assert not np.allclose(h, h_scaled)


def test_block_mlp_matches_the_stated_formula():
    h = np.array([0.25, -1.0])
    y = mlp(
        h,
        W1=np.array([[1.0, 1.0], [0.0, 1.0]]),
        b1=np.array([0.0, 0.1]),
        W2=np.eye(2),
        b2=np.array([0.0, 0.0]),
    )
    pre = np.array([1.0, 1.0]) @ h
    assert y[0] == pytest.approx(float(relu(np.array([pre]))[0]))
    assert y[1] == pytest.approx(float(relu(np.array([-1.0 + 0.1]))[0]))
