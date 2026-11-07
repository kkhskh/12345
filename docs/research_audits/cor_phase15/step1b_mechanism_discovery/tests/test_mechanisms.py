"""Step 1B synthetic checks.

These are finite systems. A passing test is arithmetic. It is not evidence that
a Transformer implements the mechanism, and it is not a Phase 2 result.
Randomized searches use seed 20261107. Failure to find a counterexample is not
a proof.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))

from mechanism_math import (  # noqa: E402
    TruncationSCM,
    affine_linearization,
    attention_mix,
    layer_norm,
    linear_path_contributions,
    majority_vote,
    pairwise_disagreement,
    residual_block,
    softmax,
)

TOL = 1e-9
SEED = 20261107


def test_step1a_ce01_reproduced():
    s = np.array([1.0, 0.5])
    residual = s[0] - s[1]
    o = residual**2
    assert o == pytest.approx(0.25)
    assert float(np.array([1.0, 0.0]) @ s) == pytest.approx(1.0)
    assert float(np.array([-1.0, 0.0]) @ s) == pytest.approx(-1.0)


def test_affine_offset_is_not_the_jacobian_times_the_state():
    def f(h):
        h = np.asarray(h, dtype=np.float64)
        return h**2

    h0 = np.array([1.0])
    delta = np.array([0.1])
    f0, J, affine = affine_linearization(f, h0, delta, eps=1e-8)
    assert f0[0] == pytest.approx(1.0)
    assert J[0, 0] == pytest.approx(2.0, abs=1e-5)
    jh = (J @ h0).reshape(-1)[0]
    assert jh == pytest.approx(2.0, abs=1e-5)
    assert jh != pytest.approx(f(h0).reshape(-1)[0])
    exact = f(h0 + delta)
    assert affine[0] == pytest.approx(1.0 + 2.0 * 0.1, abs=1e-5)
    assert exact[0] == pytest.approx(1.21)
    assert abs(affine[0] - exact[0]) > 1e-4


def test_lipschitz_bound_requires_factorization():
    def L_full(h):
        return float(h[0] + h[1])

    P = np.array([[1.0, 0.0]])
    h = np.array([1.0, 4.0])
    nearest_in_z = np.array([1.0, 0.0])
    gap_in_z = abs((P @ h)[0] - (P @ nearest_in_z)[0])
    gap_in_L = abs(L_full(h) - L_full(nearest_in_z))
    assert gap_in_z == pytest.approx(0.0)
    assert gap_in_L == pytest.approx(4.0)
    for K in (0.5, 1.0, 10.0, 1e6):
        assert gap_in_L > K * gap_in_z


def test_blind_statistic_cannot_sign_a_kernel_response():
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(40):
        d = 6
        p = rng.normal(size=(2, d))
        h = rng.normal(size=d)
        _, _, vh = np.linalg.svd(p, full_matrices=True)
        delta = vh[-1]
        q_before = p @ h
        q_after = p @ (h + delta)
        assert np.linalg.norm(q_after - q_before) < 1e-10
        w = rng.normal(size=d)
        dL = float(w @ delta)
        if abs(dL) < 1e-12:
            continue
        hits += 1
        assert np.sign(dL) == pytest.approx(-np.sign(float((-w) @ delta)))
        assert np.linalg.norm((p @ (h + delta)) - (p @ h)) < 1e-10
    assert hits >= 10


def test_joint_sufficiency_when_parts_collide():
    # L = h0 + h1. P1 keeps h0, P2 keeps h1.
    h_a = np.array([1.0, 4.0])
    h_b = np.array([1.0, -3.0])
    h_c = np.array([2.0, 4.0])
    P1 = np.array([[1.0, 0.0]])
    P2 = np.array([[0.0, 1.0]])

    def L(h):
        return float(h[0] + h[1])

    assert (P1 @ h_a) == pytest.approx(P1 @ h_b)
    assert L(h_a) != pytest.approx(L(h_b))
    assert (P2 @ h_a) == pytest.approx(P2 @ h_c)
    assert L(h_a) != pytest.approx(L(h_c))
    joint_a = np.concatenate([P1 @ h_a, P2 @ h_a])
    joint_b = np.concatenate([P1 @ h_b, P2 @ h_b])
    assert not np.allclose(joint_a, joint_b)
    assert L(h_a) == pytest.approx(float(joint_a.sum()))


def test_sufficiency_is_relative_to_an_intervention_family():
    P = np.array([[1.0, 0.0]])
    h = np.array([2.0, 5.0])
    w = np.array([0.0, 1.0])
    rowspace_delta = np.array([1.0, 0.0])
    kernel_delta = np.array([0.0, 1.0])
    L = lambda x: float(w @ x)
    assert L(h + rowspace_delta) == pytest.approx(L(h))
    assert L(h + kernel_delta) != pytest.approx(L(h))
    assert np.allclose(P @ (h + kernel_delta), P @ h)


def test_linear_ablation_is_the_path_inner_product():
    paths = np.array([[1.0, 0.0], [0.0, 3.0], [2.0, -1.0]])
    w = np.array([1.0, 1.0])
    pack = linear_path_contributions(paths, w)
    assert pack["contrib"][0] == pytest.approx(1.0)
    assert pack["contrib"][1] == pytest.approx(3.0)
    assert pack["L"] == pytest.approx(pack["contrib"].sum())
    assert pack["ablate"][1] == pytest.approx(pack["L"] - 3.0)


def test_linear_redundancy_is_not_agreement_of_the_vectors():
    # Two paths can be opposite as vectors and still add on the readout.
    p1 = np.array([1.0, 1.0])
    p2 = np.array([-1.0, 1.0])
    w = np.array([0.0, 1.0])
    assert float(p1 @ p2) == pytest.approx(0.0)
    assert float(w @ p1) == pytest.approx(float(w @ p2))
    L = float(w @ (p1 + p2))
    assert L == pytest.approx(2.0)
    assert float(w @ p1) == pytest.approx(1.0)


def test_cancellation_hides_large_paths_from_the_linear_readout():
    p1 = np.array([10.0, 0.0])
    p2 = np.array([-10.0, 0.0])
    w = np.array([1.0, 0.0])
    L = float(w @ (p1 + p2))
    assert L == pytest.approx(0.0)
    assert abs(float(w @ p1)) == pytest.approx(10.0)
    disagreement = np.linalg.norm(p1 - p2)
    assert disagreement == pytest.approx(20.0)


def test_nonlinear_gating_can_look_like_correction():
    def y(a, b):
        return float(np.tanh(a) + 0.001 * b)

    assert abs(y(2.0, 50.0) - y(2.0, 0.0)) < 0.1
    assert y(2.0, 50.0) > 0
    assert y(-2.0, 50.0) < 0


def test_majority_vote_is_real_correction_and_disagreement_does_not_rank_it():
    correct = np.array([1, 1, 1])
    one_flip = np.array([1, 1, 0])
    two_flip = np.array([1, 0, 0])
    all_wrong = np.array([0, 0, 0])
    assert majority_vote(correct) == 1
    assert majority_vote(one_flip) == 1
    assert majority_vote(two_flip) == 0
    assert majority_vote(all_wrong) == 0
    assert pairwise_disagreement(correct) == 0
    assert pairwise_disagreement(one_flip) == 2
    assert pairwise_disagreement(all_wrong) == 0
    # Zero disagreement is compatible with both the right and the wrong decision.


def test_pairwise_parent_equality_is_the_wrong_constraint():
    u, w, v = 1.0, 1.0, 2.0
    joint = u + w - v
    pair_u = u - v
    pair_w = w - v
    assert joint == pytest.approx(0.0)
    assert pair_u != pytest.approx(0.0)
    assert pair_w != pytest.approx(0.0)


def test_compatible_parents_wrong_aggregate_needs_an_external_target():
    # Both parents equal 1, aggregate 1, external target 0.
    u = w = v = 1.0
    assert u - v == pytest.approx(0.0)
    assert w - v == pytest.approx(0.0)
    target = 0.0
    assert v != pytest.approx(target)


def test_truncation_defines_nested_interventions():
    scm = TruncationSCM(f=lambda p: 2.0 * p, r=lambda c: c)
    p0 = 1.0
    _, _, y0 = scm.natural(p0)
    _, _, ya = scm.do_parent(p0, 3.0)
    _, c_b, yb = scm.do_child(p0, 10.0)
    _, c_ab, yab = scm.do_both(3.0, 10.0)
    assert y0 == pytest.approx(2.0)
    assert ya == pytest.approx(6.0)
    assert yb == pytest.approx(10.0)
    assert yab == pytest.approx(10.0)
    assert c_b == pytest.approx(c_ab)
    I = scm.interaction(p0, 3.0, 10.0)
    assert I == pytest.approx(10.0 - 6.0 - 10.0 + 2.0)
    assert I == pytest.approx(-4.0)


def test_linear_readout_of_a_sum_has_zero_interaction():
    def F(u, w):
        return 3.0 * (u + w)

    I = F(1.0, 2.0) - F(1.0, 0.0) - F(0.0, 2.0) + F(0.0, 0.0)
    assert I == pytest.approx(0.0)


def test_shared_square_bottleneck_has_interaction_two():
    def F(u, w):
        return (u + w) ** 2

    I = F(1.0, 1.0) - F(1.0, 0.0) - F(0.0, 1.0) + F(0.0, 0.0)
    assert I == pytest.approx(2.0)


def test_same_unintervened_residual_two_interaction_values():
    def residual(u, w, v):
        return u + w - v

    assert residual(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def I_of(F):
        return F(1.0, 1.0) - F(1.0, 0.0) - F(0.0, 1.0) + F(0.0, 0.0)

    assert I_of(lambda u, w: u + w) == pytest.approx(0.0)
    assert I_of(lambda u, w: (u + w) ** 2) == pytest.approx(2.0)


def test_layernorm_changes_euclidean_geometry():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    pre = np.linalg.norm(a - b)
    post = np.linalg.norm(layer_norm(a) - layer_norm(b))
    assert pre == pytest.approx(np.sqrt(2.0))
    assert post != pytest.approx(pre)
    # Inner product with a Euclidean obstruction weight is not LN-invariant.
    w = np.array([1.0, 0.0, 0.0])
    assert float(w @ a) != pytest.approx(float(w @ layer_norm(a)))


def test_attention_weights_are_input_dependent():
    values = np.array([[1.0, 0.0], [0.0, 1.0]])
    mix_eq = attention_mix(values, np.array([0.0, 0.0]))
    mix_hi = attention_mix(values, np.array([5.0, 0.0]))
    assert mix_eq == pytest.approx(np.array([0.5, 0.5]))
    assert mix_hi[0] > 0.95
    rho = 0.5 * np.eye(2)
    # A shared linear rho that matches the equal-weight mix fails on the peaked mix.
    pred_hi = rho @ values[0] + rho @ values[1]
    assert not np.allclose(pred_hi, mix_hi)


def test_residual_block_is_not_a_constant_linear_map():
    def attn(h):
        return np.tanh(h)

    def mlp(h):
        return np.maximum(h, 0.0)

    h = np.array([1.0, -2.0])
    out = residual_block(h, attn, mlp)
    # No matrix A with A h = out for all h, checked on two points.
    h2 = np.array([-1.0, 2.0])
    out2 = residual_block(h2, attn, mlp)
    A, *_ = np.linalg.lstsq(np.stack([h, h2]), np.stack([out, out2]), rcond=None)
    recon = np.stack([h, h2]) @ A
    assert np.linalg.norm(recon - np.stack([out, out2])) > 0.1


def test_vector_residual_keeps_the_sign_a_scalar_discards():
    r = np.array([2.0, -1.0])
    scalar = float(r @ r)
    assert scalar == pytest.approx(5.0)
    assert scalar == pytest.approx(float((-r) @ (-r)))
    w = np.array([1.0, 0.0])
    assert float(w @ r) == pytest.approx(-float(w @ (-r)))


def test_signed_path_effect_is_the_gradient_product():
    w = np.array([0.3, -0.7, 1.1])
    delta = np.array([1.0, 2.0, 0.0])
    dL = float(w @ delta)
    assert dL == pytest.approx(0.3 + -1.4)


def test_probe_preserving_direction_can_move_the_logit():
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    delta = np.array([0.0, 0.0, 2.0])
    h = np.array([0.4, -0.1, 0.2])
    w = np.array([0.0, 0.0, 1.0])
    assert np.allclose(P @ (h + delta), P @ h)
    assert float(w @ (h + delta)) - float(w @ h) == pytest.approx(2.0)


def test_randomized_linear_interaction_is_zero(seed=SEED):
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(50):
        Wu = rng.normal(size=3)
        Ww = rng.normal(size=3)
        v = rng.normal(size=3)

        def F(u_s, w_s):
            return float(v @ (u_s * Wu + w_s * Ww))

        I = F(1.0, 1.0) - F(1.0, 0.0) - F(0.0, 1.0) + F(0.0, 0.0)
        if abs(I) > 1e-10:
            failures += 1
    assert failures == 0


def test_randomized_blind_kernel_search():
    rng = np.random.default_rng(SEED + 1)
    signed = []
    for _ in range(30):
        P = rng.normal(size=(3, 8))
        w = rng.normal(size=8)
        _, _, vh = np.linalg.svd(P, full_matrices=True)
        delta = vh[-1]
        dL = float(w @ delta)
        signed.append(np.sign(dL) if abs(dL) > 1e-12 else 0.0)
    assert set(signed) >= {-1.0, 1.0}


def test_description_is_not_mediation():
    h = np.array([1.0, 2.0])
    Q = float(h @ h)
    L = float(h[0])
    h2 = h + np.array([1.0, 0.0])
    assert Q != pytest.approx(float(h2 @ h2))
    assert L != pytest.approx(float(h2[0]))
    # Both moved because h moved. Q is not on the path to L: L ignores h[1],
    # and changing only h[1] moves Q and not L.
    h3 = h + np.array([0.0, 5.0])
    assert float(h3 @ h3) != pytest.approx(Q)
    assert float(h3[0]) == pytest.approx(L)
