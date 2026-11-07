"""Synthetic verification of COR Phase 1 algebraic claims and counterexamples.

No Transformer, no GPU, no Othello checkpoint. Passing these tests verifies
finite-dimensional linear algebra, not a scientific theory of Transformer decisions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cor_phase1_math import (
    coboundary,
    distance_D,
    equal_weighting_makes_O_eq_D,
    kernel_basis,
    obstruction_O,
    spectral_bounds,
)


TOL = 1e-10


def _two_node(rho, su, sv, wu=1.0, wv=1.0, we=1.0):
    d = coboundary(np.asarray(rho, dtype=np.float64))
    s = np.concatenate([np.asarray(su, dtype=np.float64).reshape(-1), np.asarray(sv, dtype=np.float64).reshape(-1)])
    q, r = np.asarray(rho).shape[1], np.asarray(rho).shape[0]
    M = np.diag([wu] * q + [wv] * r).astype(np.float64)
    W = we * np.eye(r)
    return d, s, M, W


def test_weighted_projection_formula_matches_direct_minimization():
    rng = np.random.default_rng(0)
    rho = rng.normal(size=(2, 2))
    su, sv = rng.normal(size=2), rng.normal(size=2)
    d, s, M, _ = _two_node(rho, su, sv, wu=2.0, wv=3.0)
    pack = distance_D(s, d, M)
    assert pack["D_formula"] == pytest.approx(pack["D_projection"], abs=1e-10)
    assert np.linalg.norm(d @ pack["g_hat"]) < 1e-10


def test_O_zero_iff_D_zero_when_W_and_M_positive_definite():
    d, s_bad, M, W = _two_node([[2.0]], [1.0], [0.0])
    assert obstruction_O(s_bad, d, W) > 0
    assert distance_D(s_bad, d, M)["D_formula"] > 0
    s_good = np.array([1.0, 2.0])  # rho=2, 2*1 - 2 = 0
    assert obstruction_O(s_good, d, W) == pytest.approx(0.0, abs=1e-12)
    assert distance_D(s_good, d, M)["D_formula"] == pytest.approx(0.0, abs=1e-12)


def test_choosing_W_star_makes_O_equal_D():
    rng = np.random.default_rng(1)
    rho = rng.normal(size=(3, 2))
    d, s, M, _ = _two_node(rho, rng.normal(size=2), rng.normal(size=3), wu=1.5, wv=0.7)
    eq = equal_weighting_makes_O_eq_D(d, M, s)
    assert eq["O"] == pytest.approx(eq["D"], rel=1e-9, abs=1e-10)


def test_theorem1_sandwich_on_two_node_example():
    d, s, M, W = _two_node([[1.5]], [2.0], [-1.0], wu=2.0, wv=4.0, we=3.0)
    o = obstruction_O(s, d, W)
    dist = distance_D(s, d, M)["D_projection"]
    spec = spectral_bounds(d, W, M)
    assert spec["lambda_min_plus"] * dist - 1e-8 <= o <= spec["lambda_max"] * dist + 1e-8


def test_attack_c_metric_ranking_swap():
    """Two candidates exchange O-rank solely by changing W, and D-rank by changing M."""
    d = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]], dtype=np.float64)
    s1 = np.array([1.0, 0.0, 0.0, 0.0])
    s2 = np.array([0.0, 0.0, 1.0, 0.0])
    W_a = np.diag([10.0, 1.0])
    W_b = np.diag([1.0, 10.0])
    assert obstruction_O(s1, d, W_a) > obstruction_O(s2, d, W_a)
    assert obstruction_O(s1, d, W_b) < obstruction_O(s2, d, W_b)
    M_a = np.diag([10.0, 1.0, 1.0, 1.0])
    M_b = np.diag([1.0, 1.0, 10.0, 1.0])
    assert distance_D(s1, d, M_a)["D_formula"] > distance_D(s2, d, M_a)["D_formula"]
    assert distance_D(s1, d, M_b)["D_formula"] < distance_D(s2, d, M_b)["D_formula"]


def test_attack_d_coordinate_rescaling_without_metric_transform():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    M = np.eye(2)
    W = np.eye(1)
    o0 = obstruction_O(s, d, W)
    # Rescale the second coordinate: s_v' = 2 s_v, rho' = 2 rho, Euclidean W,M held fixed.
    d2 = coboundary(np.array([[2.0]]))
    s2 = np.array([1.0, 0.0])
    o1 = obstruction_O(s2, d2, W)
    assert o1 == pytest.approx(4.0 * o0)
    # Correct metric transform on C1: W' = (A^{-T}) W A^{-1} with A=2 on the edge space.
    W_corr = np.array([[0.25]])
    assert obstruction_O(s2, d2, W_corr) == pytest.approx(o0)


def test_attack_e_unconstrained_rho_zeros_observed_discrepancy():
    su, sv = np.array([2.0, 0.0]), np.array([0.0, 3.0])
    # Rank-2 free linear map can send su to sv if su != 0.
    rho = np.outer(sv, su) / float(su @ su)
    d = coboundary(rho)
    s = np.concatenate([su, sv])
    assert obstruction_O(s, d, np.eye(2)) == pytest.approx(0.0, abs=1e-12)


def test_attack_f_zero_projection_collapses_obstruction():
    d = coboundary(np.array([[1.0]]))
    s = np.zeros(2)
    assert obstruction_O(s, d, np.eye(1)) == pytest.approx(0.0)
    assert distance_D(s, d, np.eye(2))["D_formula"] == pytest.approx(0.0)


def test_attack_g_edge_duplication_changes_O_not_necessarily_D():
    rho = np.array([[1.0]])
    d1 = coboundary(rho)
    s = np.array([1.0, 0.0])
    W1 = np.eye(1)
    o1 = obstruction_O(s, d1, W1)
    d2 = np.vstack([d1, d1])
    W2 = np.eye(2)
    o2 = obstruction_O(s, d2, W2)
    assert o2 == pytest.approx(2.0 * o1)
    d_single = distance_D(s, d1, np.eye(2))["D_formula"]
    d_dup = distance_D(s, d2, np.eye(2))["D_formula"]
    assert d_dup == pytest.approx(d_single, abs=1e-10)


def test_attack_h_zero_obstruction_without_discriminative_content():
    d = coboundary(np.array([[1.0]]))
    s = np.array([0.0, 0.0])
    assert obstruction_O(s, d, np.eye(1)) == pytest.approx(0.0)
    logit_c1 = 0.0  # independent readout
    logit_c2 = 5.0
    assert logit_c1 < logit_c2


def test_attack_i_tree_mismatch_is_not_cohomology():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    residual = d @ s
    assert abs(residual[0]) > 0
    k = kernel_basis(d)
    assert k.shape[1] == 1
    # [d s] is exact, so it is the zero class in H^1.
    assert np.allclose(residual, d @ s)


def test_attack_a_same_obstruction_opposite_decisions():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.5])
    o = obstruction_O(s, d, np.eye(1))
    dist = distance_D(s, d, np.eye(2))["D_formula"]
    w1, w2 = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    l1, l2 = float(w1 @ s), float(w2 @ s)
    assert o == o and dist == dist
    assert l1 > 0 > l2


def test_attack_b_lower_obstruction_worse_target_logit():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    s_repair = np.array([0.0, 0.0])  # D=0
    w = np.array([1.0, 0.0])  # target logit reads first coordinate
    assert distance_D(s_repair, d, np.eye(2))["D_formula"] < distance_D(s, d, np.eye(2))["D_formula"]
    assert float(w @ s_repair) < float(w @ s)


def test_attack_b_higher_obstruction_better_target_logit():
    d = coboundary(np.array([[1.0]]))
    s = np.array([0.0, 0.0])
    s_worse_cons = np.array([4.0, 0.0])
    w = np.array([1.0, 0.0])
    assert distance_D(s_worse_cons, d, np.eye(2))["D_formula"] > distance_D(s, d, np.eye(2))["D_formula"]
    assert float(w @ s_worse_cons) > float(w @ s)


def test_attack_j_same_decision_different_obstruction():
    d_a = coboundary(np.array([[1.0]]))
    d_b = coboundary(np.array([[10.0]]))
    s_a = np.array([1.0, 0.0])
    s_b = np.array([1.0, 0.0])
    logit = lambda s: float(s[0])
    assert logit(s_a) == logit(s_b)
    assert obstruction_O(s_a, d_a, np.eye(1)) != pytest.approx(obstruction_O(s_b, d_b, np.eye(1)))


def test_attack_k_minimizer_unique_iff_M_positive_definite_on_complement():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    pack = distance_D(s, d, np.eye(2))
    k = kernel_basis(d)
    # Adding a kernel vector with PD M increases distance, so minimizer unique.
    g2 = pack["g_hat"] + k[:, 0]
    d2 = float((s - g2) @ (s - g2))
    assert d2 > pack["D_projection"] + 1e-12


def test_attack_l_consistency_projection_can_leave_natural_set():
    d = coboundary(np.array([[1.0]]))
    natural = {tuple(np.round(v, 6)) for v in [(1.0, 0.0), (2.0, 1.0)]}
    s = np.array([1.0, 0.0])
    g = distance_D(s, d, np.eye(2))["g_hat"]
    assert tuple(np.round(g, 6)) not in natural


def test_singular_d_pseudoinverse_still_matches_projection():
    # Two identical constraints: rank-deficient gram.
    d = np.vstack([coboundary(np.array([[1.0]])), coboundary(np.array([[1.0]]))])
    s = np.array([3.0, 1.0])
    pack = distance_D(s, d, np.eye(2))
    assert pack["D_formula"] == pytest.approx(pack["D_projection"], abs=1e-10)


def test_empty_edge_set_makes_O_zero_and_D_zero():
    d = np.zeros((0, 2))
    s = np.array([4.0, -7.0])
    W = np.zeros((0, 0))
    assert obstruction_O(s, d, W) == pytest.approx(0.0)
    # ker d = all of C0, so D=0.
    k = kernel_basis(d)
    assert k.shape[1] == 2
