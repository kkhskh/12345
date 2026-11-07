"""Synthetic checks for the Phase 1.5 causal-bridge audit.

These are finite-dimensional counterexamples and identities. They do not run a
Transformer, fit an obstruction score to model outputs, or touch confirmation
data. A passing test records a calculation. It is not evidence that a causal
bridge exists.
"""

from __future__ import annotations

import numpy as np
import pytest


TOL = 1e-10


def coboundary(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    r, q = rho.shape
    d = np.zeros((r, q + r))
    d[:, :q] = rho
    d[:, q:] = -np.eye(r)
    return d


def two_edge_coboundary() -> np.ndarray:
    """Two disconnected identity edges on (u1, v1, u2, v2)."""
    d = np.zeros((2, 4))
    d[0, 0] = 1.0
    d[0, 1] = -1.0
    d[1, 2] = 1.0
    d[1, 3] = -1.0
    return d


def distance_D(s: np.ndarray, d: np.ndarray, M: np.ndarray | None = None) -> float:
    """M-distance to ker d. Rank-deficient rows are allowed; a tiny operator is not."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    d = np.asarray(d, dtype=np.float64)
    if M is None:
        M = np.eye(s.size)
    M = np.asarray(M, dtype=np.float64)
    evals = np.linalg.eigvalsh(0.5 * (M + M.T))
    assert float(evals.min()) > 1e-8, "M is not positive definite"
    singular = np.linalg.svd(d, compute_uv=False)
    scale = float(singular.max()) if singular.size else 0.0
    assert scale > 1e-8, f"ill-conditioned coboundary scale={scale}; not a stable counterexample"
    _, sv, vh = np.linalg.svd(d, full_matrices=True)
    rank = int(np.sum(sv > scale * 1e-8)) if sv.size else 0
    basis = vh[rank:].T
    if basis.size == 0:
        projected = float(s @ M @ s)
    else:
        coeff = np.linalg.solve(basis.T @ M @ basis, basis.T @ M @ s)
        diff = s - basis @ coeff
        projected = float(diff @ M @ diff)
    gram = d @ np.linalg.solve(M, d.T)
    residual = d @ s
    formula = float(residual @ np.linalg.pinv(gram, rcond=1e-12) @ residual)
    assert formula == pytest.approx(projected, abs=1e-8)
    return projected


def obstruction_O(s, d, W=None) -> float:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    residual = d @ s
    if W is None:
        return float(residual @ residual)
    return float(residual @ W @ residual)


def test_pseudoinverse_formula_matches_projection():
    rho = np.array([[1.0, 0.0], [0.5, 2.0]])
    d = coboundary(rho)
    s = np.array([0.3, -1.0, 2.0, 0.5])
    M = np.diag([1.0, 2.0, 3.0, 4.0])
    assert distance_D(s, d, M) > 0


def test_ce01_equal_obstruction_opposite_decisions():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.5])
    assert obstruction_O(s, d) == pytest.approx(0.25)
    assert distance_D(s, d) == pytest.approx(0.125)
    assert float(np.array([1.0, 0.0]) @ s) == pytest.approx(1.0)
    assert float(np.array([-1.0, 0.0]) @ s) == pytest.approx(-1.0)


def test_ce02_reduced_obstruction_decreases_logit():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    s_repaired = np.array([0.0, 0.0])
    assert distance_D(s_repaired, d) < distance_D(s, d)
    assert s_repaired[0] < s[0]


def test_ce03_increased_obstruction_increases_logit():
    d = coboundary(np.array([[1.0]]))
    s = np.array([0.0, 0.0])
    s_worse = np.array([4.0, 0.0])
    assert distance_D(s_worse, d) > distance_D(s, d)
    assert s_worse[0] > s[0]


def test_ce04_perfect_consistency_incorrect_decision():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 1.0])
    assert distance_D(s, d) == pytest.approx(0.0)
    logit = float(np.array([-1.0, 0.0]) @ s)
    assert logit < 0


def test_ce05_exact_commutation_is_zero_after_recomputation():
    def f(h):
        return 2.0 * h

    rho = 2.0
    for h in (-1.5, 0.0, 0.75):
        assert rho * h - f(h) == pytest.approx(0.0)
        for delta in (-0.4, 1.25):
            h_do = h + delta
            assert rho * h_do - f(h_do) == pytest.approx(0.0)
    stitched = rho * 1.0 - 0.0
    assert stitched == pytest.approx(2.0)


def test_ce06_nonzero_residual_is_lost_coordinate():
    # f is the identity on R^2. The parent stalk keeps only the first coordinate.
    parents = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    children = parents.copy()
    parent_stalk = parents[:, :1]
    # Least-squares rho in R^{2x1} cannot reproduce the second coordinate.
    rho, *_ = np.linalg.lstsq(parent_stalk, children, rcond=None)
    fitted = parent_stalk @ rho
    assert np.linalg.norm(fitted - children) > 0.5
    assert np.max(np.abs(fitted[:, 1])) == pytest.approx(0.0)


def test_ce07_metric_change_reverses_ranking():
    d = two_edge_coboundary()
    s1 = np.array([1.0, 0.0, 0.0, 0.0])
    s2 = np.array([0.0, 0.0, 1.0, 0.0])
    heavy_first = np.diag([10.0, 10.0, 1.0, 1.0])
    heavy_second = np.diag([1.0, 1.0, 10.0, 10.0])
    assert distance_D(s1, d, heavy_first) > distance_D(s2, d, heavy_first)
    assert distance_D(s1, d, heavy_second) < distance_D(s2, d, heavy_second)
    w_a = np.diag([10.0, 1.0])
    w_b = np.diag([1.0, 10.0])
    assert obstruction_O(s1, d, w_a) > obstruction_O(s2, d, w_a)
    assert obstruction_O(s1, d, w_b) < obstruction_O(s2, d, w_b)


def test_ce08_zero_projection_kills_discrepancy():
    d = coboundary(np.array([[1.0]]))
    s = np.zeros(2)
    assert obstruction_O(s, d) == pytest.approx(0.0)
    assert distance_D(s, d) == pytest.approx(0.0)


def test_ce09_input_specific_map_manufactures_agreement():
    examples = [(np.array([1.0, 0.0]), np.array([0.0, 4.0])), (np.array([0.0, 1.0]), np.array([3.0, -2.0]))]
    for su, sv in examples:
        rho = np.outer(sv, su)  # su is a basis vector, so rho su = sv
        assert np.linalg.norm(rho @ su - sv) == pytest.approx(0.0)
    rho_shared = np.zeros((2, 2))
    residuals = [np.linalg.norm(rho_shared @ su - sv) for su, sv in examples]
    assert min(residuals) > 1.0


def test_ce10_reparameterization_changes_euclidean_obstruction():
    s = np.array([1.0, 0.0])
    d = coboundary(np.array([[1.0]]))
    euclidean = obstruction_O(s, d)
    # s_v' = 2 s_v and rho' = 2, with the Euclidean metric left unchanged.
    s_scaled = np.array([1.0, 0.0])
    d_scaled = coboundary(np.array([[2.0]]))
    assert obstruction_O(s_scaled, d_scaled) == pytest.approx(4.0)
    assert euclidean == pytest.approx(1.0)
    w_corrected = np.array([[0.25]])
    assert obstruction_O(s_scaled, d_scaled, w_corrected) == pytest.approx(euclidean)


def test_ce11_zero_obstruction_state_need_not_be_attainable():
    natural = np.array([1.0, 1.0])
    off = np.array([2.0, 2.0])
    d = coboundary(np.array([[1.0]]))
    assert distance_D(off, d) == pytest.approx(0.0)
    assert not np.allclose(off, natural)
    inconsistent = np.array([1.0, 0.0])
    repair = np.array([0.5, 0.5])
    assert distance_D(repair, d) == pytest.approx(0.0)
    assert not np.allclose(repair, natural)


def test_ce12_omitted_parent_creates_discrepancy():
    su, sw, sv = 1.0, 1.0, 2.0
    joint = su + sw - sv
    pairwise = su - sv
    assert joint == pytest.approx(0.0)
    assert pairwise == pytest.approx(-1.0)
    # Duplicating the mismatched pairwise edge doubles Euclidean O.
    once = pairwise**2
    twice = 2.0 * once
    assert twice == pytest.approx(2.0)


def test_residual_skip_omission_is_not_a_conflict():
    h = np.array([0.5, -0.25])
    branch = h  # residual update equal to the skip input
    h_v = h + branch
    rho_skip = np.eye(2)
    rho_full = 2.0 * np.eye(2)
    assert np.linalg.norm(rho_skip @ h - h_v) == pytest.approx(np.linalg.norm(h))
    assert np.linalg.norm(rho_full @ h - h_v) == pytest.approx(0.0)


def test_relu_global_linear_map_residual_is_the_kink():
    h = np.array([-2.0, 2.0])
    f = np.array([0.0, 2.0])
    rho = float(np.linalg.lstsq(h[:, None], f, rcond=None)[0][0])
    assert rho == pytest.approx(0.5)
    residual = rho * h - f
    assert residual == pytest.approx(np.array([-1.0, -1.0]))
    positive = h[h > 0]
    rho_piece = float(np.linalg.lstsq(positive[:, None], f[h > 0], rcond=None)[0][0])
    assert rho_piece == pytest.approx(1.0)


def test_ce13_exact_abstraction_can_ignore_the_decision():
    def f(h):
        return h.copy()

    def stalk(h):
        return h[0]

    def logit(h):
        return h[1]

    rho = 1.0
    for h in (np.array([1.0, 3.0]), np.array([1.0, -4.0])):
        assert rho * stalk(h) - stalk(f(h)) == pytest.approx(0.0)
    h_a = np.array([1.0, 3.0])
    h_b = np.array([1.0, -4.0])
    assert stalk(h_a) == pytest.approx(stalk(h_b))
    assert logit(h_a) != pytest.approx(logit(h_b))


def test_ce14_useful_feature_with_a_wrong_map():
    h = np.array([1.5])
    logit = float(h[0])
    rho_wrong = 0.0
    transported = 1.0 * h  # the feature is actually copied
    assert logit == pytest.approx(1.5)
    assert np.linalg.norm(rho_wrong * h - transported) > 0.5


def test_ce15_bound_does_not_determine_sign():
    s = np.array([1.0, 0.0])
    projection = np.array([0.5, 0.5])
    d = coboundary(np.array([[1.0]]))
    gap = abs(s[0] - projection[0])
    assert gap <= np.sqrt(distance_D(s, d)) + TOL
    assert s[0] - projection[0] > 0
    s_neg = np.array([-1.0, 0.0])
    assert s_neg[0] - (-0.5) < 0
    assert distance_D(s, d) == pytest.approx(distance_D(s_neg, d))


def test_squared_distance_discards_sign():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    assert distance_D(s, d) == pytest.approx(distance_D(-s, d))
    assert s[0] == pytest.approx(-(-s[0]))


def test_ce16_monotonic_assumption_is_the_whole_implication():
    values = np.array([0.0, 0.25, 1.0])
    assumed = -values
    assert np.all(np.diff(assumed) < 0) or np.all(np.diff(values) > 0)
    flipped = values
    # The opposite monotone function reverses the response that the assumption created.
    assert flipped[2] > flipped[0]
    assert assumed[2] < assumed[0]


def test_ce17_obstruction_descent_can_oppose_the_logit_gradient():
    # O = (s_u - s_v)^2 at (1, 0). L = s_u.
    grad_o = np.array([2.0, -2.0])
    grad_l = np.array([1.0, 0.0])
    assert float(grad_l @ grad_o) > 0
    descent = -grad_o
    assert float(grad_l @ descent) < 0
    attribution = grad_l
    assert float(grad_l @ attribution) > 0
    assert not np.allclose(descent / np.linalg.norm(descent), attribution)


def test_factorization_through_kernel_makes_discrepancy_idle():
    d = coboundary(np.array([[1.0]]))
    s = np.array([1.0, 0.0])
    consistent = np.array([0.5, 0.5])
    # L depends only on the consistent component s_u + s_v.
    def logit(z):
        return float(z[0] + z[1])

    assert logit(s) == pytest.approx(logit(consistent))
    assert distance_D(consistent, d) < distance_D(s, d)
    moved = consistent + np.array([0.7, -0.7])
    assert logit(moved) == pytest.approx(logit(consistent))
    assert distance_D(moved, d) > distance_D(consistent, d)


def test_ce18_lookup_alignment_fits_an_unrelated_algorithm():
    activations = [np.array([0.2]), np.array([0.9])]
    algorithm = 1.0  # constant, independent of the activation
    network_outputs = [0.0, 1.0]
    aligned = [algorithm for _ in activations]
    assert aligned == [1.0, 1.0]
    assert network_outputs != aligned


def test_ce19_recomputation_preserves_consistency_and_changes_the_logit():
    h = 0.25
    rho = 2.0

    def step(value):
        child = 2.0 * value
        return child, rho * value - child, value

    _, residual_before, logit_before = step(h)
    h_do = h + 1.0
    _, residual_after, logit_after = step(h_do)
    assert residual_before == pytest.approx(0.0)
    assert residual_after == pytest.approx(0.0)
    assert logit_after - logit_before == pytest.approx(1.0)


def test_ce20_stitch_can_have_large_discrepancy_and_no_output_effect():
    # The logit is a constant function of a site the stitch does not feed.
    def logit(_stitch):
        return 0.1

    before = (0.0, 0.0)
    stitched = (0.0, 100.0)
    discrepancy = stitched[0] - stitched[1]
    assert abs(discrepancy) == pytest.approx(100.0)
    assert logit(before) == pytest.approx(logit(stitched))


def test_attention_mix_pairwise_residual_cancels_in_the_aggregate():
    u1, u2, v = 1.0, -1.0, 0.0
    aggregate = 0.5 * u1 + 0.5 * u2 - v
    pairwise_1 = u1 - v
    pairwise_2 = u2 - v
    assert aggregate == pytest.approx(0.0)
    assert pairwise_1 == pytest.approx(1.0)
    assert pairwise_2 == pytest.approx(-1.0)


def test_smooth_square_map_is_not_a_linear_transport():
    h = np.array([-1.0, 0.0, 1.0])
    f = h**2
    rho = float(np.linalg.lstsq(h[:, None], f, rcond=None)[0][0])
    assert rho == pytest.approx(0.0)
    assert np.linalg.norm(rho * h - f) == pytest.approx(np.sqrt(2.0))


def test_duplicate_edge_keeps_distance_and_doubles_energy():
    d = np.array([[1.0, -1.0], [1.0, -1.0]])
    s = np.array([1.0, 0.0])
    assert distance_D(s, d) == pytest.approx(0.5)
    assert obstruction_O(s, d) == pytest.approx(2.0)


def test_nonlinear_head_has_interaction_while_recomputed_residual_is_zero():
    # v = u + w, y = v^2. Exact aggregation, nonlinear readout.
    def world(u, w):
        v = u + w
        return v * v, u + w - v

    clean_y, clean_r = world(0.0, 0.0)
    y_u, r_u = world(1.0, 0.0)
    y_w, r_w = world(0.0, 1.0)
    y_both, r_both = world(1.0, 1.0)
    interaction = y_both - y_u - y_w + clean_y
    assert interaction == pytest.approx(2.0)
    assert clean_r == r_u == r_w == r_both == pytest.approx(0.0)


def test_pointwise_jacobian_does_not_bound_a_finite_step():
    # L(t) = 1 - 2 t^2. Jacobian at 0 is 0. The value at t=1 has the opposite sign.
    def logit(t):
        return 1.0 - 2.0 * t * t

    assert logit(0.0) == pytest.approx(1.0)
    assert -4.0 * 0.0 == pytest.approx(0.0)
    assert logit(1.0) == pytest.approx(-1.0)


def test_ill_conditioned_gram_is_rejected():
    d = np.array([[1e-14, -1e-14]])
    s = np.array([1.0, 0.0])
    with pytest.raises(AssertionError, match="ill-conditioned"):
        distance_D(s, d)
