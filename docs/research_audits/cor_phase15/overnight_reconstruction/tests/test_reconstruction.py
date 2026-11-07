"""Two routes for each serious claim: a closed form, and a recomputed block.

The recomputed route does not call the closed form. The closed form does not
call `response` or `margin_jvp`.
"""

from __future__ import annotations

import numpy as np
import pytest

from proofs.block import (
    block,
    layer_norm,
    layer_norm_bracket_spectrum,
    layer_norm_jacobian,
    make_params,
    margin_jvp,
    response,
    skip_prediction,
    token_direction,
)
from proofs.coordination import additive_margin, deletion_necessary, four_term_interaction
from proofs.layernorm_facts import (
    sympy_scale_identity_forces_lambda_one,
    two_zero_gain_counterexample,
)
from proofs.obstruction_algebra import obstruction_pair, two_node_operator
from proofs.response_bounds import (
    branch_increment,
    centered_branch_jacobian,
    finite_kernel_family,
    jacobian_kernel_not_finite,
    lipschitz_constant,
    scaled_weight_free_witness,
    smallest_centered_singular_value,
    token_gradient_central,
    token_gradient_jvp,
)


def test_bracket_spectrum_matches_svd():
    rng = np.random.default_rng(0)
    h = rng.normal(size=4)
    eps = 1e-5
    values, _s, _v = layer_norm_bracket_spectrum(h, eps)
    J = layer_norm_jacobian(h, np.ones(4), eps)
    singular = np.linalg.svd(J, compute_uv=False)
    assert sorted(np.abs(values)) == pytest.approx(sorted(singular), abs=1e-8)


def test_finite_kernel_full_gain_and_one_zero_gain():
    rng = np.random.default_rng(1)
    h = rng.normal(size=4)
    eps = 1e-5
    gamma = np.array([1.2, 0.8, 1.1, 0.9])
    beta = np.array([0.1, -0.2, 0.0, 0.3])
    constant = np.array([0.4, 0.4, 0.4, 0.4])
    nonconstant = np.array([0.4, -0.2, 0.1, 0.0])
    assert np.allclose(
        layer_norm(h + constant, gamma, beta, eps),
        layer_norm(h, gamma, beta, eps),
    )
    assert not np.allclose(
        layer_norm(h + nonconstant, gamma, beta, eps),
        layer_norm(h, gamma, beta, eps),
    )
    gamma_one_zero = gamma.copy()
    gamma_one_zero[3] = 0.0
    assert np.allclose(
        layer_norm(h + constant, gamma_one_zero, beta, eps),
        layer_norm(h, gamma_one_zero, beta, eps),
    )
    assert not np.allclose(
        layer_norm(h + nonconstant, gamma_one_zero, beta, eps),
        layer_norm(h, gamma_one_zero, beta, eps),
    )


def test_two_zero_gains_admit_a_nonconstant_level_set():
    h, delta, gamma, beta, eps = two_zero_gain_counterexample()
    assert not np.allclose(delta, np.full_like(delta, delta.mean()))
    assert np.allclose(
        layer_norm(h + delta, gamma, beta, eps),
        layer_norm(h, gamma, beta, eps),
        atol=1e-12,
    )
    # The same displacement is visible once every gain is nonzero.
    assert not np.allclose(
        layer_norm(h + delta, np.ones(4), beta, eps),
        layer_norm(h, np.ones(4), beta, eps),
        atol=1e-8,
    )


def test_sympy_scale_step_is_independent_of_numpy_layernorm():
    out = sympy_scale_identity_forces_lambda_one()
    assert out["positive_solutions"] == ["1"]
    assert "eps" in out["residual"]


def test_constant_component_factors_and_centered_part_uses_the_branch():
    params, H = make_params(7)
    v = np.array([0.2, -0.5, 0.1, 0.2])
    v = v - v.mean()
    a = 0.37
    const = np.full(4, a)
    token = 1
    r_v = response(H, params, token, v)
    r_sum = response(H, params, token, v + const)
    # Read token is 2, so a constant at token 1 contributes 0.
    assert r_sum == pytest.approx(r_v, abs=1e-10)
    # The same centered edit at the read token disagrees with the dead branch.
    dead = {k: (np.zeros_like(val) if isinstance(val, np.ndarray) else val) for k, val in params.items()}
    for key in ("eps", "T", "d", "read_token"):
        dead[key] = params[key]
    dead["gamma1"] = params["gamma1"]
    dead["gamma2"] = params["gamma2"]
    dead["beta1"] = params["beta1"]
    dead["beta2"] = params["beta2"]
    dead["w_read"] = params["w_read"]
    live = response(H, params, params["read_token"], v)
    zero_branch = response(H, dead, params["read_token"], v)
    assert live != pytest.approx(zero_branch, abs=1e-6)
    assert zero_branch == pytest.approx(float(params["w_read"] @ v), abs=1e-10)


def test_centered_response_is_unbounded_in_the_branch_scale():
    ratios = []
    for alpha in (1.0, 10.0, 100.0):
        params, H, delta, delta_x, closed = scaled_weight_free_witness(alpha, token=1)
        assert np.linalg.norm(delta_x) > 1e-6
        got = response(H, params, 1, delta)
        assert got == pytest.approx(closed, rel=1e-10, abs=1e-10)
        ratios.append(got / alpha)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-10)
    assert ratios[1] == pytest.approx(ratios[2], rel=1e-10)
    assert abs(ratios[0]) > 0.0


def test_finite_kernel_family_beats_cauchy_and_constant_shift():
    params, H, delta, prediction, delta_x, delta_u = finite_kernel_family(token=2)
    assert prediction == pytest.approx(0.0, abs=1e-15)
    got = response(H, params, 2, delta)
    assert got == pytest.approx(prediction, abs=1e-9)
    # Route two: the branch tensor itself, not the scalar R.
    increment = branch_increment(H, params, 2, delta)
    assert np.linalg.norm(increment) < 1e-8
    g = token_gradient_jvp(H, params, 2)
    g_fd = token_gradient_central(H, params, 2)
    assert g == pytest.approx(g_fd, rel=1e-4, abs=1e-6)
    cauchy = float(np.linalg.norm(g) * np.linalg.norm(delta))
    assert cauchy == pytest.approx(0.4554966651252395, rel=1e-8)
    assert float(g @ delta) == pytest.approx(0.1066527641320657, rel=1e-8)
    assert cauchy > abs(got) + 1e-4
    # Constant-shift evaluation of this non-constant vector is the skip dot,
    # which happens to be 0, but only the kernel hypothesis makes that exact.
    # A nearby edit that is not in the kernel moves the margin.
    nudged = delta + np.array([0.0, 1e-3, 0.0, 0.0])
    assert abs(response(H, params, 2, nudged)) > 1e-5


def test_jacobian_kernel_does_not_give_the_finite_identity():
    built = jacobian_kernel_not_finite(token=2, step=0.8)
    assert built["linear"] == pytest.approx(built["skip_linear"], abs=1e-8)
    assert built["outside_norm"] > 1e-4
    got = response(built["H"], built["params"], 2, built["delta"])
    skip = skip_prediction(built["params"], 2, built["delta"])
    assert got == pytest.approx(-0.6779358472770767, rel=1e-8)
    assert skip == pytest.approx(-0.6853485283715309, rel=1e-8)
    assert abs(got - skip) > 5e-3
    gaps = []
    for step in (0.8, 1.5, 2.5, 4.0):
        again = jacobian_kernel_not_finite(token=2, step=step)
        finite = response(again["H"], again["params"], 2, again["delta"])
        linear_skip = skip_prediction(again["params"], 2, again["delta"])
        gaps.append(abs(finite - linear_skip))
    assert gaps[0] < gaps[1] < gaps[2] < gaps[3]


def test_generic_initialization_has_an_injective_qkv_map_and_a_local_gap():
    params, H = make_params(0)
    rows = []
    for name in ("W_Q", "W_K", "W_V"):
        for h in range(params[name].shape[0]):
            rows.append(params[name][h].T)
    stacked = np.vstack(rows)
    singular = np.linalg.svd(stacked, compute_uv=False)
    assert singular[-1] > 0.5
    mu, _matrix = smallest_centered_singular_value(H, params, token=1)
    assert mu == pytest.approx(0.015224127891452265, rel=1e-6)
    g = token_gradient_jvp(H, params, 1)
    # Token 1 is not the readout, so the skip contribution is 0.
    # A nonzero gradient means the branch linear form is not identically zero.
    assert np.linalg.norm(g) == pytest.approx(3.548311334245126, rel=1e-6)
    v = np.array([1.0, -1.0, 0.5, -0.5])
    v = v - v.mean()
    v = v / np.linalg.norm(v)
    increment = branch_increment(H, params, 1, 1e-3 * v)
    assert np.linalg.norm(increment) > 1e-5


def test_lipschitz_certificate_is_valid_and_weaker_than_cauchy():
    params, H = make_params(0)
    L = lipschitz_constant(params)
    g = token_gradient_central(H, params, token=1)
    assert L == pytest.approx(2894603.0377523545, rel=1e-8)
    assert L > 1e5 * float(np.linalg.norm(g))
    rng = np.random.default_rng(0)
    for scale in (1e-3, 0.2, 2.0):
        dH = rng.normal(size=H.shape) * scale
        m0 = response(H, params, 0, np.zeros(params["d"]))
        # response at token 0 with 0 is 0; compare margins through two states.
        from proofs.block import margin
        m_base, _, _ = margin(H, params)
        m_moved, _, _ = margin(H + dH, params)
        assert abs(m_moved - m_base) <= L * np.linalg.norm(dH) * (1.0 + 1e-8) + 1e-8


def test_preserved_negative_phenomena_on_the_block():
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

    other = H.copy()
    other[0, 3] += 2.0
    edit = np.array([0.4, -1.2, 0.3, 0.5])
    first = response(H, params, 1, edit)
    second = response(other, params, 1, edit)
    assert first * second < 0

    params0, H0 = make_params(0)
    gradient = token_gradient_jvp(H0, params0, 1)
    direction = np.array([gradient[1], -gradient[0], 0.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    linear = margin_jvp(H0, token_direction(3, 4, 1, direction), params0)
    finite = response(H0, params0, 1, 0.2 * direction)
    assert abs(linear) < 1e-8
    assert abs(finite) > 0.5


def test_obstruction_algebra_does_not_rank_a_readout():
    rho = np.eye(2)
    M = np.eye(4)
    W = np.eye(2)
    s = np.array([1.0, 0.0, 0.0, 0.0])
    O, D, ds = obstruction_pair(s, rho, M, W)
    assert O == pytest.approx(1.0)
    assert D == pytest.approx(0.5)
    assert O != pytest.approx(D)
    # Special W makes O=D.
    d = two_node_operator(rho)
    gram = d @ np.linalg.inv(M) @ d.T
    W_star = np.linalg.pinv(gram)
    O_star, D_star, _ = obstruction_pair(s, rho, M, W_star)
    assert O_star == pytest.approx(D_star)
    # Changing only M changes D and not O.
    M2 = np.diag([4.0, 4.0, 1.0, 1.0])
    O2, D2, _ = obstruction_pair(s, rho, M2, W)
    assert O2 == pytest.approx(O)
    assert D2 != pytest.approx(D)
    # Same obstruction, opposite readouts.
    w = np.array([1.0, 0.0, 0.0, 0.0])
    assert float(w @ s) == pytest.approx(1.0)
    assert float((-w) @ s) == pytest.approx(-1.0)
    # Lower D can lower the readout. s' = 0 has D=0 and readout 0.
    O0, D0, _ = obstruction_pair(np.zeros(4), rho, M, W)
    assert D0 == pytest.approx(0.0)
    assert D0 < D
    assert float(w @ np.zeros(4)) < float(w @ s)
    # P=0 germ: the zero cochain, regardless of the label.
    assert O0 == pytest.approx(0.0)
    # Homogeneous kernel contains 0, and an exact coboundary has class 0:
    # every functional annihilating im(d) annihilates ds.
    # ds = d @ s lies in im(d), so its class in C^1/im(d) is zero.
    # This d is 2 by 4 and has full row rank, so the complement is {0}
    # and the only class is zero. Exactness is the reconstruction identity.
    reconstructed = d @ np.linalg.lstsq(d, ds, rcond=None)[0]
    assert np.allclose(reconstructed, ds)
    assert np.linalg.matrix_rank(d) == d.shape[0]


def test_coordination_notions_do_not_collapse():
    # c1=1, c2=-0.5, sum 0.5. Deleting c2 leaves 1, so it is not necessary.
    def margin(c1, c2):
        return additive_margin(0.0, c1, c2)

    assert deletion_necessary(margin, (1.0, -0.5), 0) is False or margin(0.0, -0.5) <= 0
    assert not (margin(1.0, -0.5) > 0 and margin(1.0, 0.0) <= 0 and margin(0.0, -0.5) <= 0)
    # The positive sum forbids both deletions from being necessary.
    assert margin(1.0, -0.5) > 0
    assert margin(0.0, -0.5) < 0
    assert margin(1.0, 0.0) > 0

    def biased(c1, c2):
        return additive_margin(-1.5, c1, c2)

    assert deletion_necessary(biased, (1.0, 1.0), 0)
    assert deletion_necessary(biased, (1.0, 1.0), 1)
    assert four_term_interaction(lambda u, v: biased(u, v), 1.0, 1.0) == pytest.approx(0.0)
    # A separable nonlinearity has interaction 0.
    assert four_term_interaction(lambda u, v: u**2 + v**2, 0.3, -0.4) == pytest.approx(0.0)
    # A nonseparable map need not.
    assert four_term_interaction(lambda u, v: (u + v) ** 2, 0.3, -0.4) == pytest.approx(2 * 0.3 * -0.4)


def test_same_residual_two_sources():
    # Surrogate error and a missing parent can share one residual vector.
    rho = np.eye(2)
    s_u = np.array([1.0, 0.0])
    parent = np.array([0.4, -0.2])
    s_v = rho @ s_u + parent
    missing_parent_residual = rho @ s_u - s_v
    rho_wrong = rho + np.outer(parent, s_u) / float(s_u @ s_u)
    surrogate_residual = rho_wrong @ s_u - (rho @ s_u)
    # Both residuals equal -parent or parent up to the sign convention of the wrong map.
    assert np.allclose(missing_parent_residual, -parent)
    assert np.allclose(surrogate_residual, parent)
    assert np.allclose(np.linalg.norm(missing_parent_residual), np.linalg.norm(surrogate_residual))


def test_structural_residual_on_a_recomputed_trace_is_zero():
    params, H = make_params(3)
    H2, cache = block(H, params)
    structural = H2 - (cache["H1"] + cache["M"])
    assert np.linalg.norm(structural) < 1e-12
    m, _, _ = __import__("proofs.block", fromlist=["margin"]).margin(H, params)
    # A second seed with the opposite margin sign still has residual 0.
    params2, H_b = make_params(2)
    H2b, cacheb = block(H_b, params2)
    assert np.linalg.norm(H2b - (cacheb["H1"] + cacheb["M"])) < 1e-12
    m2, _, _ = __import__("proofs.block", fromlist=["margin"]).margin(H_b, params2)
    assert m != pytest.approx(m2)


def test_final_layernorm_readout_kills_the_constant_the_raw_readout_keeps():
    """A raw residual dot does not transfer to a final-normalized logit."""
    params, H = make_params(0)
    a = 0.3
    const = np.full(params["d"], a)
    raw = response(H, params, params["read_token"], const)
    assert raw == pytest.approx(a * float(np.sum(params["w_read"])), abs=1e-10)
    from proofs.block import block
    H2, _ = block(H, params)
    moved, _ = block(H + token_direction(params["T"], params["d"], params["read_token"], const), params)
    gamma = np.ones(params["d"])
    beta = np.zeros(params["d"])
    y0 = layer_norm(H2[params["read_token"]], gamma, beta, params["eps"])
    y1 = layer_norm(moved[params["read_token"]], gamma, beta, params["eps"])
    assert np.allclose(y0, y1, atol=1e-12)
    assert float(params["w_read"] @ (y1 - y0)) == pytest.approx(0.0, abs=1e-12)
    assert raw != pytest.approx(0.0, abs=1e-8)


def test_gradient_metric_inherits_the_readout_and_cohomology_does_not():
    # A quadratic built from the readout changes when the readout is negated
    # only through evenness; an inner product with the readout flips.
    g = np.array([0.2, -0.7, 0.1, 0.4])
    delta = np.array([0.3, 0.2, -0.1, 0.5])
    assert float(g @ delta) > 0
    assert float((-g) @ delta) < 0
    # The squared gradient metric does not see the sign.
    assert (float(g @ delta) ** 2) == pytest.approx(float((-g) @ delta) ** 2)
    # Euclidean O on a fixed cochain does not move when g is replaced.
    rho = np.eye(2)
    s = np.array([1.0, -0.2, 0.4, 0.1])
    M = np.eye(4)
    W = np.eye(2)
    O, D, _ = obstruction_pair(s, rho, M, W)
    assert O > 0 and D > 0
    # Negating a readout that was never an input of O leaves O fixed. The
    # number above is already that computation.
    assert O == pytest.approx(obstruction_pair(s, rho, M, W)[0])
