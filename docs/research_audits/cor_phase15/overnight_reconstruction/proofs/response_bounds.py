"""Bounds and exact families for the finite response of one pre-norm block.

Information budgets are arguments of the functions. None of the predictors
below takes the scalar R or the downstream derivative Dm as an input.
Comparisons with R and Dm are checks, and they live in the tests.
"""

from __future__ import annotations

import numpy as np

from proofs.block import (
    block,
    block_jvp,
    centered,
    layer_norm,
    layer_norm_jacobian,
    make_params,
    margin_jvp,
    response,
    skip_prediction,
    token_direction,
)


def _kill_right(W, vector):
    """Return a copy of W whose columns are orthogonal to vector.

    W has shape (d, d_head) or (n, d, d_head). The map x |-> x @ W then kills
    `vector`.
    """
    vector = np.asarray(vector, dtype=float)
    denom = float(vector @ vector)
    if denom == 0.0:
        return np.array(W, dtype=float, copy=True)
    out = np.array(W, dtype=float, copy=True)
    if out.ndim == 2:
        coeff = out.T @ vector
        out = out - np.outer(vector, coeff) / denom
        return out
    for h in range(out.shape[0]):
        out[h] = _kill_right(out[h], vector)
    return out


def _kill_left(A, vector):
    """Return a copy of A (m, d) with A @ vector = 0."""
    vector = np.asarray(vector, dtype=float)
    denom = float(vector @ vector)
    out = np.array(A, dtype=float, copy=True)
    if denom == 0.0:
        return out
    coeff = out @ vector
    out = out - np.outer(coeff, vector) / denom
    return out


def branch_increment(H, params, token, delta):
    """B(H+E) - B(H), where H2 = H + B(H). Does not dot with the readout."""
    H2, _ = block(H, params)
    H2_i, _ = block(
        H + token_direction(H.shape[0], H.shape[1], token, delta),
        params,
    )
    skip = token_direction(H.shape[0], H.shape[1], token, delta)
    return (H2_i - H2) - skip


def scaled_weight_free_witness(alpha, token=1, delta=None):
    """A non-constant edit whose response grows without bound as alpha grows.

    Budget for the closed form: architecture, eps, gains, the edit, w_read,
    and the scalar alpha that scales W_V W_O. The closed form does not read R
    or Dm. The branch matrices are set, not estimated from a response.

    W_Q = W_K = 0 makes every legal score zero, so each causal row is uniform
    and does not depend on X. MLP weights are zero, so the MLP difference is
    zero. The attention difference at the read token is then exactly
    (1/T) DeltaX @ W_Vcat @ W_O.
    """
    params, H = make_params(0)
    d = params["d"]
    T = params["T"]
    if delta is None:
        delta = centered(np.array([0.3, -0.1, 0.4, -0.6]))
    delta = np.asarray(delta, dtype=float)
    h = np.zeros(d)
    # Use a constant base token so DeltaX = LN(delta) - LN(0) = LN(delta) - beta.
    H = np.zeros((T, d))
    params["gamma1"] = np.ones(d)
    params["beta1"] = np.zeros(d)
    params["gamma2"] = np.ones(d)
    params["beta2"] = np.zeros(d)
    params["W_Q"] = np.zeros_like(params["W_Q"])
    params["W_K"] = np.zeros_like(params["W_K"])
    params["b_O"] = np.zeros(d)
    params["W1"] = np.zeros_like(params["W1"])
    params["b1"] = np.zeros_like(params["b1"])
    params["W2"] = np.zeros_like(params["W2"])
    params["b2"] = np.zeros(d)
    params["w_read"] = np.array([0.2, -0.1, 0.0, 0.3])
    x0 = layer_norm(h, params["gamma1"], params["beta1"], params["eps"])
    x1 = layer_norm(h + delta, params["gamma1"], params["beta1"], params["eps"])
    delta_x = x1 - x0
    # W_Vcat = I, W_O = alpha * outer(delta_x, w) / ||delta_x||^2
    # so delta_x @ W_Vcat @ W_O = alpha * w.
    w = params["w_read"]
    scale = float(delta_x @ delta_x)
    W_O = alpha * np.outer(delta_x, w) / scale
    params["W_O"] = W_O
    W_V = np.zeros_like(params["W_V"])
    # Two heads, d_head=2, concatenate to I_4.
    W_V[0] = np.eye(d)[:, :2]
    W_V[1] = np.eye(d)[:, 2:]
    params["W_V"] = W_V
    closed_form = alpha * float(w @ w) / T
    if token == params["read_token"]:
        closed_form = closed_form + float(w @ delta)
    return params, H, delta, delta_x, closed_form


def finite_kernel_family(token=2):
    """Non-constant edit whose finite branch increment is exactly zero.

    The branch weights are built so that the finite LayerNorm increments lie
    in the kernels of W_Q, W_K, W_V and of W1. The construction uses H, delta,
    and those matrices. It does not use R or Dm.

    The edit is orthogonal to w_read, so the skip prediction is 0 even though
    the edited token is the readout. The constant-shift identity does not
    apply: delta is not constant.
    """
    params, H = make_params(1)
    d = params["d"]
    params["w_read"] = np.array([1.0, 0.0, 0.0, 0.0])
    delta = np.array([0.0, 0.45, -0.25, 0.15])
    assert abs(float(params["w_read"] @ delta)) < 1e-15
    assert not np.allclose(delta, delta.mean())
    h = H[token]
    delta_x = layer_norm(h + delta, params["gamma1"], params["beta1"], params["eps"])
    delta_x = delta_x - layer_norm(h, params["gamma1"], params["beta1"], params["eps"])
    assert np.linalg.norm(delta_x) > 1e-6
    params["W_Q"] = _kill_right(params["W_Q"], delta_x)
    params["W_K"] = _kill_right(params["W_K"], delta_x)
    params["W_V"] = _kill_right(params["W_V"], delta_x)
    # Attention is now invariant, so H1 moves only by the skip.
    _H2, cache = block(H, params)
    h1 = cache["H1"][token]
    delta_u = layer_norm(h1 + delta, params["gamma2"], params["beta2"], params["eps"])
    delta_u = delta_u - layer_norm(h1, params["gamma2"], params["beta2"], params["eps"])
    assert np.linalg.norm(delta_u) > 1e-6
    params["W1"] = _kill_left(params["W1"], delta_u)
    prediction = skip_prediction(params, token, delta)
    return params, H, delta, prediction, delta_x, delta_u


def jacobian_kernel_not_finite(token=2, step=0.8):
    """A direction killed by the branch Jacobian, not by the finite branch.

    k = J_LN(h) u lies in every Q,K,V kernel, and J_LN2(H1) u lies in ker W1,
    so the directional derivative of the branch is zero. The finite LayerNorm
    increment at step*u does not lie in that kernel.
    """
    params, H = make_params(2)
    d = params["d"]
    rng = np.random.default_rng(2)
    u = centered(rng.normal(size=d))
    u = u / np.linalg.norm(u)
    h = H[token]
    k = layer_norm_jacobian(h, params["gamma1"], params["eps"]) @ u
    assert np.linalg.norm(k) > 1e-6
    params["W_Q"] = _kill_right(params["W_Q"], k)
    params["W_K"] = _kill_right(params["W_K"], k)
    params["W_V"] = _kill_right(params["W_V"], k)
    _H2, cache = block(H, params)
    n = layer_norm_jacobian(cache["H1"][token], params["gamma2"], params["eps"]) @ u
    if np.linalg.norm(n) > 1e-10:
        params["W1"] = _kill_left(params["W1"], n)
    dH = token_direction(H.shape[0], d, token, u)
    linear = margin_jvp(H, dH, params)
    skip_linear = skip_prediction(params, token, u)
    delta = step * u
    h = H[token]
    delta_x = layer_norm(h + delta, params["gamma1"], params["beta1"], params["eps"])
    delta_x = delta_x - layer_norm(h, params["gamma1"], params["beta1"], params["eps"])
    # Distance from the finite increment to span{k}. If this is positive, the
    # finite increment is not the infinitesimal direction.
    coeff = float(delta_x @ k) / float(k @ k)
    outside = delta_x - coeff * k
    return {
        "params": params,
        "H": H,
        "u": u,
        "k": k,
        "linear": linear,
        "skip_linear": skip_linear,
        "step": step,
        "delta": delta,
        "outside_norm": float(np.linalg.norm(outside)),
    }


def centered_branch_jacobian(H, params, token):
    """Derivative of the branch increment at the read token, on centered edits.

    Columns are images of an orthonormal basis of 1^perp. Rank d-1 means the
    derivative is injective on that subspace. The matrix is a derivative of the
    block, used only to test the local-uniqueness hypothesis, not as a predictor.
    """
    d = H.shape[1]
    C = np.eye(d) - np.ones((d, d)) / d
    # Orthonormal basis of the centered subspace from the nonzero eigenvectors of C.
    _vals, vecs = np.linalg.eigh(C)
    basis = vecs[:, _vals > 0.5]
    columns = []
    for i in range(basis.shape[1]):
        dH = token_direction(H.shape[0], d, token, basis[:, i])
        d_branch = block_jvp(H, dH, params) - dH
        columns.append(d_branch[params["read_token"]])
    return np.column_stack(columns), basis


def spectral_norm(matrix):
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def lipschitz_constant(params):
    """A proved Euclidean Lipschitz constant of m on all residual states.

    The derivation is in workstream 1 and proofs/response_bounds.md is not a
    separate file: the factor names below match the writeup in
    workstream_1/finite_response.md.

    Budget: weights, gains, biases, eps, architecture. Not H, not R, not Dm.
    """
    eps = params["eps"]
    T = params["T"]
    d = params["d"]
    dk = params["W_Q"].shape[-1]
    n_heads = params["W_Q"].shape[0]

    def token_radius(gamma, beta):
        return float(np.max(np.abs(gamma)) * np.sqrt(d) + np.linalg.norm(beta))

    R1 = token_radius(params["gamma1"], params["beta1"])
    L1 = float(np.max(np.abs(params["gamma1"])) / np.sqrt(eps))
    L2 = float(np.max(np.abs(params["gamma2"])) / np.sqrt(eps))
    head_constants = []
    for h in range(n_heads):
        s_q = spectral_norm(params["W_Q"][h])
        s_k = spectral_norm(params["W_K"][h])
        s_v = spectral_norm(params["W_V"][h])
        # ||Δhead|| <= L_h ||ΔX||_F. See the three-term expansion of QK^T.
        l_h = (
            2.0 * s_q * s_k * s_v * T * R1**2 / np.sqrt(dk)
            + np.sqrt(T) * s_v
        )
        head_constants.append(l_h)
    L_A = spectral_norm(params["W_O"]) * float(np.sqrt(np.sum(np.square(head_constants))))
    L_mlp = spectral_norm(params["W1"]) * spectral_norm(params["W2"])
    L_state = (1.0 + L_mlp * L2) * (1.0 + L_A * L1)
    return float(np.linalg.norm(params["w_read"]) * L_state)


def token_gradient_jvp(H, params, token):
    d = H.shape[1]
    g = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        g[i] = margin_jvp(H, token_direction(H.shape[0], d, token, e), params)
    return g


def token_gradient_central(H, params, token, step=1e-6):
    """Independent of margin_jvp. Uses two finite responses per coordinate."""
    d = H.shape[1]
    g = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        g[i] = (
            response(H, params, token, step * e)
            - response(H, params, token, -step * e)
        ) / (2.0 * step)
    return g


def smallest_centered_singular_value(H, params, token):
    matrix, _basis = centered_branch_jacobian(H, params, token)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular[-1]), matrix
