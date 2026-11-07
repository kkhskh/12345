"""Regime-1 finite-response certificates.

These functions take weights, gains, biases, epsilon, and the intervention.
They do not evaluate m, R, or a derivative of m. Spectral norms are the
permitted primitive. The derivations are in 06_direction_specific_finite_bounds.md.
"""

from __future__ import annotations

import numpy as np


def spectral_norm(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0.0
    return float(singular[0])


def layer_norm_radius(gamma, beta, width):
    """Euclidean radius of one LayerNorm image. Proved in the certificate note."""
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    return float(np.max(np.abs(gamma)) * np.sqrt(width) + np.linalg.norm(beta))


def layer_norm_lipschitz(gamma, eps):
    """Global Euclidean Lipschitz constant of LN for eps > 0."""
    if eps <= 0.0:
        raise ValueError("the global LayerNorm Lipschitz bound requires eps > 0")
    return float(np.max(np.abs(np.asarray(gamma, dtype=float))) / np.sqrt(eps))


def head_lipschitz(W_Q, W_K, W_V, radius, tokens):
    """Frobenius Lipschitz constant of one head on the row ball of radius `radius`.

    l = σ(Q)σ(K)σ(V) R^2 T / sqrt(d_k) + σ(V) sqrt(T).
    """
    head_width = W_Q.shape[-1]
    sigma_q = spectral_norm(W_Q)
    sigma_k = spectral_norm(W_K)
    sigma_v = spectral_norm(W_V)
    bilinear = sigma_q * sigma_k * sigma_v * (radius**2) * tokens / np.sqrt(head_width)
    value = sigma_v * np.sqrt(tokens)
    return float(bilinear + value)


def attention_factors(params):
    """Return (L_A, L_concat, R1). L_A = σ(W_O) L_concat."""
    tokens = int(params["T"])
    width = int(params["d"])
    radius = layer_norm_radius(params["gamma1"], params["beta1"], width)
    head_constants = []
    for head in range(params["W_Q"].shape[0]):
        head_constants.append(
            head_lipschitz(
                params["W_Q"][head],
                params["W_K"][head],
                params["W_V"][head],
                radius,
                tokens,
            )
        )
    concat = float(np.sqrt(np.sum(np.square(head_constants))))
    output = spectral_norm(params["W_O"]) * concat
    return output, concat, radius, head_constants


def mlp_lipschitz(params):
    """Global Euclidean Lipschitz constant of one MLP token map. tanh is 1-Lipschitz."""
    return spectral_norm(params["W1"]) * spectral_norm(params["W2"])


def factor_table(params):
    if params["eps"] <= 0.0:
        raise ValueError("eps > 0 is required")
    attention, concat, radius, heads = attention_factors(params)
    first = layer_norm_lipschitz(params["gamma1"], params["eps"])
    second = layer_norm_lipschitz(params["gamma2"], params["eps"])
    mlp = mlp_lipschitz(params)
    return {
        "L_A": attention,
        "L_concat": concat,
        "R1": radius,
        "head": heads,
        "L1": first,
        "L2": second,
        "L_M": mlp,
        "sigma_W1": spectral_norm(params["W1"]),
        "sigma_W2": spectral_norm(params["W2"]),
        "sigma_WO": spectral_norm(params["W_O"]),
        "w_norm": float(np.linalg.norm(params["w_read"])),
        "WO_w_norm": float(np.linalg.norm(params["W_O"] @ params["w_read"])),
        # M = hidden @ W2.T, so w·M = hidden·(W2.T w).
        "W2_w_norm": float(np.linalg.norm(params["W2"].T @ params["w_read"])),
    }


def branch_rate(table):
    """Coefficient of ||δ|| in ||ΔA_r|| + ||ΔM_r|| for a single-token edit."""
    coupled = table["L_A"] * table["L1"]
    return coupled + table["L_M"] * table["L2"] * (1.0 + coupled)


def aligned_branch_rate(table):
    """Same decomposition, with each branch charged by its readout-aligned gain.

    |w·ΔA| ≤ ||W_O w|| L_concat L1 ||δ||
    |w·ΔM| ≤ ||W2 w|| σ(W1) L2 (1 + L_A L1) ||δ||
    The MLP input radius still uses the full attention vector bound.
    """
    attention = table["WO_w_norm"] * table["L_concat"] * table["L1"]
    mlp = table["W2_w_norm"] * table["sigma_W1"] * table["L2"] * (1.0 + table["L_A"] * table["L1"])
    return attention + mlp


def isotropic_certificate(params, delta_norm):
    """|R| ≤ ||w|| (1 + L_A L1)(1 + L_M L2) ||δ||. Valid for every finite δ."""
    table = factor_table(params)
    state = (1.0 + table["L_A"] * table["L1"]) * (1.0 + table["L_M"] * table["L2"])
    return float(table["w_norm"] * state * delta_norm), table


def directional_certificate(params, delta, token):
    """|R| ≤ |1_{t=r} w·δ| + ||w|| branch_rate ||δ||.

    Valid for every finite δ and every edited token. When the edited token is
    not the readout, the skip inner product is absent.
    """
    delta = np.asarray(delta, dtype=float)
    delta_norm = float(np.linalg.norm(delta))
    table = factor_table(params)
    rate = branch_rate(table)
    skip = 0.0
    if token == params["read_token"]:
        skip = abs(float(params["w_read"] @ delta))
    value = skip + table["w_norm"] * rate * delta_norm
    return float(value), table, rate


def aligned_certificate(params, delta, token):
    """Directional certificate with per-path readout gains ||W_O w|| and ||W2 w||."""
    delta = np.asarray(delta, dtype=float)
    delta_norm = float(np.linalg.norm(delta))
    table = factor_table(params)
    rate = aligned_branch_rate(table)
    skip = 0.0
    if token == params["read_token"]:
        skip = abs(float(params["w_read"] @ delta))
    value = skip + rate * delta_norm
    return float(value), table, rate


def final_layer_norm_radius(gamma, beta, width):
    return layer_norm_radius(gamma, beta, width)


def output_scale_rate_threshold(params, target=0.5):
    """Largest α≥0 with branch_rate(α W_O, α W2) ≤ target, other factors fixed.

    rate(α) = α b + α^2 A with A, b taken from the unscaled factors.
    The positive root of A α^2 + b α - target = 0 is returned.
    """
    table = factor_table(params)
    quadratic = table["L_M"] * table["L2"] * table["L_A"] * table["L1"]
    linear = table["L_A"] * table["L1"] + table["L_M"] * table["L2"]
    if quadratic == 0.0:
        if linear == 0.0:
            return float("inf")
        return float(target / linear)
    return float((-linear + np.sqrt(linear**2 + 4.0 * quadratic * target)) / (2.0 * quadratic))


def final_readout_certificate(w, gamma, beta, width):
    """|m| ≤ ||w|| R_f and therefore |Δm| ≤ 2 ||w|| R_f, for m = w·LN_final(h)."""
    radius = final_layer_norm_radius(gamma, beta, width)
    return float(2.0 * np.linalg.norm(w) * radius), float(radius)
