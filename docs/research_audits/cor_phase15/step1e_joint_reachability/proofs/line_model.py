"""Scalar reduction of the structural block along its invariant line.

These formulas do not call the block. They are the object named by the
invariant-line theorems. Width 2, mean-zero unit axis n = (1, -1)/sqrt(2),
and a state h = y n have LayerNorm image λ(y) n, where

    λ(y) = y / sqrt(y^2 / 2 + ε),    λ'(y) = ε / (y^2 / 2 + ε)^(3/2).

A fixed causal pattern and an isotropic value map of gain μ send a
single-token edit of coordinate t to an attention readout μ Δλ. The rank-one
MLP with row -α w and column s_gain w, w = n, has readout
s_gain tanh(-α λ(y1)).
"""

from __future__ import annotations

import math


def lam(y, eps):
    return y / math.sqrt(y * y / 2.0 + eps)


def lam_prime(y, eps):
    scale = math.sqrt(y * y / 2.0 + eps)
    return eps / (scale**3)


def attention_readout(step, mu, y0, eps):
    return mu * (lam(y0 + step, eps) - lam(y0, eps))


def hidden_coordinate(step, mu, y0, y1, eps, direct_skip):
    attention = attention_readout(step, mu, y0, eps)
    skip = step if direct_skip else 0.0
    return y1 + skip + attention


def mlp_readout(step, mu, alpha, s_gain, y0, y1, eps, direct_skip):
    start = s_gain * math.tanh(-alpha * lam(y1, eps))
    arrived = hidden_coordinate(step, mu, y0, y1, eps, direct_skip)
    end = s_gain * math.tanh(-alpha * lam(arrived, eps))
    return end - start


def branch_readout(step, mu, alpha, s_gain, y0, y1, eps, direct_skip):
    attention = attention_readout(step, mu, y0, eps)
    mlp = mlp_readout(step, mu, alpha, s_gain, y0, y1, eps, direct_skip)
    return attention, mlp, attention + mlp


def full_raw_response(step, mu, alpha, s_gain, y0, y1, eps, direct_skip, skip_gain):
    """skip_gain is w · n. It is 1 when the readout is the line axis."""
    _attention, _mlp, branch = branch_readout(
        step, mu, alpha, s_gain, y0, y1, eps, direct_skip
    )
    skip = skip_gain * step if direct_skip else 0.0
    return skip + branch


def attention_derivative(step, mu, y0, eps):
    return mu * lam_prime(y0 + step, eps)


def mlp_derivative(step, mu, alpha, s_gain, y0, y1, eps, direct_skip):
    arrived = hidden_coordinate(step, mu, y0, y1, eps, direct_skip)
    slope = lam_prime(arrived, eps)
    sech2 = 1.0 / math.cosh(-alpha * lam(arrived, eps)) ** 2
    hidden_slope = (1.0 if direct_skip else 0.0) + attention_derivative(step, mu, y0, eps)
    return s_gain * sech2 * (-alpha) * slope * hidden_slope


def separate_marginal_upper(step_end, mu, alpha, s_gain, y0, y1, eps, direct_skip):
    """Exact sum of marginal maxima on [0, step_end] when μ>0, α>0, s_gain>0.

    Attention readout is strictly increasing and MLP readout is strictly
    decreasing, so the maxima are the right endpoint and 0.
    """
    if step_end < 0.0:
        raise ValueError("this marginal formula is stated for a forward segment")
    attention = attention_readout(step_end, mu, y0, eps)
    return attention + 0.0


def special_joint_branch_upper():
    """Exact joint maximum of the branch readout under the special-gain theorem."""
    return 0.0
