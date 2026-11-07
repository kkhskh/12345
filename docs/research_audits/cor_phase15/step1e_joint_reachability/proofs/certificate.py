"""Weight-side certificate for the structural line.

The functions below do not evaluate a perturbed block and do not import the
forward map. The number 0 is a bound only under the special-gain hypotheses
proved in 05_support_function_theorems.md. The general strict comparison is
the disjoint-argmax statement, whose numerical value still depends on the
attention marginal.
"""

from __future__ import annotations

import math


def factorial(n):
    value = 1
    for factor in range(2, n + 1):
        value *= factor
    return value


def exponential_lower_beats_half_sqrt_two(order=8):
    """Integer proof that a partial sum of e already forces tanh(1) > sqrt(2)/2.

    Let s = sum_{k=0}^{order} 1/k! = num/den. Then e > s, so e^2 > s^2 and
    tanh(1) = (e^2-1)/(e^2+1) > (s^2-1)/(s^2+1). The comparison of that
    rational with sqrt(2)/2 squares to an integer inequality.
    """
    if order < 8:
        raise ValueError("the recorded factorial bound uses order at least 8")
    den = factorial(order)
    num = sum(den // factorial(k) for k in range(order + 1))
    left = 2 * (num * num - den * den) ** 2
    right = (num * num + den * den) ** 2
    return left > right


def special_gain_constants():
    """Integer facts used by the case split for ε = 10^{-5}, μ = 1/2, α = s = 1."""
    eps_num, eps_den = 1, 100000
    # (1/100)^2 > 2ε, so λ(1/100) > 1.
    threshold = (1 * eps_den) > (10000 * 2 * eps_num)
    # k^2 = 50000 / 12 and 4k > 3, i.e. 16 k^2 > 9.
    cubic = (16 * 50000) > (9 * 12)
    tanh_gap = exponential_lower_beats_half_sqrt_two()
    return {
        "lambda_threshold": threshold,
        "cubic_room": cubic,
        "tanh_gap": tanh_gap,
        "epsilon": "1/100000",
        "mu": "1/2",
        "alpha": 1,
        "s_gain": 1,
    }


def hypotheses_hold(record):
    """Weight hypotheses for the invariant-line sign theorem. No intervention."""
    required = (
        record["mu"] > 0.0
        and record["alpha"] > 0.0
        and record["s_gain"] > 0.0
        and record["eps"] > 0.0
    )
    return required


def special_gain(record):
    constants = special_gain_constants()
    if not all(constants[name] for name in ("lambda_threshold", "cubic_room", "tanh_gap")):
        return False
    return (
        hypotheses_hold(record)
        and math.isclose(record["mu"], 0.5, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(record["alpha"], 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(record["s_gain"], 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(record["eps"], 1e-5, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(record["y0"], 0.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(record["y1"], 0.0, rel_tol=0.0, abs_tol=0.0)
    )


def joint_branch_certificate(record):
    """Return the proved upper bound on the positive branch sum, or None."""
    if special_gain(record):
        return 0.0
    return None


def separate_branch_certificate(record, attention_marginal_max):
    """Product of the exact marginal maxima under the sign hypotheses.

    The MLP marginal maximum on a forward segment is 0. The attention marginal
    maximum is supplied by the caller because it is λ, not a joint evaluation.
    """
    if not hypotheses_hold(record):
        raise ValueError("the separate marginal sum is stated for positive gains")
    if attention_marginal_max < 0.0:
        raise ValueError("attention marginal maximum cannot be negative in this regime")
    return float(attention_marginal_max)
