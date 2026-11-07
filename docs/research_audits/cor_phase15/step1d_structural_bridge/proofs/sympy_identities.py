"""Symbolic checks that do not call the numeric block.

1. Positive-epsilon LayerNorm scale identity: ε = λ^2 ε implies λ = 1 for λ > 0.
2. Softmax Jacobian on two coordinates attains operator norm 1/2.
3. The Euclidean Jacobian of tanh is at most 1.
"""

from __future__ import annotations

import sympy as sp


def layer_norm_scale_identity():
    eps, lam, variance = sp.symbols("eps lam variance", positive=True)
    # z' = λ z and s'^2 = λ^2 s^2, with s^2 = variance + eps, forces λ = 1.
    residual = sp.simplify(lam**2 * variance + eps - lam**2 * (variance + eps))
    factored = sp.factor(residual)
    solutions = sp.solve(sp.Eq(lam**2 * (variance + eps), lam**2 * variance + eps), lam)
    positive = [sol for sol in solutions if sol.is_real and sol > 0]
    return {
        "residual": str(residual),
        "factored": str(factored),
        "positive_solutions": [str(sol) for sol in positive],
    }


def softmax_two_point_norm():
    """At p = (1/2, 1/2) and v = (1/sqrt(2), -1/sqrt(2)), the quadratic form is 1/2."""
    v0, v1 = 1 / sp.sqrt(2), -1 / sp.sqrt(2)
    p0 = sp.Rational(1, 2)
    # v^T (diag(p) - p p^T) v = sum p_i v_i^2 - (sum p_i v_i)^2
    mean = p0 * v0 + p0 * v1
    second = p0 * v0**2 + p0 * v1**2
    form = sp.simplify(second - mean**2)
    return {
        "quadratic_form": str(form),
        "equals_one_half": bool(form == sp.Rational(1, 2)),
    }


def tanh_derivative_bound():
    x = sp.symbols("x", real=True)
    derivative = sp.diff(sp.tanh(x), x)
    # sech^2 = 1 - tanh^2 ≤ 1.
    identity = sp.simplify(derivative - (1 - sp.tanh(x) ** 2))
    return {"derivative": str(derivative), "gap_from_one_minus_tanh_sq": str(identity)}
