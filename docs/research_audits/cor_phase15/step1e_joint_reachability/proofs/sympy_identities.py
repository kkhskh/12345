"""Symbolic checks that do not import the block.

The second route is SymPy together with the loop implementation. These
identities are the derivative and equality steps cited by the line theorem
and by the linear support theorem.
"""

from __future__ import annotations

import sympy as sp


def layer_norm_derivative_gap():
    """λ'(y) - ε / (y^2/2 + ε)^(3/2) simplifies to 0."""
    y, eps = sp.symbols("y eps", real=True, positive=True)
    lam = y / sp.sqrt(y**2 / 2 + eps)
    gap = sp.simplify(sp.diff(lam, y) - eps / (y**2 / 2 + eps) ** sp.Rational(3, 2))
    return gap


def tanh_cubic_gap():
    """Derivative of tanh(z) - z + z^3/3 is z^2 - tanh(z)^2, nonnegative in form."""
    z = sp.symbols("z", real=True)
    expression = sp.tanh(z) - z + z**3 / 3
    derivative = sp.simplify(sp.diff(expression, z))
    reduced = sp.simplify(derivative - (z**2 - sp.tanh(z) ** 2))
    return derivative, reduced


def triangle_equality_gap():
    """(||p|| + ||q||)^2 - ||p+q||^2 = 2(||p|| ||q|| - p·q) for vectors in R^2."""
    p1, p2, q1, q2 = sp.symbols("p1 p2 q1 q2", real=True)
    pnorm = sp.sqrt(p1**2 + p2**2)
    qnorm = sp.sqrt(q1**2 + q2**2)
    separate = (pnorm + qnorm) ** 2
    joint = (p1 + q1) ** 2 + (p2 + q2) ** 2
    inner = p1 * q1 + p2 * q2
    gap = sp.simplify(separate - joint - 2 * (pnorm * qnorm - inner))
    return gap


def run():
    derivative_gap = layer_norm_derivative_gap()
    _derivative, cubic_gap = tanh_cubic_gap()
    triangle_gap = triangle_equality_gap()
    return {
        "lambda_derivative": derivative_gap == 0,
        "tanh_cubic": cubic_gap == 0,
        "triangle": triangle_gap == 0,
    }
