"""Finite level set of LayerNorm. The Jacobian kernel is an earlier result.

This module records the finite statement and an independent SymPy check of the
scale identity that forces the multiplier to be 1 when eps>0.
"""

from __future__ import annotations

import numpy as np

from proofs.block import layer_norm


def finite_level_set_is_constants(h, delta, gamma, beta, eps):
    """Return whether LN(h+delta) equals LN(h), and whether delta is constant.

    The caller supplies a case. The theorem, proved in the notes, is:

    If eps>0 and at most one gain is zero, then equality holds if and only if
    delta is a constant vector. Two or more zero gains leave a larger level set.
    """
    h = np.asarray(h, dtype=float)
    delta = np.asarray(delta, dtype=float)
    left = layer_norm(h + delta, gamma, beta, eps)
    right = layer_norm(h, gamma, beta, eps)
    equal = np.allclose(left, right, atol=1e-10, rtol=1e-10)
    constant = np.allclose(delta, delta.mean(), atol=1e-10)
    return equal, constant


def two_zero_gain_counterexample(eps=1.0):
    """Algebraic non-constant delta with LN(h+delta)=LN(h).

    d=4, gamma=(1,1,0,0), h=(1,-1,0,0), b=sqrt(2),
    a=sqrt(1+b^2/2)=sqrt(2), h'=(a,-a,b,-b).
    Both vectors are centered. The normalized first two coordinates agree and
    the last two gains are zero, so the LayerNorm outputs agree.
    """
    b = np.sqrt(2.0)
    a = np.sqrt(1.0 + 0.5 * b * b)
    h = np.array([1.0, -1.0, 0.0, 0.0])
    h_prime = np.array([a, -a, b, -b])
    delta = h_prime - h
    gamma = np.array([1.0, 1.0, 0.0, 0.0])
    beta = np.zeros(4)
    return h, delta, gamma, beta, float(eps)


def sympy_scale_identity_forces_lambda_one():
    """Symbolic check: z' = λ z and s' = λ s imply eps = λ^2 eps.

    This is the algebraic step in the finite-kernel proof. It does not import
    the numeric LayerNorm.
    """
    import sympy as sp

    eps, lam, v = sp.symbols("eps lam v", positive=True)
    s = sp.sqrt(v + eps)
    # z' = lam z implies v' = lam**2 v, hence s'^2 = lam**2 v + eps.
    # The normalized-vector identity also requires s' = lam s.
    residual = sp.simplify((lam**2 * v + eps) - (lam * s) ** 2)
    # residual = eps - lam**2 eps = eps (1-lam**2)
    factored = sp.factor(residual)
    solutions = sp.solve(sp.Eq(residual, 0), lam)
    return {
        "residual": str(factored),
        "positive_solutions": [str(s) for s in solutions if s.is_positive],
    }
