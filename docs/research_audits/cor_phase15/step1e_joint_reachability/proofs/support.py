"""Support comparison for a jointly reachable pair and for two linear maps.

The equality criterion is elementary. It is recorded because it is the
precise condition under which a product of marginals is exact for a separable
objective, and because Step 1D's product ball is a different set.
"""

from __future__ import annotations

import numpy as np


def positively_parallel(left, right, tol=1e-10):
    """True when one vector is a nonnegative multiple of the other."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.linalg.norm(left) <= tol or np.linalg.norm(right) <= tol:
        return True
    cross = float(np.linalg.norm(left) * np.linalg.norm(right))
    return bool(float(left @ right) >= cross * (1.0 - tol))


def linear_ball_supports(first, second, readout, radius):
    """Exact supports of a=Pδ, b=Qδ on the Euclidean ball of the given radius.

    Joint support: radius * ||(P+Q)^T w||.
    Sum of marginal supports: radius * (||P^T w|| + ||Q^T w||).
    """
    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    left = np.asarray(first, dtype=float).T @ np.asarray(readout, dtype=float)
    right = np.asarray(second, dtype=float).T @ np.asarray(readout, dtype=float)
    joint = radius * float(np.linalg.norm(left + right))
    separate = radius * (float(np.linalg.norm(left)) + float(np.linalg.norm(right)))
    return {
        "joint": joint,
        "separate": separate,
        "gap": separate - joint,
        "aligned": positively_parallel(left, right),
        "attention_gradient": left,
        "mlp_gradient": right,
    }


def triangle_gap_identity(left, right):
    """||p|| + ||q|| - ||p+q||, with the parallel defect of the squared norms."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    sum_norm = float(np.linalg.norm(left + right))
    separate = float(np.linalg.norm(left) + np.linalg.norm(right))
    squared_gap = separate**2 - float(np.linalg.norm(left + right) ** 2)
    inner_defect = (
        float(np.linalg.norm(left)) * float(np.linalg.norm(right)) - float(left @ right)
    )
    return {
        "gap": separate - sum_norm,
        "squared_gap": squared_gap,
        "inner_defect": inner_defect,
    }
