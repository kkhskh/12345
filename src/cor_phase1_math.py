"""Finite-dimensional COR algebra used by the Phase 1 audit.

These objects are synthetic graph-consistency constructions. They are not a
Transformer, not a causal identification result, and not an implementation of
obstruction energy on Othello-GPT.
"""

from __future__ import annotations

import numpy as np


def coboundary(rho: np.ndarray) -> np.ndarray:
    """Two-node coboundary: (d s) = rho s_u - s_v, s = (s_u, s_v)."""
    rho = np.asarray(rho, dtype=np.float64)
    r, q = rho.shape
    d = np.zeros((r, q + r), dtype=np.float64)
    d[:, :q] = rho
    d[:, q:] = -np.eye(r)
    return d


def laplacian(d: np.ndarray, W: np.ndarray) -> np.ndarray:
    return d.T @ W @ d


def obstruction_O(s: np.ndarray, d: np.ndarray, W: np.ndarray) -> float:
    r = d @ s
    return float(r @ W @ r)


def distance_D(s: np.ndarray, d: np.ndarray, M: np.ndarray, tol: float = 1e-12) -> dict:
    """M-distance from s to ker d, plus the pseudoinverse formula."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    M = np.asarray(M, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    gram = d @ np.linalg.solve(M, d.T)
    r = d @ s
    gplus = np.linalg.pinv(gram, rcond=tol)
    d_formula = float(r @ gplus @ r)
    k = kernel_basis(d, tol)
    if k.shape[1] == 0:
        d_proj = float(s @ M @ s)
        g_hat = np.zeros_like(s)
    else:
        gram_k = k.T @ M @ k
        y = np.linalg.solve(gram_k, k.T @ M @ s)
        g_hat = k @ y
        diff = s - g_hat
        d_proj = float(diff @ M @ diff)
    return {
        "D_formula": d_formula,
        "D_projection": d_proj,
        "g_hat": g_hat,
        "residual": s - g_hat,
        "gram": gram,
        "gplus": gplus,
    }


def kernel_basis(d: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    _, sv, vh = np.linalg.svd(d, full_matrices=True)
    if sv.size == 0:
        return np.eye(d.shape[1])
    rank = int(np.sum(sv > max(tol, sv[0] * 1e-12)))
    return vh[rank:].T


def spectral_bounds(d: np.ndarray, W: np.ndarray, M: np.ndarray, tol: float = 1e-10) -> dict:
    L = 0.5 * (laplacian(d, W) + laplacian(d, W).T)
    evals_m, evecs_m = np.linalg.eigh(M)
    minv = evecs_m @ np.diag(evals_m**-0.5) @ evecs_m.T
    reduced = minv @ L @ minv
    reduced = 0.5 * (reduced + reduced.T)
    evals = np.linalg.eigvalsh(reduced)
    scale = max(1.0, float(np.max(np.abs(evals))))
    pos = evals[evals > tol * scale]
    return {
        "lambda_min_plus": float(pos.min()) if pos.size else float("nan"),
        "lambda_max": float(pos.max()) if pos.size else float("nan"),
        "kappa": float(pos.max() / pos.min()) if pos.size else float("nan"),
        "n_pos": int(pos.size),
    }


def equal_weighting_makes_O_eq_D(d: np.ndarray, M: np.ndarray, s: np.ndarray) -> dict:
    pack = distance_D(s, d, M)
    Wstar = pack["gplus"]
    return {
        "O": obstruction_O(s, d, Wstar),
        "D": pack["D_formula"],
        "Wstar": Wstar,
    }
