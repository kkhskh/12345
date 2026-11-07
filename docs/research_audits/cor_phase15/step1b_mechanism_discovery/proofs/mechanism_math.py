"""Finite computational systems used by the Phase 1.5 Step 1B investigation.

These objects are not a Transformer, not an obstruction implementation, and not
a reconstruction of official COR maps. They exist so theorems and counterexamples
can be checked as arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def layer_norm(h: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    mu = h.mean()
    var = h.var()
    return (h - mu) / np.sqrt(var + eps)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def affine_linearization(f: Callable[[np.ndarray], np.ndarray], h0: np.ndarray, delta: np.ndarray, eps: float = 1e-6):
    """First-order affine map of f at h0, not the origin-through linear map J @ h."""
    h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    f0 = np.asarray(f(h0), dtype=np.float64).reshape(-1)
    cols = []
    for i in range(h0.size):
        step = np.zeros_like(h0)
        step[i] = eps
        cols.append((np.asarray(f(h0 + step), dtype=np.float64).reshape(-1) - f0) / eps)
    J = np.column_stack(cols)
    return f0, J, f0 + J @ delta


@dataclass
class TruncationSCM:
    """A two-node chain y = r(c), c = f(p), with optional extra parent-to-y path.

    Hard interventions use truncation: a clamped node is not recomputed from its
    parents. Descendants of a clamped node are recomputed from the clamped value.
    Ancestor and descendant may be clamped together. That is a defined operation.
    """

    f: Callable[[float], float]
    r: Callable[[float], float]

    def natural(self, p: float) -> tuple[float, float, float]:
        c = self.f(p)
        y = self.r(c)
        return p, c, y

    def do_parent(self, p0: float, p: float) -> tuple[float, float, float]:
        c = self.f(p)
        y = self.r(c)
        return p, c, y

    def do_child(self, p0: float, c: float) -> tuple[float, float, float]:
        y = self.r(c)
        return p0, c, y

    def do_both(self, p: float, c: float) -> tuple[float, float, float]:
        y = self.r(c)
        return p, c, y

    def interaction(self, p0: float, p: float, c: float) -> float:
        y0 = self.natural(p0)[2]
        ya = self.do_parent(p0, p)[2]
        yb = self.do_child(p0, c)[2]
        yab = self.do_both(p, c)[2]
        return yab - ya - yb + y0


def residual_block(h: np.ndarray, attn: Callable[[np.ndarray], np.ndarray], mlp: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Pre-norm residual step on a single vector standing in for one token."""
    h = np.asarray(h, dtype=np.float64)
    a = attn(layer_norm(h))
    h1 = h + a
    m = mlp(layer_norm(h1))
    return h1 + m


def attention_mix(values: np.ndarray, logits: np.ndarray) -> np.ndarray:
    w = softmax(logits)
    return w @ values


def linear_path_contributions(paths: np.ndarray, w: np.ndarray) -> dict:
    """paths: shape (k, d). Linear readout L = w · sum_i paths[i]."""
    paths = np.asarray(paths, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    total = paths.sum(axis=0)
    L = float(w @ total)
    contrib = paths @ w
    ablate = L - contrib
    return {"L": L, "contrib": contrib, "ablate": ablate, "total": total}


def majority_vote(bits: np.ndarray) -> int:
    bits = np.asarray(bits, dtype=np.int64)
    return 1 if bits.sum() > bits.size / 2 else 0


def pairwise_disagreement(bits: np.ndarray) -> int:
    bits = np.asarray(bits, dtype=np.int64)
    n = 0
    for i in range(bits.size):
        for j in range(i + 1, bits.size):
            n += int(bits[i] != bits[j])
    return n
