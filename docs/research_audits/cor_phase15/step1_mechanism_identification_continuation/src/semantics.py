"""Common, noncircular source-semantics generator (§4) and operational consistency definitions (§5).

Every quantity here is a function of generator variables only. Nothing reads a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ binary generator


@dataclass
class Batch:
    a: np.ndarray
    b: np.ndarray
    p: np.ndarray            # prior token value (+-1); zeros where the task has no prior
    y: np.ndarray            # hard target in {+1,-1}
    soft: np.ndarray         # P(c) target (equals (y+1)/2 for hard tasks)
    task: str

    # generator-level semantic variables (§4)
    @property
    def strength(self):
        return np.abs(self.a), np.abs(self.b)

    @property
    def support(self):
        return np.sign(self.a), np.sign(self.b)

    @property
    def agree(self):
        return (self.a * self.b > 0).astype(int)


def target_binary(task, a, b, p):
    s = np.sign
    if task == "T1":
        y = np.where(a * b > 0, 1.0, -1.0); soft = (y + 1) / 2
    elif task == "T2":
        soft = 1.0 / (1.0 + np.exp(-(a + b))); y = np.where(a + b > 0, 1.0, -1.0)
    elif task == "T3":
        y = np.where(a * b > 0, s(a), p); soft = (y + 1) / 2
    elif task == "T4":
        y = np.where(a * b > 0, -s(a), s(a + b)); soft = (y + 1) / 2
    else:
        raise ValueError(task)
    y = np.where(y == 0, 1.0, y)
    return y.astype(float), soft.astype(float)


def sample_binary(task, n, rng, weak_frac=0.3):
    weak = rng.random(n) < weak_frac
    a = np.where(weak, rng.uniform(-0.5, 0.5, n), rng.uniform(-3, 3, n))
    b = np.where(weak, rng.uniform(-0.5, 0.5, n), rng.uniform(-3, 3, n))
    p = rng.choice([-1.0, 1.0], n) if task == "T3" else np.zeros(n)
    y, soft = target_binary(task, a, b, p)
    return Batch(a, b, p, y, soft, task)


def scenario_catalogue():
    """The §4 scenario list, as explicit generator settings (a, b, p) with their semantics."""
    return {
        "agreement for c": (1.5, 1.2, 0), "agreement for c'": (-1.5, -1.2, 0),
        "disagreement": (1.5, -1.2, 0), "unequal strengths": (2.5, 0.3, 0),
        "weak agreement": (0.08, 0.08, 0), "weak disagreement": (0.08, -0.08, 0),
        "strong disagreement": (2.8, -2.8, 0), "balanced disagreement": (1.0, -1.0, 0),
        "conflict, prior c": (1.0, -1.0, 1), "conflict, prior c'": (1.0, -1.0, -1),
        # T4: coherent but wrong; T2: conflicting but correct (stronger source wins under Bayes)
        "coherent-but-wrong (T4)": (1.5, 1.2, 0), "conflicting-but-correct (T2)": (2.0, -0.5, 0),
    }


# ------------------------------------------------------------------ multi-candidate generator (T5)


@dataclass
class BatchK:
    zA: np.ndarray
    zB: np.ndarray
    p: np.ndarray
    y: np.ndarray
    soft: np.ndarray
    task: str
    truth: np.ndarray


def sample_k(task, n, rng, K=3, p_true=0.6):
    Y = rng.integers(0, K, n)
    def src():
        hit = rng.random(n) < p_true
        Ys = np.where(hit, Y, rng.integers(0, K, n))
        mu = rng.uniform(0, 3, n)[:, None]
        return mu * np.eye(K)[Ys] + rng.normal(0, 1, (n, K))
    zA, zB = src(), src()
    p = rng.integers(0, K, n)
    if task == "T5a":
        lg = zA + zB
        soft = np.exp(lg - lg.max(1, keepdims=True)); soft /= soft.sum(1, keepdims=True)
        y = soft.argmax(1)
    elif task == "T5b":
        ag = zA.argmax(1) == zB.argmax(1)
        y = np.where(ag, zA.argmax(1), p)
        soft = np.eye(K)[y]
    else:
        raise ValueError(task)
    return BatchK(zA, zB, p, y, soft, task, Y)


# ------------------------------------------------------------------ §5 operational definitions


def semantic_agreement(zA, zB):
    """(D1) Do the sources support the same candidate under the generator's semantics?
    Domain: any K; reference variables: candidate labels; coordinate-free given the candidate set."""
    zA, zB = np.atleast_2d(zA), np.atleast_2d(zB)
    if zA.shape[-1] == 1:
        return (np.sign(zA[:, 0]) == np.sign(zB[:, 0])).astype(int)
    return (zA.argmax(-1) == zB.argmax(-1)).astype(int)


def representation_agreement(hA, hB, M=None):
    """(D2) Inner product under a fixed alignment M (identity by default). Coordinate dependent:
    changes under any non-orthogonal change of basis unless M transforms covariantly."""
    M = np.eye(hA.shape[-1]) if M is None else M
    return np.einsum("ni,ij,nj->n", hA, M, hB)


def candidate_relative_margin_agreement(zA, zB):
    """Candidate-relative alternative to D2: both sources' margins for the same top candidate.
    r = min over the two sources of (z_s[k*] - max_{j != k*} z_s[j]) with k* = argmax z_A."""
    k = zA.argmax(-1)
    idx = np.arange(len(k))
    def marg(z):
        top = z[idx, k]; zz = z.copy(); zz[idx, k] = -np.inf
        return top - zz.max(-1)
    return np.minimum(marg(zA), marg(zB))
