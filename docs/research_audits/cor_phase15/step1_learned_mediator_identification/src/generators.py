"""B: controlled source-relationship generators.

Every variable is defined by the generator; nothing is read from a model.

Binary tasks (sources a, b in R; prior p in {-1,+1} where used):
  AGG-LIN   soft target sigma(a+b)                       (= historical T2)  depends on z = a+b only
  AGG-NL    y = +1 iff |a+b| < 1.5                        NEW: non-monotone h(z); aggregate-only control
  REL-XNOR  y = sign(ab)                                  (= historical T1)  needs information beyond z
  REL-GATE  y = sign(a) if ab>0 else p                    (= historical T3)  needs information beyond (z, p)
  REL-MIS   y = -sign(a) if ab>0 else sign(a+b)           (= historical T4)  agreement misleading
K = 3 tasks (z_A, z_B in R^3):
  K3-BAYES  soft softmax(z_A + z_B)                       (= historical T5a) aggregate-only
  K3-GATE   argmax z_A if argmax z_A == argmax z_B else p (= historical T5b)

Relation variable sought: s = 1[sources support the same candidate] (D1).
Aggregate statistic: z = a + b (binary) or z = z_A + z_B (K = 3), together with p where present.

Identifiability note (proved in MAIN doc): for binary sources at fixed z, every quantity is a function of
d = a - b, and s = 1[|d| < |z|]. So "uses agreement" and "uses the spread |a-b| thresholded at |z|" are the
same function on this input family; no intervention on these inputs can separate them.
"""
from __future__ import annotations

import numpy as np

HIST = {"AGG-LIN": "T2", "REL-XNOR": "T1", "REL-GATE": "T3", "REL-MIS": "T4", "K3-BAYES": "T5a", "K3-GATE": "T5b"}
BINARY = ["AGG-LIN", "AGG-NL", "REL-XNOR", "REL-GATE", "REL-MIS", "REL-XOR3"]
KTASKS = ["K3-BAYES", "K3-GATE"]


def target(task, a, b, p):
    s = np.sign
    if task == "AGG-LIN":
        soft = 1 / (1 + np.exp(-(a + b))); y = np.where(a + b > 0, 1.0, -1.0)
        return y, soft
    if task == "AGG-NL":
        y = np.where(np.abs(a + b) < 1.5, 1.0, -1.0)
    elif task == "REL-XNOR":
        y = np.where(a * b > 0, 1.0, -1.0)
    elif task == "REL-GATE":
        y = np.where(a * b > 0, s(a), p)
    elif task == "REL-MIS":
        y = np.where(a * b > 0, -s(a), s(a + b))
    elif task == "REL-XOR3":   # added after review R_causal F1: agreement is an intermediate that differs from the target
        y = s(a * b) * p
    else:
        raise ValueError(task)
    y = np.where(y == 0, 1.0, y).astype(float)
    return y, (y + 1) / 2


def needs_prior(task):
    return task in ("REL-GATE", "K3-GATE", "REL-XOR3")


def sample(task, n, rng, dist="train"):
    """dist: 'train' = 70% U[-3,3]^2 + 30% U[-0.5,0.5]^2 (the historical mixture);
             'gauss' = N(0, 1.2^2) clipped to [-3, 3]  (a NEW evaluation distribution, never used for selection)."""
    if dist == "train":
        weak = rng.random(n) < 0.3
        a = np.where(weak, rng.uniform(-0.5, 0.5, n), rng.uniform(-3, 3, n))
        b = np.where(weak, rng.uniform(-0.5, 0.5, n), rng.uniform(-3, 3, n))
    elif dist == "gauss":
        a = np.clip(rng.normal(0, 1.2, n), -3, 3); b = np.clip(rng.normal(0, 1.2, n), -3, 3)
    else:
        raise ValueError(dist)
    p = rng.choice([-1.0, 1.0], n) if needs_prior(task) else np.zeros(n)
    y, soft = target(task, a, b, p)
    return dict(a=a, b=b, p=p, y=y, soft=soft, z=a + b, s=(a * b > 0).astype(int))


def matched_pairs(task, n, rng, zmax=2.5, zmin=0.2):
    """Pairs with IDENTICAL z (and p) but opposite relation s. Both members lie in [-3,3]^2 (training support).
    For |z| < 3: agreement needs |d| < |z|; conflict needs |z| < |d| <= 6 - |z|."""
    z = rng.uniform(zmin, zmax, n) * rng.choice([-1, 1], n)
    da = rng.uniform(-1, 1, n) * np.abs(z) * 0.98
    dc = (np.abs(z) + 0.02 + rng.uniform(0, 1, n) * (6 - 2 * np.abs(z) - 0.04)) * rng.choice([-1, 1], n)
    p = rng.choice([-1.0, 1.0], n) if needs_prior(task) else np.zeros(n)
    A = dict(a=(z + da) / 2, b=(z - da) / 2, p=p)
    C = dict(a=(z + dc) / 2, b=(z - dc) / 2, p=p)
    for X in (A, C):
        X["y"], X["soft"] = target(task, X["a"], X["b"], X["p"]); X["z"] = X["a"] + X["b"]; X["s"] = (X["a"] * X["b"] > 0).astype(int)
    assert np.allclose(A["z"], C["z"]) and (A["s"] == 1).all() and (C["s"] == 0).all()
    assert np.abs(np.concatenate([A["a"], A["b"], C["a"], C["b"]])).max() <= 3 + 1e-9
    return A, C


def single_source_changes(task, n, rng):
    """Change A holding B (and vice versa): flip sign of one source keeping |.|, B / A fixed."""
    a = rng.uniform(-3, 3, n); b = rng.uniform(-3, 3, n)
    p = rng.choice([-1.0, 1.0], n) if needs_prior(task) else np.zeros(n)
    def pack(a, b):
        y, soft = target(task, a, b, p)
        return dict(a=a, b=b, p=p, y=y, soft=soft, z=a + b, s=(a * b > 0).astype(int))
    return pack(a, b), pack(-a, b), pack(a, -b)


# ----------------------------------------------------------------- K = 3


def sample_k(task, n, rng, dist="train", K=3, p_true=0.6):
    Y = rng.integers(0, K, n)
    def src():
        hit = rng.random(n) < p_true
        Ys = np.where(hit, Y, rng.integers(0, K, n))
        mu = rng.uniform(0, 3, n)[:, None] if dist == "train" else rng.uniform(0.5, 2, n)[:, None]
        return mu * np.eye(K)[Ys] + rng.normal(0, 1, (n, K))
    zA, zB = src(), src()
    p = rng.integers(0, K, n)
    if task == "K3-BAYES":
        lg = zA + zB; soft = np.exp(lg - lg.max(1, keepdims=True)); soft /= soft.sum(1, keepdims=True); y = soft.argmax(1)
    elif task == "K3-GATE":
        ag = zA.argmax(1) == zB.argmax(1); y = np.where(ag, zA.argmax(1), p); soft = np.eye(K)[y]
    else:
        raise ValueError(task)
    return dict(zA=zA, zB=zB, p=p, y=y, soft=soft, z=zA + zB, s=(zA.argmax(1) == zB.argmax(1)).astype(int))


def k_target(task, zA, zB, p):
    if task == "K3-GATE":
        ag = zA.argmax(1) == zB.argmax(1); return np.where(ag, zA.argmax(1), p)
    lg = zA + zB; return lg.argmax(1)


def matched_pairs_k(task, n, rng, tries=200):
    """Pairs (zA, zB) and (zA + delta, zB - delta): identical z, opposite argmax agreement."""
    base = sample_k(task, n * 4, rng)
    out_A, out_C = [], []
    for i in range(len(base["y"])):
        zA, zB, p = base["zA"][i], base["zB"][i], base["p"][i]
        s0 = zA.argmax() == zB.argmax()
        for _ in range(tries):
            dl = rng.normal(0, 1.5, 3)
            if ((zA + dl).argmax() == (zB - dl).argmax()) != s0:
                pair = [(zA, zB), (zA + dl, zB - dl)]
                if not s0:
                    pair = pair[::-1]
                out_A.append((*pair[0], p)); out_C.append((*pair[1], p))
                break
        if len(out_A) >= n:
            break
    def pack(lst):
        zA = np.array([x[0] for x in lst]); zB = np.array([x[1] for x in lst]); p = np.array([x[2] for x in lst])
        y = k_target(task, zA, zB, p)
        return dict(zA=zA, zB=zB, p=p, y=y, z=zA + zB, s=(zA.argmax(1) == zB.argmax(1)).astype(int))
    A, C = pack(out_A), pack(out_C)
    assert np.allclose(A["z"], C["z"]) and (A["s"] == 1).all() and (C["s"] == 0).all()
    return A, C


# ----------------------------------------------------------------- aggregate-only ceilings


def aggregate_only_ceiling(task, n=2_000_000, seed=0, bins=1200):
    """Best accuracy achievable by ANY function of the aggregate (z, p): majority label per fine z-bin (and p)."""
    rng = np.random.default_rng(seed)
    if task in KTASKS:
        return None
    d = sample(task, n, rng)
    zb = np.clip(((d["z"] + 6) / 12 * bins).astype(int), 0, bins - 1)
    key = zb * 3 + (d["p"] + 1).astype(int)
    pos = np.bincount(key, weights=(d["y"] > 0), minlength=bins * 3)
    cnt = np.bincount(key, minlength=bins * 3)
    return float(np.maximum(pos, cnt - pos).sum() / n)
