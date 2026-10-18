"""Editor C implementation tests. Synthetic maps only."""

from __future__ import annotations

import numpy as np

from src.othello_r2 import apply_reusable_map, fit_reusable_map, project_budget


def _onehot_diff(rec, tgt):
    a = np.eye(3)[np.asarray(rec, dtype=int)].reshape(-1)
    b = np.eye(3)[np.asarray(tgt, dtype=int)].reshape(-1)
    return b - a


def test_fit_recovers_known_lowrank_map():
    rng = np.random.default_rng(0)
    d = 8
    true = rng.normal(size=(192, d))
    u, s, vt = np.linalg.svd(true, full_matrices=False)
    Wtrue = (u[:, :4] * s[:4]) @ vt[:4]
    Xsrc, Xtgt, Lsrc, Ltgt = [], [], [], []
    for _ in range(80):
        rec = rng.integers(0, 3, size=64)
        tgt = rec.copy()
        tgt[rng.choice(64, size=6, replace=False)] = rng.integers(0, 3, size=6)
        xs = rng.normal(size=d)
        xt = xs + _onehot_diff(rec, tgt) @ Wtrue
        Xsrc.append(xs)
        Xtgt.append(xt)
        Lsrc.append(rec)
        Ltgt.append(tgt)
    fit = fit_reusable_map(Xsrc, Xtgt, Lsrc, Ltgt, rank=4, ridge=1e-6, min_pairs=32)
    assert fit["status"] == "FIT"
    rec = Lsrc[0]
    tgt = Ltgt[0]
    out = apply_reusable_map(fit["W"], Xsrc[0], rec, tgt, budget=100.0)
    assert out["valid_edit"] == 1
    expected = Xsrc[0] + _onehot_diff(rec, tgt) @ Wtrue
    assert np.linalg.norm(out["edited"] - expected) < 0.25


def test_heldout_generalization_and_rank():
    rng = np.random.default_rng(1)
    d = 6
    Wtrue = rng.normal(size=(192, d))
    u, s, vt = np.linalg.svd(Wtrue, full_matrices=False)
    Wtrue = (u[:, :3] * s[:3]) @ vt[:3]
    train = []
    for _ in range(50):
        rec = rng.integers(0, 3, size=64)
        tgt = rec.copy()
        tgt[:5] = (tgt[:5] + 1) % 3
        xs = rng.normal(size=d)
        train.append((xs, xs + _onehot_diff(rec, tgt) @ Wtrue, rec, tgt))
    fit = fit_reusable_map([t[0] for t in train], [t[1] for t in train], [t[2] for t in train], [t[3] for t in train], rank=3, ridge=1e-5)
    assert fit["rank"] == 3
    rec = rng.integers(0, 3, size=64)
    tgt = rec.copy()
    tgt[10:15] = (tgt[10:15] + 2) % 3
    xs = rng.normal(size=d)
    out = apply_reusable_map(fit["W"], xs, rec, tgt, budget=50.0)
    assert out["used_donor_activation"] == 0
    assert out["valid_edit"] == 1


def test_norm_budget_and_fit_failure():
    rng = np.random.default_rng(2)
    rec = rng.integers(0, 3, size=64)
    tgt = rec.copy()
    tgt[0] = (tgt[0] + 1) % 3
    W = rng.normal(size=(192, 4))
    xs = np.zeros(4)
    out = apply_reusable_map(W, xs, rec, tgt, budget=0.5)
    assert out["delta_norm"] <= 0.5 + 1e-8
    fail = fit_reusable_map([xs], [xs], [rec], [rec], min_pairs=32)
    assert fail["status"] == "EDITOR_C_FIT_FAILED"
    assert fail["reason"] == "INSUFFICIENT_TRAIN_PAIRS"
    same = fit_reusable_map([xs] * 40, [xs] * 40, [rec] * 40, [rec] * 40, min_pairs=32)
    assert same["status"] == "EDITOR_C_FIT_FAILED"
    assert same["reason"] == "INSUFFICIENT_STATE_CHANGES"
    missing = apply_reusable_map(None, xs, rec, tgt, 1.0)
    assert missing["valid_edit"] == 0
    assert missing["edited"] is None
