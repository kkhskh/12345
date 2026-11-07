"""Previously silent no-op / skip paths must now fail closed."""

from __future__ import annotations

import numpy as np

from src.othello_r2 import mean_square_edit_safe, probe_direction_edit_safe, wrong_square_edit_safe
from tests.test_othello_r2_probe_parity import FakeLR, _pack


def test_probe_direction_uses_classes_not_row_index():
    pack = _pack()
    # Reverse class order on square 0 so row 0 is class 2.
    clf = pack["models"][0]
    pack["models"][0] = FakeLR(clf.coef_[::-1], clf.intercept_[::-1], np.array([2, 1, 0]))
    x = np.zeros(8)
    rec = np.zeros(64, dtype=int)
    tgt = rec.copy()
    tgt[0] = 2
    out = probe_direction_edit_safe(x, rec, tgt, pack["models"], 3.0)
    assert out["valid_edit"] == 1
    # Direction must be class2 minus class0 in the remapped rows: coef[0]-coef[2] of reversed = original coef[2]-coef[0]
    expected = pack["models"][0].coef_[0] - pack["models"][0].coef_[2]
    got = out["edited"] - x
    assert np.allclose(got / np.linalg.norm(got), expected / np.linalg.norm(expected), atol=1e-6)


def test_probe_direction_handles_sklearn_binary_one_row_coef():
    pack = _pack()
    d = pack["models"][0].coef_.shape[1]
    w = np.linspace(0.2, 1.3, d)
    pack["models"][5] = FakeLR(w.reshape(1, -1), np.array([0.1]), np.array([0, 1]))
    x = np.zeros(d)
    rec = np.zeros(64, dtype=int)
    tgt = rec.copy()
    tgt[5] = 1
    out = probe_direction_edit_safe(x, rec, tgt, pack["models"], 3.0)
    assert out["valid_edit"] == 1
    got = out["edited"] - x
    assert np.allclose(got / np.linalg.norm(got), w / np.linalg.norm(w), atol=1e-6)


def test_mean_square_does_not_claim_identity_as_edit():
    X = np.zeros((10, 4))
    y = np.zeros((10, 64), dtype=int)
    rec = np.zeros(64, dtype=int)
    tgt = rec.copy()
    tgt[:3] = 1
    out = mean_square_edit_safe(X, y, np.zeros(4), rec, tgt, 2.0)
    assert out["valid_edit"] == 0
    assert out["edited"] is None
    assert out["reason"] == "mean_square_insufficient_class_counts"


def test_wrong_square_none_is_failed_not_skipped():
    pack = _pack()
    rec = np.zeros(64, dtype=int)
    tgt = rec.copy()
    # Make every square an intended change so no wrong-square candidate remains except centers.
    tgt[:] = 1
    out = wrong_square_edit_safe(np.zeros(8), rec, tgt, pack["models"], 2.0, np.random.default_rng(0))
    assert out["status"] in {"EXECUTED", "FAILED"}
    if out["valid_edit"] == 0:
        assert out["edited"] is None
        assert out["reason"]
