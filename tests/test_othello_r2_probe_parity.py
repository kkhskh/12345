"""Synthetic decoder-parity checks. Not a real-model scientific result."""

from __future__ import annotations

import numpy as np

import pytest

from src.othello_r2 import LinearBoardDecoder, parity_report, sklearn_proba_pack


class FakeLR:
    def __init__(self, coef, intercept, classes):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64)
        self.classes_ = np.asarray(classes, dtype=np.int64)

    def predict_proba(self, X):
        z = np.asarray(X, dtype=np.float64) @ self.coef_.T + self.intercept_
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _pack(d_model=8, seed=0, missing_class_square=3):
    rng = np.random.default_rng(seed)
    models = []
    for j in range(64):
        if j == missing_class_square:
            coef = rng.normal(size=(2, d_model))
            intercept = rng.normal(size=(2,))
            classes = np.array([0, 1])
        else:
            coef = rng.normal(size=(3, d_model))
            intercept = rng.normal(size=(3,))
            classes = np.array([0, 1, 2])
        models.append(FakeLR(coef, intercept, classes))
    return {"hook": "blocks.6.hook_resid_mid", "target": "rel", "method": "linear", "models": models}


def test_pytorch_optional_numpy_decoder_matches_fake_sklearn():
    pack = _pack()
    dec = LinearBoardDecoder(pack)
    x = np.random.default_rng(1).normal(size=(12, dec.d_model))
    report = parity_report(dec, pack["models"], x)
    assert report["max_abs_proba"] < 1e-10
    assert report["mean_abs_proba"] < 1e-12
    assert report["square_agree"] == 1.0
    assert report["board_agree"] == 1.0
    assert dec.class_sets[3] == (0, 1)
    assert dec.class_sets[0] == (0, 1, 2)


def test_missing_class_has_near_zero_probability():
    pack = _pack()
    dec = LinearBoardDecoder(pack)
    x = np.zeros((1, dec.d_model))
    p = dec.proba_np(x)[0, 3]
    assert p[2] < 1e-6
    assert np.isclose(p.sum(), 1.0)
    sk = sklearn_proba_pack(pack["models"], x)[0, 3]
    assert sk[2] == 0.0


def test_torch_decoder_matches_sklearn_fixture_not_only_numpy():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 8))
    models = []
    for j in range(64):
        n_cls = 2 if j % 11 == 0 else 3
        y = np.resize(np.arange(n_cls), 80)
        rng.shuffle(y)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X, y)
        models.append(clf)
    pack = {"hook": "blocks.6.hook_resid_mid", "target": "rel", "method": "linear", "models": models}
    dec = LinearBoardDecoder(pack)
    from src.othello_r2 import torch_parity_report

    report = torch_parity_report(dec, models, X[:10])
    assert report["backend"] == "torch_vs_sklearn"
    assert report["max_abs_proba"] < 1e-4
    assert report["board_agree"] == 1.0


def test_rowspace_is_orthonormal_and_records_rank():
    pack = _pack()
    dec = LinearBoardDecoder(pack)
    q = dec.rowspace_basis()
    gram = q.T @ q
    assert q.shape[0] == dec.d_model
    assert q.shape[1] >= 1
    assert np.allclose(gram, np.eye(q.shape[1]), atol=1e-8)
