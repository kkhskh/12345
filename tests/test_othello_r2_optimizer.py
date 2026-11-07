"""Optimizer contract tests. Torch tests skip when the real stack is absent."""

from __future__ import annotations

import numpy as np
import pytest

from src.othello_r2 import LinearBoardDecoder, iterative_state_edit, project_budget


class FakeLR:
    def __init__(self, coef, intercept, classes):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64)
        self.classes_ = np.asarray(classes, dtype=np.int64)


def _tiny_decoder(d_model=8, seed=4):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(64):
        models.append(FakeLR(rng.normal(size=(3, d_model)), rng.normal(size=(3,)), np.array([0, 1, 2])))
    return LinearBoardDecoder({"hook": "blocks.6.hook_resid_mid", "target": "rel", "method": "linear", "models": models})


def test_steps_less_than_two_are_rejected():
    pytest.importorskip("torch")
    dec = _tiny_decoder()
    h = np.zeros(dec.d_model, dtype=np.float32)
    rec = np.zeros(64, dtype=np.int64)
    tgt = rec.copy()
    tgt[:4] = 1
    out = iterative_state_edit(h, tgt, rec, dec.as_torch(), steps=1, budget=2.0)
    assert out["valid_edit"] == 0
    assert out["edited"] is None
    assert out["reason"] == "invalid_step_count"


def test_iterative_editor_runs_multiple_steps_and_respects_budget():
    pytest.importorskip("torch")
    dec = _tiny_decoder()
    h = np.zeros(dec.d_model, dtype=np.float32)
    rec = np.zeros(64, dtype=np.int64)
    tgt = rec.copy()
    tgt[::2] = 1
    out = iterative_state_edit(h, tgt, rec, dec.as_torch(), steps=8, lr=0.5, budget=2.0, lambda_norm=0.0)
    assert out["valid_edit"] == 1
    assert out["edited"] is not None
    assert out["steps_completed"] >= 2
    assert len(out["traces"]) >= 2
    assert out["used_donor_activation"] == 0
    assert out["used_model_logits"] == 0
    assert out["delta_norm"] <= 2.0 + 1e-5
    assert np.isfinite(out["final_loss"])
    assert out["status"] == "EXECUTED"


def test_rowspace_constraint_stays_in_span():
    pytest.importorskip("torch")
    dec = _tiny_decoder()
    basis = dec.rowspace_basis()
    h = np.zeros(dec.d_model, dtype=np.float32)
    rec = np.zeros(64, dtype=np.int64)
    tgt = rec.copy()
    tgt[:8] = 2
    out = iterative_state_edit(h, tgt, rec, dec.as_torch(), steps=6, lr=0.4, budget=3.0, rowspace=basis)
    assert out["valid_edit"] == 1
    recon = (out["delta"] @ basis) @ basis.T
    assert np.allclose(out["delta"], recon, atol=1e-4)


def test_nonfinite_input_does_not_count_as_successful_edit():
    pytest.importorskip("torch")
    dec = _tiny_decoder()
    h = np.zeros(dec.d_model, dtype=np.float32)
    h[0] = np.nan
    rec = np.zeros(64, dtype=np.int64)
    tgt = rec.copy()
    tgt[:3] = 1
    out = iterative_state_edit(h, tgt, rec, dec.as_torch(), steps=4, budget=2.0)
    assert out["valid_edit"] == 0
    assert out["edited"] is None
    assert out["reason"] == "nonfinite_input_activation"


def test_invalid_labels_fail_closed():
    pytest.importorskip("torch")
    dec = _tiny_decoder()
    h = np.zeros(dec.d_model, dtype=np.float32)
    rec = np.zeros(64, dtype=np.int64)
    tgt = rec.copy()
    tgt[0] = 7
    out = iterative_state_edit(h, tgt, rec, dec.as_torch(), steps=4, budget=2.0)
    assert out["valid_edit"] == 0
    assert out["edited"] is None


def test_project_budget_is_closed_form_helper_not_an_editor():
    v = np.array([3.0, 4.0])
    out = project_budget(v, 2.5)
    assert np.isclose(np.linalg.norm(out), 2.5)
    assert np.allclose(project_budget(v, 10.0), v)
