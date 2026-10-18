"""Unit tests for 08R2-FIBER geometry. No real checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.othello_r2 import LinearBoardDecoder
from src.othello_r2_fiber import (
    STATUS_COMPLETE,
    VERDICT_NOT_ESTABLISHED,
    VERDICT_SUPPORTED,
    contrast_parity_report,
    decide_verdict,
    evaluate_fiber_completeness,
    factorial_activations,
    factorial_effects,
    independent_contrasts,
    isolate_fiber_results_dir,
    js_divergence,
    pair_demonstrates_insufficiency,
    probe_invariance,
    reject_unexpected_preprocessing,
    svd_probe_geometry,
    y_functional,
)


class FakeLR:
    def __init__(self, coef, intercept, classes):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64)
        self.classes_ = np.asarray(classes, dtype=np.int64)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        w = self.coef_
        b = self.intercept_
        if w.ndim == 1:
            w = w.reshape(1, -1)
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 1), dtype=np.float64)
        if len(self.classes_) == 2 and w.shape[0] == 1:
            s = X @ w[0] + float(np.asarray(b).reshape(-1)[0])
            p1 = 1.0 / (1.0 + np.exp(-np.clip(s, -40, 40)))
            return np.stack([1.0 - p1, p1], axis=1)
        z = X @ w.T + b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _pack(d_model=12, seed=0):
    rng = np.random.default_rng(seed)
    models = []
    for j in range(64):
        if j < 3:
            coef = rng.normal(size=(3, d_model))
            intercept = rng.normal(size=(3,))
            models.append(FakeLR(coef, intercept, np.array([0, 1, 2])))
        elif j == 3:
            w = rng.normal(size=(d_model,))
            models.append(FakeLR(w.reshape(1, -1), np.array([0.2]), np.array([0, 1])))
        else:
            models.append(FakeLR(np.zeros((1, d_model)), np.array([0.0]), np.array([0])))
    return {"hook": "blocks.6.hook_resid_mid", "target": "rel", "method": "linear", "models": models}


def test_independent_contrasts_preserve_all_probe_probabilities():
    pack = _pack()
    dec = LinearBoardDecoder(pack)
    con = independent_contrasts(dec)
    assert con["n_contrasts"] == 3 * 2 + 1
    assert con["n_constant"] == 60
    assert con["n_binary"] == 1
    assert con["n_multiclass"] == 3
    x = np.random.default_rng(4).normal(size=(10, dec.d_model))
    report = contrast_parity_report(dec, pack["models"], x, con, 1e-10)
    assert report["ok"] == 1
    svd = svd_probe_geometry(con["W"], 1e-8, [1e-6, 1e-8, 1e-10])
    assert svd["has_meaningful_nullspace"] == 1
    assert svd["nullspace_dim"] == dec.d_model - svd["numerical_rank"]
    rng = np.random.default_rng(5)
    for _ in range(6):
        n = svd["P_N"] @ rng.normal(size=dec.d_model)
        p0 = dec.proba_np(x)
        p1 = dec.proba_np(x + n)
        assert np.max(np.abs(p0 - p1)) < 1e-8


def test_factorial_cells_are_probe_equivalent_in_pairs():
    pack = _pack()
    dec = LinearBoardDecoder(pack)
    con = independent_contrasts(dec)
    svd = svd_probe_geometry(con["W"], 1e-8, [1e-8])
    rng = np.random.default_rng(6)
    h_e = rng.normal(size=dec.d_model)
    h_d = rng.normal(size=dec.d_model)
    fact = factorial_activations(h_e, h_d, svd["P_R"], svd["P_N"])
    assert fact["h11_minus_donor"] < 1e-12
    assert fact["split_error"] < 1e-12
    inv01 = probe_invariance(dec, fact["cells"]["H01"], fact["cells"]["H00"], 1e-8)
    inv10 = probe_invariance(dec, fact["cells"]["H10"], fact["cells"]["H11"], 1e-8)
    assert inv01["ok"] == 1
    assert inv10["ok"] == 1


def test_unexpected_scaler_is_rejected():
    assert reject_unexpected_preprocessing({"models": [], "hook": "x"})["ok"] is True
    assert reject_unexpected_preprocessing({"models": [], "scaler": object()})["ok"] is False


def test_d_rows_are_not_primary():
    from src.othello_r2_fiber import PRIMARY

    assert "D_wrong_target" not in PRIMARY
    assert "C_same_legal_diff_board" not in PRIMARY


def test_verdict_and_demonstration_floors_are_predefined():
    assert pair_demonstrates_insufficiency(0.0, 0.0, 0.01, 0.05, 0.001) is False
    assert pair_demonstrates_insufficiency(0.002, 0.0, 0.0, 0.05, 0.001) is True
    assert pair_demonstrates_insufficiency(0.0, 0.0, 0.06, 0.05, 0.001) is True
    assert decide_verdict(8, 2, 32, 400, True, True) == VERDICT_SUPPORTED
    assert decide_verdict(8, 0, 32, 400, True, True) == VERDICT_NOT_ESTABLISHED
    assert decide_verdict(0, 0, 32, 400, True, True) == "FIBER_DIAGNOSTIC_INCONCLUSIVE"
    assert decide_verdict(8, 2, 32, 0, True, True) == "FIBER_DIAGNOSTIC_INCONCLUSIVE"


def test_y_and_js_are_finite():
    probs = np.ones(61) / 61.0
    y = y_functional(probs, [0, 1, 2], [1, 2, 3])
    assert y["Y"] == y["Y"]
    assert js_divergence(probs, probs) < 1e-12


def test_factorial_effects_recover_interaction():
    fx = factorial_effects(0.0, 0.2, 0.3, 0.8)
    assert fx["E_R"] == pytest.approx(0.2)
    assert fx["E_N"] == pytest.approx(0.3)
    assert fx["I"] == pytest.approx(0.3)


def test_fiber_dir_is_isolated_from_validation():
    out = isolate_fiber_results_dir("/tmp/results/toy_othello")
    assert out["results_dir"].endswith("08r2/fiber")
    assert "validation" not in out["results_dir"]


def test_completeness_fails_without_real_outputs(tmp_path: Path):
    fiber = tmp_path / "08r2" / "fiber"
    fiber.mkdir(parents=True)
    gate = evaluate_fiber_completeness(fiber, real_model=True)
    assert gate["ok"] == 0
    assert any(f.startswith("missing_file:") for f in gate["failures"])


def test_completeness_rejects_complete_status_without_valid_pairs(tmp_path: Path):
    fiber = tmp_path / "08r2" / "fiber"
    fiber.mkdir(parents=True)
    for name in (
        "08r2_fiber_dependency_manifest.json",
        "08r2_fiber_svd_report.json",
        "08r2_fiber_completion_status.json",
        "08r2_fiber_scientific_report.md",
    ):
        if name.endswith(".json"):
            (fiber / name).write_text(json.dumps({
                "status": STATUS_COMPLETE,
                "checkpoint_loaded": 1,
                "probe_loaded": 1,
                "parity_ok": 1,
                "nullspace_ok": 1,
                "hooks_ok": 1,
                "full_outputs_saved": 1,
                "n_factorial_cells": 0,
                "n_control_rows": 4,
                "n_valid": 0,
                "used_confirmation": 0,
            }), encoding="utf-8")
        else:
            (fiber / name).write_text("x", encoding="utf-8")
    for name in (
        "08r2_fiber_probe_contrast_parity.csv",
        "08r2_fiber_matched_pairs.csv",
        "08r2_fiber_factorial_interventions.csv",
        "08r2_fiber_natural_anchor.csv",
        "08r2_fiber_controls.csv",
        "08r2_fiber_bootstrap.csv",
    ):
        (fiber / name).write_text("a\n1\n", encoding="utf-8")
    np.savez(fiber / "08r2_fiber_full_logits.npz", x=np.zeros(2))
    np.savez(fiber / "08r2_fiber_probe_probabilities.npz", x=np.zeros(2))
    gate = evaluate_fiber_completeness(fiber, real_model=True)
    assert gate["ok"] == 0
    assert "complete_without_valid_pairs" in gate["failures"]
