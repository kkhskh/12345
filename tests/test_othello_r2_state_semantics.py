"""Engineering tests for 08R2-STATE-SEMANTICS. No real checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.othello_r2_null_state import FEATURE_NAMES, fit_pca, ridge_fit, symbolic_state
from src.othello_r2_state_semantics import (
    VERDICT_BOARD,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    VERDICT_LEGAL,
    VERDICT_MIXED,
    ablation_delta,
    crossover_scores,
    decide_semantics_verdict,
    decompose_full_stage,
    direction_parity,
    feature_indices,
    isolate_sem_dir,
    pack_ablation,
    pack_full_from_nst,
    paired_boot,
    predict_zhat,
)


def test_feature_slices_drop_the_right_blocks():
    assert feature_indices("FULL").size == len(FEATURE_NAMES)
    board = set(feature_indices("BOARD_ONLY"))
    legal = set(feature_indices("LEGAL_ONLY"))
    stage = set(feature_indices("STAGE_ONLY"))
    assert 10 not in legal
    assert 200 not in board
    assert stage == {256, 257}
    assert 256 in board and 256 in legal
    row = {"rel_lab": np.zeros(64, dtype=np.int64), "legal": [3], "player": 1, "occupied": 8, "board": np.zeros(64)}
    b = symbolic_state(row)
    assert b[192 + 3] == 1.0
    assert b[feature_indices("LEGAL_ONLY")][3] == 1.0


def test_decompose_adds_and_stays_in_nullspace():
    rng = np.random.default_rng(0)
    d = 16
    P = np.eye(d)
    pca = {"mu": rng.normal(size=d), "V": np.eye(d)[:, :4], "k": 4}
    full = pack_ablation({"name": "FULL", "idx": feature_indices("FULL"), "ridge": ridge_fit(rng.normal(size=(20, 258)), rng.normal(size=(20, 4)), 10.0), "alpha": 10.0}, pca, P)
    stage = pack_ablation({"name": "STAGE_ONLY", "idx": feature_indices("STAGE_ONLY"), "ridge": ridge_fit(rng.normal(size=(20, 2)), rng.normal(size=(20, 4)), 10.0), "alpha": 10.0}, pca, P)
    h = rng.normal(size=d)
    b = rng.normal(size=258)
    dec = decompose_full_stage(h, b, full, stage)
    assert dec["sum_err"] < 1e-10
    assert np.allclose(P @ dec["delta_target"], dec["delta_target"])


def test_crossover_requires_both_arms():
    out = crossover_scores(0.10, 0.04, 0.08, 0.07)
    assert out["A_T"] == pytest.approx(0.06)
    assert out["A_W"] == pytest.approx(0.01)
    assert out["C"] == pytest.approx(0.07)
    assert out["both_positive"] == 1
    weak = crossover_scores(0.10, 0.04, 0.02, 0.05)
    assert weak["A_W"] < 0
    assert weak["both_positive"] == 0


def test_verdict_rules_and_isolate():
    gate = {k: 1 for k in ("reconstruction_ok", "split_ok", "integrity_ok", "invariance_ok", "js_match_ok", "coverage_ok")}
    gate["board_increment_ok"] = 1
    gate["legal_increment_ok"] = 0
    gate["legal_matches_full"] = 0
    assert decide_semantics_verdict(gate) == VERDICT_BOARD
    gate["board_increment_ok"] = 0
    gate["legal_matches_full"] = 1
    assert decide_semantics_verdict(gate) == VERDICT_LEGAL
    gate["board_increment_ok"] = 1
    gate["legal_increment_ok"] = 1
    assert decide_semantics_verdict(gate) == VERDICT_MIXED
    gate["coverage_ok"] = 0
    assert decide_semantics_verdict(gate) == VERDICT_INCONCLUSIVE
    gate["coverage_ok"] = 1
    gate["integrity_ok"] = 0
    assert decide_semantics_verdict(gate) == VERDICT_ENGINEERING
    out = isolate_sem_dir("/tmp/toy_othello")
    assert out["results_dir"].endswith("results/08r2/state_semantics")
    with pytest.raises(ValueError):
        isolate_sem_dir("/tmp/results/08r2/null_state_transfer")


def test_full_pack_reuses_nst_matrices():
    rng = np.random.default_rng(1)
    nst = {"mu": rng.normal(size=8), "V": rng.normal(size=(8, 4)), "W": rng.normal(size=(258, 4)), "c": rng.normal(size=4), "P_N": np.eye(8), "k": 4, "alpha": 10.0}
    pack = pack_full_from_nst(nst)
    assert pack["name"] == "FULL"
    assert pack["W"].shape == (258, 4)
    h = rng.normal(size=8)
    b = rng.normal(size=258)
    d1 = ablation_delta(h, b, pack)
    d2 = ablation_delta(h, b, pack)
    assert np.allclose(d1, d2)
    chk = direction_parity(d1, d2)
    assert chk["ok"] == 1
    boot = paired_boot([0.02, 0.03, 0.01], n_boot=200, seed=1)
    assert boot["n"] == 3
    assert boot["ci95_low"] <= boot["point"] <= boot["ci95_high"]
