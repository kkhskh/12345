"""Engineering tests for 08R2-NULL-STATE-TRANSFER. No real checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.othello_r2_fiber_spec import y_components
from src.othello_r2_null_state import (
    FEATURE_NAMES,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    VERDICT_NOT_SUPPORTED,
    VERDICT_SUPPORTED,
    audit_saved_y,
    decide_nst_verdict,
    donor_separation_audit,
    isolate_nst_dir,
    learned_null_delta,
    learned_uses_only_allowed_inputs,
    pack_model,
    predict_z,
    rescore_y_from_logits,
    resolve_pair_rows,
    saved_y_present,
    select_ridge_pca,
    split_leakage_audit,
    stage_mean_name,
    symbolic_state,
)


def test_symbolic_state_uses_only_simulator_fields():
    rel = np.zeros(64, dtype=np.int64)
    rel[0] = 1
    rel[1] = 2
    row = {"rel_lab": rel, "legal": [10, 11], "player": 1, "occupied": 16, "board": np.zeros(64)}
    b = symbolic_state(row)
    assert b.shape == (len(FEATURE_NAMES),)
    assert b[1] == 1.0
    assert b[192 + 10] == 1.0
    assert b[-2] == 1.0
    assert b[-1] == pytest.approx(16 / 64)


def test_learned_delta_signature_has_no_donor():
    assert "h_d" not in learned_uses_only_allowed_inputs()
    rng = np.random.default_rng(0)
    d = 16
    P = np.eye(d)
    Z = rng.normal(size=(40, d))
    B = rng.normal(size=(40, 8))
    games = [f"g{i // 4}" for i in range(40)]
    sel = select_ridge_pca(Z, B, games, ks=(4,), alphas=(1.0,), n_folds=2, seed=1)
    model = pack_model(sel["pca"], sel["ridge"], P, {(1, 2): Z.mean(0)})
    h_e = rng.normal(size=d)
    b = rng.normal(size=8)
    d1 = learned_null_delta(h_e, b, model)
    d2 = learned_null_delta(h_e, b, model)
    assert np.allclose(d1, d2)
    h_d = rng.normal(size=d)
    d3 = learned_null_delta(h_e, b, model)
    assert np.allclose(d1, d3)
    assert h_d is not None
    assert np.allclose(model["P_N"] @ d1, d1)
    sep = donor_separation_audit()
    assert sep["ok"] == 1
    assert sep["has_donor_argument"] == 0


def test_cv_selection_does_not_need_causal_y():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 12))
    B = rng.normal(size=(30, 6))
    games = [f"g{i // 3}" for i in range(30)]
    sel = select_ridge_pca(Z, B, games, ks=(4, 8), alphas=(0.1, 10.0), n_folds=3, seed=2)
    assert "k" in sel["best"]
    pred = predict_z(B, sel["pca"], sel["ridge"])
    assert pred.shape == Z.shape


def test_rescore_y_matches_direct_components():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=61)
    rec = [10, 11, 12]
    tgt = [12, 13, 14]
    got = rescore_y_from_logits(logits, rec, tgt)
    exp = y_components(__import__("src.othello_r2_fiber", fromlist=["softmax"]).softmax(logits), rec, tgt)
    assert got["Y"] == pytest.approx(exp["Y"])
    chk = audit_saved_y({"Y": exp["Y"], "G": exp["G"], "Q": exp["Q"]}, got)
    assert chk["ok"] == 1
    bad = audit_saved_y({"Y": exp["Y"] + 0.01, "G": exp["G"], "Q": exp["Q"]}, got)
    assert bad["ok"] == 0
    missing = audit_saved_y({"cell": "editor_A"}, got)
    assert missing["ok"] is None
    assert missing["compared"] == 0
    assert not saved_y_present({"cell": "editor_A", "Y": ""})


def test_resolve_pair_uses_frozen_index_not_last_prefix():
    val = [
        {"game_id": "g1", "legal": [1], "board": [0]},
        {"game_id": "g1", "legal": [2], "board": [1]},
        {"game_id": "g2", "legal": [3], "board": [2]},
        {"game_id": "g2", "legal": [4], "board": [3]},
    ]
    got = resolve_pair_rows(val, {"recip": 0, "donor": 2, "recip_game": "g1", "target_game": "g2"})
    assert got["rec"]["legal"] == [1]
    assert got["tgt"]["legal"] == [3]
    assert got["last_prefix_same_as_pair"] == 0


def test_isolate_avoids_prior_experiments():
    out = isolate_nst_dir("/tmp/toy_othello")
    assert out["results_dir"].endswith("results/08r2/null_state_transfer")
    assert "fiber_cost" not in Path(out["results_dir"]).parts
    assert "fiber_spec" not in Path(out["results_dir"]).parts
    with pytest.raises(ValueError):
        isolate_nst_dir("/tmp/results/08r2/fiber_cost_fixed_js")
    packed = pack_model(
        {"mu": np.zeros(4), "V": np.eye(4), "k": 4},
        {"W": np.zeros((3, 4)), "c": np.zeros(4), "alpha": 1.0},
        np.eye(4),
        {(1, 3): np.ones(4)},
    )
    assert "1_3" in packed["stage_means"]
    assert stage_mean_name((1, 3)) == "1_3"


def test_split_leakage_detects_eval_overlap():
    ok = split_leakage_audit(["tr1"], ["tr1"], ["r1"], ["t1"], ["w1"], ["old"], ["hold"])
    assert ok["ok"] == 1
    leak = split_leakage_audit(["r1"], ["tr1"], ["r1"], ["t1"], ["w1"], ["old"], ["hold"])
    assert leak["ok"] == 0
    assert leak["leaks"]["train_vs_eval"] == ["r1"]


def test_verdict_requires_transfer_not_just_r2():
    gate = {k: 1 for k in (
        "split_ok", "no_donor_leak", "invariance_ok", "js_match_ok", "coverage_ok",
        "mean_vs_wrong_ge_floor", "ci_wrong_gt_0", "beats_random", "beats_mean_state",
        "beats_editor", "components_vs_editor_ok", "integrity_ok",
    )}
    assert decide_nst_verdict(gate) == VERDICT_SUPPORTED
    gate["mean_vs_wrong_ge_floor"] = 0
    assert decide_nst_verdict(gate) == VERDICT_NOT_SUPPORTED
    gate["mean_vs_wrong_ge_floor"] = 1
    gate["coverage_ok"] = 0
    assert decide_nst_verdict(gate) == VERDICT_INCONCLUSIVE
    gate["coverage_ok"] = 1
    gate["integrity_ok"] = 0
    assert decide_nst_verdict(gate) == VERDICT_ENGINEERING
