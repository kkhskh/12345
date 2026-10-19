"""Engineering tests for 08R2-SYMMETRIC-TARGET. No real checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.othello_r2_state_semantics import crossover_scores
from src.othello_r2_symmetric_target import (
    VERDICT_ANCHOR,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    VERDICT_NOT,
    VERDICT_SUPPORTED,
    anchor_dependence_ok,
    arithmetic_levels,
    decide_sym_verdict,
    isolate_sym_dir,
    natural_symmetric_ok,
    select_triplets,
    stage_only_identical,
    triplet_hash,
)


def test_arithmetic_correction_is_level_not_mechanism():
    out = arithmetic_levels(-0.582, -0.606, -0.636)
    assert out["FULL_minus_EditorA"] == pytest.approx(0.054)
    assert out["STAGE_ONLY_minus_EditorA"] == pytest.approx(0.030)
    assert out["FULL_minus_STAGE_ONLY"] == pytest.approx(0.024)
    assert "not independent" in out["note"]


def test_stage_only_identity_uses_player_and_occ():
    b_t = np.zeros(258)
    b_w = np.zeros(258)
    b_t[256], b_t[257] = 1.0, 0.25
    b_w[256], b_w[257] = 1.0, 0.25
    b_t[10] = 1.0
    b_w[20] = 1.0
    assert stage_only_identical(b_t, b_w)
    b_w[257] = 0.5
    assert not stage_only_identical(b_t, b_w)


def test_verdict_requires_both_natural_arms():
    gate = {k: 1 for k in ("split_ok", "integrity_ok", "invariance_ok", "js_match_ok", "coverage_ok", "reconstruction_ok")}
    gate["natural_symmetric_ok"] = 1
    gate["anchor_dependence_ok"] = 0
    assert decide_sym_verdict(gate) == VERDICT_SUPPORTED
    gate["natural_symmetric_ok"] = 0
    gate["anchor_dependence_ok"] = 1
    assert decide_sym_verdict(gate) == VERDICT_ANCHOR
    gate["anchor_dependence_ok"] = 0
    assert decide_sym_verdict(gate) == VERDICT_NOT
    gate["coverage_ok"] = 0
    assert decide_sym_verdict(gate) == VERDICT_INCONCLUSIVE
    gate["coverage_ok"] = 1
    gate["integrity_ok"] = 0
    assert decide_sym_verdict(gate) == VERDICT_ENGINEERING


def test_natural_gate_and_anchor_rule():
    ok = {"point": 0.02, "ci95_low": 0.01}
    weak = {"point": 0.004, "ci95_low": -0.001}
    assert natural_symmetric_ok(ok, ok, {"point": 0.04}, 0.7) == 1
    assert natural_symmetric_ok(ok, weak, {"point": 0.02}, 0.7) == 0
    nat = {"A_T": ok, "A_W": weak}
    at = {"A_T": {"point": 0.03}, "A_W": {"point": 0.00}}
    aw = {"A_T": {"point": 0.00}, "A_W": {"point": 0.03}}
    assert anchor_dependence_ok(nat, at, aw) == 1
    xo = crossover_scores(0.1, 0.05, 0.08, 0.07)
    assert xo["both_positive"] == 1


def test_isolate_and_selection():
    out = isolate_sym_dir("/tmp/toy_othello")
    assert out["results_dir"].endswith("results/08r2/symmetric_target")
    with pytest.raises(ValueError):
        isolate_sym_dir("/tmp/results/08r2/state_semantics")
    found = {
        "A_legal_diff": [
            {"recip": 0, "target_t": 1, "target_w": 2, "r2_type": "A_legal_diff", "recip_game": "r0", "target_t_game": "t0", "target_w_game": "w0", "hamming_t": 12, "hamming_w": 14},
            {"recip": 3, "target_t": 4, "target_w": 5, "r2_type": "A_legal_diff", "recip_game": "r1", "target_t_game": "t1", "target_w_game": "w1", "hamming_t": 20, "hamming_w": 22},
        ],
        "B_similar_board_legal_change": [
            {"recip": 6, "target_t": 7, "target_w": 8, "r2_type": "B_similar_board_legal_change", "recip_game": "r2", "target_t_game": "t2", "target_w_game": "w2", "hamming_t": 4, "hamming_w": 6},
        ],
    }
    sel = select_triplets(found, 1, 1, 7182, n_trials=20)
    assert sel["n_a"] == 1 and sel["n_b"] == 1
    assert sel["feasible"]
    assert triplet_hash(sel["triplets"]) == sel["triplet_sha256"]


def test_synthetic_b_triplet_is_detected():
    from src.othello_r2_symmetric_target import enumerate_triplets
    from src.othello_r2_symmetric_target_census_audit import (
        isolate_audit_dir,
        max_triplet_packing,
        progressive_funnel,
        synthetic_b_triplet_rows,
    )
    from src.othello_runutil import add_labels
    from src.toy_othello import board_hamming, generate_game, prefixes_from_game

    rows = add_labels(synthetic_b_triplet_rows())
    funnel = progressive_funnel(rows, [0, 1, 2])
    assert funnel["pairs"]["B_pairs"] >= 2
    assert funnel["B"]["n_triplets"] >= 1
    assert funnel["B"]["packing"]["max_packing"] >= 1
    enum = enumerate_triplets(rows, set(), set())
    assert len(enum["found"]["B_similar_board_legal_change"]) >= 1
    empty = max_triplet_packing([])
    assert empty["max_packing"] == 0
    assert empty["method"] == "empty_candidate_certificate"
    with pytest.raises(ValueError):
        isolate_audit_dir("/tmp/results/08r2/symmetric_target")
    squares = generate_game(np.random.default_rng(7), li_compatible=True, max_moves=20)
    recs = prefixes_from_game("fix_0", squares, "synthetic", li_compatible=True)
    assert recs
    ham = int(board_hamming(np.asarray(rows[0]["board"]).reshape(8, 8), np.asarray(rows[1]["board"]).reshape(8, 8)))
    assert 2 <= ham <= 8
