"""Engineering tests for 08R2-FIBER-SPEC. No real checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.othello_r2_fiber_spec import (
    VERDICT_NOT_SUPPORTED,
    VERDICT_SUPPORTED,
    decide_verdict,
    evaluate_spec_completeness,
    isolate_spec_dir,
    match_l2,
    pair_contrasts,
    select_disjoint_pairs,
)


def test_l2_matching_preserves_direction_and_rejects_degenerate():
    v = np.array([3.0, 4.0, 0.0])
    out = match_l2(v, 10.0)
    assert out["ok"] == 1
    assert np.isclose(out["l2"], 10.0)
    assert np.allclose(out["vec"] / np.linalg.norm(out["vec"]), v / np.linalg.norm(v))
    bad = match_l2(np.zeros(3), 10.0)
    assert bad["ok"] == 0
    assert bad["reason"] == "degenerate_direction"


def test_pair_contrasts_aggregate_controls_inside_the_pair():
    ctr = pair_contrasts(1.0, [0.4, 0.5, 0.6], [0.2, 0.3, 0.4])
    assert ctr["delta_specific"] == pytest.approx(0.5)
    assert ctr["delta_random"] == pytest.approx(0.7)
    assert ctr["n_wrong"] == 3
    assert ctr["n_random"] == 3


def test_verdict_requires_full_gate():
    gate = {
        "invariance_ok": 1,
        "l2_ok": 1,
        "mean_delta_specific_positive": 1,
        "ci_lower_gt_0": 1,
        "mean_delta_ge_floor": 1,
        "both_components_correct_sign": 1,
        "beats_random": 1,
        "integrity_ok": 1,
    }
    assert decide_verdict(gate) == VERDICT_SUPPORTED
    gate["mean_delta_ge_floor"] = 0
    assert decide_verdict(gate) == VERDICT_NOT_SUPPORTED


def test_disjoint_selection_does_not_reuse_games():
    found = {
        "A_legal_diff": [
            {"recip_game": "r1", "target_game": "t1", "r2_type": "A_legal_diff"},
            {"recip_game": "r2", "target_game": "t2", "r2_type": "A_legal_diff"},
            {"recip_game": "r3", "target_game": "t1", "r2_type": "A_legal_diff"},
        ],
        "B_similar_board_legal_change": [
            {"recip_game": "r4", "target_game": "t4", "r2_type": "B_similar_board_legal_change"},
            {"recip_game": "r1", "target_game": "t5", "r2_type": "B_similar_board_legal_change"},
        ],
    }
    sel = select_disjoint_pairs(found, n_a=2, n_b=1, seed=0, n_trials=50)
    recip = [p["recip_game"] for p in sel["pairs"]]
    tgt = [p["target_game"] for p in sel["pairs"]]
    assert len(recip) == len(set(recip))
    assert len(tgt) == len(set(tgt))
    assert not (set(recip) & set(tgt))


def test_spec_dir_is_isolated():
    out = isolate_spec_dir("/tmp/results/toy_othello")
    assert out["results_dir"].endswith("08r2/fiber_spec")
    assert "fiber_spec" in Path(out["results_dir"]).parts


def test_completeness_fails_on_empty_dir(tmp_path: Path):
    d = tmp_path / "08r2" / "fiber_spec"
    d.mkdir(parents=True)
    gate = evaluate_spec_completeness(d, real_model=True)
    assert gate["ok"] == 0
    assert any(x.startswith("missing_file:") for x in gate["failures"])
