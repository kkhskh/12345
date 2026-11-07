"""Engineering tests for 08R2-SYMMETRIC-TARGET-FRESH. No real checkpoint."""

from __future__ import annotations

import pytest

from src.othello_r2_symmetric_target_census_audit import synthetic_b_triplet_rows
from src.othello_r2_symmetric_target_fresh import (
    COHORT_SEED,
    PILOT_COHORT_SEED,
    PILOT_PREFIX_SEED,
    PREFIX_SEED,
    TRIPLET_SEED,
    isolate_fresh_dir,
    joint_ab_packing,
    sequence_hash,
    witness_ok,
)


def test_fresh_seeds_are_not_the_pilot():
    assert COHORT_SEED == 9182
    assert PREFIX_SEED == 9282
    assert TRIPLET_SEED == 9382
    assert {COHORT_SEED, PREFIX_SEED, TRIPLET_SEED}.isdisjoint({PILOT_COHORT_SEED, PILOT_PREFIX_SEED, 7182, 4182, 2182, 0, 1, 2})


def test_isolate_and_sequence_hash():
    out = isolate_fresh_dir("/tmp/toy_othello")
    assert out["results_dir"].endswith("results/08r2/symmetric_target_fresh")
    with pytest.raises(ValueError):
        isolate_fresh_dir("/tmp/results/08r2/symmetric_target")
    with pytest.raises(ValueError):
        isolate_fresh_dir("/tmp/results/08r2/symmetric_target_census_audit")
    assert sequence_hash([1, 2, 3]) == sequence_hash([1, 2, 3])
    assert sequence_hash([1, 2, 3]) != sequence_hash([1, 2, 4])


def test_joint_witness_requires_96_games():
    trips = []
    for i in range(16):
        trips.append({"recip_game": f"rA{i}", "target_t_game": f"tA{i}", "target_w_game": f"wA{i}", "r2_type": "A_legal_diff"})
        trips.append({"recip_game": f"rB{i}", "target_t_game": f"tB{i}", "target_w_game": f"wB{i}", "r2_type": "B_similar_board_legal_change"})
    sel = {"triplets": trips, "n_a": 16, "n_b": 16}
    wit = witness_ok(sel)
    assert wit["n_union"] == 96
    assert wit["ok"] == 1
    trips[0]["target_t_game"] = trips[1]["recip_game"]
    assert witness_ok({"triplets": trips, "n_a": 16, "n_b": 16})["ok"] == 0


def test_joint_packing_on_materialized_pools():
    found = {
        "A_legal_diff": [
            {"recip": 0, "target_t": 1, "target_w": 2, "r2_type": "A_legal_diff", "recip_game": "r0", "target_t_game": "t0", "target_w_game": "w0", "hamming_t": 12, "hamming_w": 14},
        ],
        "B_similar_board_legal_change": [
            {"recip": 3, "target_t": 4, "target_w": 5, "r2_type": "B_similar_board_legal_change", "recip_game": "r1", "target_t_game": "t1", "target_w_game": "w1", "hamming_t": 4, "hamming_w": 6},
        ],
    }
    sel = joint_ab_packing(found, 1, 1, 9382, n_trials=20)
    assert sel["feasible"]
    assert sel["n_a"] == 1 and sel["n_b"] == 1
    empty = joint_ab_packing({"A_legal_diff": found["A_legal_diff"], "B_similar_board_legal_change": []}, 1, 1, 9382, n_trials=5)
    assert not empty["feasible"]
    assert empty["infeasible_certificate"] == "b_upper_lt_16"
    assert synthetic_b_triplet_rows()
