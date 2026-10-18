"""Predicate and pair-count helpers for the read-only 08R2 audit. No real-model data."""

from __future__ import annotations

import pandas as pd

from run_08r2_postvalidation_audit import (
    PREREG,
    SIGN_ONLY,
    _prereg,
    _sign_only,
    accounting_table,
    pair_type_numeric,
)


def test_sign_only_is_not_preregistered_floor():
    assert "> 0" in SIGN_ONLY and ">= 0.05" in PREREG
    df = pd.DataFrame(
        [
            {
                "method": "editor_A_fullspace",
                "r2_type": "A_legal_diff",
                "budget": 16.0,
                "valid_edit": 1,
                "changed_acc": 1.0,
                "unchanged_acc": 1.0,
                "joint_metric_eligible": 1,
                "dP_gain": 0.02,
                "dP_loss": -0.06,
                "S_legal": 0.08,
                "recip_game": "val_1",
                "target_game": "val_2",
                "board_exact": 1.0,
                "delta_norm": 9.0,
                "kl": 0.3,
            }
        ]
    )
    assert bool(_sign_only(df).iloc[0]) is True
    assert bool(_prereg(df).iloc[0]) is False


def test_accounting_splits_d_and_explains_192():
    rows = []
    for r2, n, recip_prefix in (("A_legal_diff", 24, "a"), ("B_similar_board_legal_change", 8, "b"), ("D_wrong_target", 32, "d")):
        for i in range(n):
            for budget in (8.0, 16.0, 32.0):
                rows.append(
                    {
                        "method": "editor_A_fullspace",
                        "r2_type": r2,
                        "budget": budget,
                        "valid_edit": 1,
                        "changed_acc": 1.0,
                        "unchanged_acc": 1.0,
                        "joint_metric_eligible": 1,
                        "dP_gain": 0.02,
                        "dP_loss": -0.06,
                        "S_legal": 0.08,
                        "recip_game": f"{recip_prefix}_{i}",
                        "target_game": f"t_{i}",
                        "board_exact": 1.0,
                        "delta_norm": 9.0,
                        "kl": 0.3,
                    }
                )
    acc = accounting_table(pd.DataFrame(rows))
    all_d = acc[(acc["method"] == "editor_A_fullspace") & (acc["slice"] == "all_including_D") & (acc["budget"] == "all")]
    assert int(all_d.iloc[0]["attempted"]) == 192
    prim = acc[(acc["slice"] == "primary_AB") & (acc["budget"] == "all")]
    assert int(prim.iloc[0]["attempted"]) == 96
    assert "64 pairs" in str(all_d.iloc[0]["attempts_construction"])


def test_pair_type_query_does_not_mean_game_ids():
    pairs = pd.DataFrame(
        {
            "r2_type": ["A_legal_diff", "A_legal_diff", "B_similar_board_legal_change"],
            "recip_game": ["val_00047", "val_00048", "val_00001"],
            "target_game": ["val_t1", "val_t1", "val_t2"],
            "hamming": [20, 30, 6],
            "n_became_legal": [4, 5, 1],
            "n_became_illegal": [3, 2, 1],
        }
    )
    out = pair_type_numeric(pairs)
    a = out[out["r2_type"] == "A_legal_diff"].iloc[0]
    assert a["n_pairs"] == 2
    assert a["n_unique_recip_games"] == 2
    assert a["n_unique_target_games"] == 1
    assert abs(a["hamming_mean"] - 25) < 1e-9
