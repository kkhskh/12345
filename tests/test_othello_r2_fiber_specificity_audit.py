"""Synthetic tests for the read-only FIBER specificity audit. No checkpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from run_08r2_fiber_specificity_audit import (
    check_simplex,
    ci_excludes_zero,
    decide_claims,
    factorial_integrity,
    parse_logit_key,
    recompute_from_logits,
    softmax,
    summarize_anchor,
)
from src.othello_r2_fiber import factorial_effects, js_divergence


def test_parse_logit_key_splits_from_the_right():
    assert parse_logit_key("val_00047__val_00012__H01") == ("val_00047", "val_00012", "H01")


def test_simplex_and_independent_js():
    logits = np.array([1.0, 0.0, -1.0] + [0.0] * 58)
    p = softmax(logits)
    chk = check_simplex(p)
    assert chk["finite"] and chk["nonnegative"] and chk["normalized"]
    assert js_divergence(p, p) < 1e-12


def test_identity_holds_on_saved_y():
    fx = factorial_effects(0.1, 0.4, 0.5, 0.9)
    assert abs((fx["Y11"] - fx["Y00"]) - (fx["E_R"] + fx["E_N"] + fx["I"])) < 1e-12


def test_recompute_js_from_raw_logits():
    rng = np.random.default_rng(0)
    logits = {cell: rng.normal(size=61) for cell in ("H00", "H10", "H01", "H11")}
    matched = pd.DataFrame(
        [
            {
                "recip_game": "r1",
                "target_game": "t1",
                "r2_type": "A_legal_diff",
                "Y00": 0.0,
                "Y10": 0.2,
                "Y01": 0.3,
                "Y11": 0.4,
                "E_R": 0.2,
                "E_N": 0.3,
                "I": -0.1,
                "js_00_01": js_divergence(softmax(logits["H00"]), softmax(logits["H01"])),
                "js_10_11": js_divergence(softmax(logits["H10"]), softmax(logits["H11"])),
            }
        ]
    )
    logit_map = {("r1", "t1", cell): logits[cell] for cell in logits}
    out = recompute_from_logits(logit_map, matched)
    assert out["raw_logits_complete"].iloc[0] == 1
    assert out["identity_err"].iloc[0] < 1e-12
    assert abs(out["js_00_01_recomputed"].iloc[0] - matched["js_00_01"].iloc[0]) < 1e-12


def test_factorial_requires_all_four_cells():
    fact = pd.DataFrame(
        [
            {"recip_game": "r1", "target_game": "t1", "cell": c, "Y": 0.0, "became_legal_mass": 0.1, "became_illegal_mass": 0.1}
            for c in ("H00", "H10", "H01")
        ]
    )
    info = factorial_integrity(fact)
    assert info["n_pairs_with_four_cells"] == 0
    assert info["n_pairs_missing_cells"] == 1


def test_anchor_summaries_are_not_pooled_across_eps():
    rows = []
    for eps, dy in ((1.0, 0.01), (4.0, 0.40)):
        for sign in (1.0, -1.0):
            rows.append(
                {
                    "family": "donor_null_direction",
                    "eps": eps,
                    "sign": sign,
                    "recip_game": "r1",
                    "target_game": "t1",
                    "r2_type": "A_legal_diff",
                    "dY_vs_donor": sign * dy,
                    "js_vs_donor": dy,
                    "invariance_ok": 1,
                    "invariance_err": 1e-8,
                    "l2_to_donor": eps,
                }
            )
            for draw in range(3):
                rows.append(
                    {
                        "family": "random_nullspace",
                        "eps": eps,
                        "sign": sign,
                        "draw": draw,
                        "recip_game": "r1",
                        "target_game": "t1",
                        "r2_type": "A_legal_diff",
                        "dY_vs_donor": 0.002 * sign,
                        "js_vs_donor": 0.002,
                        "invariance_ok": 1,
                        "invariance_err": 1e-8,
                        "l2_to_donor": eps,
                    }
                )
    summary = summarize_anchor(pd.DataFrame(rows), n_boot=50, seed=0)
    assert "1.0" in summary["by_eps"] and "4.0" in summary["by_eps"]
    assert summary["by_eps"]["1.0"]["donor"]["mean_abs_dY"] < 0.05
    assert summary["by_eps"]["4.0"]["donor"]["mean_abs_dY"] > 0.2
    assert summary["by_eps"]["1.0"]["paired_vs_random"]["n_random_draws_mean"] == 3


def test_claim_c_is_not_inferred_from_a():
    claims = decide_claims(
        {"n_demo_saved": 32, "identity_max_err": 0.0, "n_pairs_four_cells": 32, "probe_invariance_ok": 1},
        {"present": 0},
        {"present": 0},
        {"behavior_floor": 0.05, "js_meaningful": 0.001},
    )
    assert claims["A_continuous_probe_insufficiency"] == "SUPPORTED"
    assert claims["C_target_specific_missing_state"] == "INCONCLUSIVE"
    assert claims["C_not_inferred_from_A_or_B"] == 1


def test_ci_excludes_zero():
    assert ci_excludes_zero({"ci95_low": 0.1, "ci95_high": 0.2}) is True
    assert ci_excludes_zero({"ci95_low": -0.1, "ci95_high": 0.2}) is False
