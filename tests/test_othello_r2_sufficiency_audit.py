"""Synthetic matching tests for 08R2-S. No real-model files."""

from __future__ import annotations

import pandas as pd

from run_08r2_decoded_board_sufficiency_audit import (
    decide_verdict,
    editor_a_budget_identity,
    match_editor_a_to_donor,
    missing_representation_fields,
    slice_stats,
)


def _row(method, r2, recip_game, target, budget, gain, loss, exact, recip=0, donor=1, **extra):
    rec = {
        "method": method,
        "r2_type": r2,
        "recip_game": recip_game,
        "target_game": target,
        "recip": recip,
        "donor": donor,
        "budget": budget,
        "valid_edit": 1,
        "dP_gain": gain,
        "dP_loss": loss,
        "S_legal": gain - loss,
        "board_exact": exact,
        "changed_acc": 1.0 if exact else 0.2,
        "kl": extra.get("kl", 1.0),
        "delta_norm": extra.get("l2", budget),
    }
    return rec


def test_matches_on_recip_and_target_and_excludes_d():
    rows = [
        _row("editor_A_fullspace", "A_legal_diff", "r1", "t1", 16.0, 0.02, -0.05, 1.0, recip=1, donor=2, l2=9.0),
        _row("editor_A_fullspace", "A_legal_diff", "r1", "t1", 32.0, 0.02, -0.05, 1.0, recip=1, donor=2, l2=9.0),
        _row("full_donor_patch", "A_legal_diff", "r1", "t1", 25.1, 0.80, -0.70, 1.0, recip=1, donor=2, l2=25.1),
        _row("editor_A_fullspace", "D_wrong_target", "r1", "w1", 16.0, 0.02, -0.05, 1.0, recip=1, donor=9, l2=9.0),
        _row("full_donor_patch", "D_wrong_target", "r1", "w1", 20.0, 0.50, -0.40, 1.0, recip=1, donor=9, l2=20.0),
        _row("editor_A_fullspace", "B_similar_board_legal_change", "r2", "t2", 16.0, 0.002, -0.09, 1.0, recip=3, donor=4, l2=7.7),
        _row("full_donor_patch", "B_similar_board_legal_change", "r2", "t2", 18.0, 0.50, -0.45, 1.0, recip=3, donor=4, l2=18.0),
    ]
    matched, flags = match_editor_a_to_donor(pd.DataFrame(rows), 16.0)
    assert flags["matched"] == 2
    assert set(matched["r2_type"]) == {"A_legal_diff", "B_similar_board_legal_change"}
    assert bool(matched["sufficiency_counterexample"].all())
    assert flags["editor_A_unmatched"] == 0


def test_does_not_group_donor_by_l2():
    rows = [
        _row("editor_A_fullspace", "A_legal_diff", "r1", "t1", 16.0, 0.02, -0.05, 1.0, recip=1, donor=2, l2=9.0),
        _row("full_donor_patch", "A_legal_diff", "r1", "t1", 25.111, 0.80, -0.70, 1.0, recip=1, donor=2, l2=25.111),
        _row("editor_A_fullspace", "A_legal_diff", "r2", "t1", 16.0, 0.01, -0.06, 1.0, recip=5, donor=2, l2=9.2),
        _row("full_donor_patch", "A_legal_diff", "r2", "t1", 25.222, 0.70, -0.60, 1.0, recip=5, donor=2, l2=25.222),
    ]
    matched, flags = match_editor_a_to_donor(pd.DataFrame(rows), 16.0)
    assert flags["matched"] == 2
    stats = slice_stats(matched, "A_legal_diff", 200, 0)
    assert stats["n_target_games"] == 1
    assert stats["n_recipient_games"] == 2
    assert stats["n_counterexamples"] == 2


def test_missing_logits_are_not_identifiable():
    ident = missing_representation_fields(["method", "dP_gain", "board_exact", "kl"])
    assert ident["decoded_board_labels"] == "IDENTIFIABLE"
    assert ident["legal_set_probability_metrics"] == "IDENTIFIABLE"
    assert ident["decoder_probabilities_or_margins"] == "NOT_IDENTIFIABLE_FROM_SAVED_DATA"
    assert ident["full_output_logits_or_probs"] == "NOT_IDENTIFIABLE_FROM_SAVED_DATA"


def test_verdict_requires_exact_board_counterexamples():
    assert decide_verdict(0, 0, 0) == "INCONCLUSIVE_FROM_SAVED_DATA"
    assert decide_verdict(8, 0, 0) == "INCONCLUSIVE_FROM_SAVED_DATA"
    assert decide_verdict(8, 8, 0) == "NOT_SUPPORTED"
    assert decide_verdict(8, 8, 3) == "DECODED_BOARD_INSUFFICIENCY_SUPPORTED"


def test_budget_16_and_32_are_not_independent_if_identical():
    rows = [
        _row("editor_A_fullspace", "A_legal_diff", "r1", "t1", 16.0, 0.02, -0.05, 1.0, recip=1, donor=2, l2=9.57),
        _row("editor_A_fullspace", "A_legal_diff", "r1", "t1", 32.0, 0.02, -0.05, 1.0, recip=1, donor=2, l2=9.57),
    ]
    ident = editor_a_budget_identity(pd.DataFrame(rows))
    assert ident["comparable"] == 1
    assert ident["identical_operating_point"] is True
