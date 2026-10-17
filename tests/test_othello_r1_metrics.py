from __future__ import annotations

import numpy as np

from src.othello_r1 import (
    intervention_probability_metrics,
    mean_prob_per_move,
    token_of,
    total_legal_mass,
)
from src.othello_runutil import mean_prob_on
from src.toy_othello import CENTER_SQUARES, square_to_token


def _hand_probs() -> np.ndarray:
    p = np.zeros(61, dtype=np.float64)
    p[square_to_token(0)] = 0.50
    p[square_to_token(1)] = 0.20
    p[square_to_token(2)] = 0.10
    p[0] = 0.05
    rest = 1.0 - p.sum()
    p[square_to_token(3)] = rest
    assert np.isclose(p.sum(), 1.0)
    return p


def test_square_to_token_skips_center_and_is_invertible_for_playable():
    for sq in range(64):
        if sq in CENTER_SQUARES:
            assert token_of(sq) is None
            continue
        tok = token_of(sq)
        assert tok is not None
        assert 1 <= tok <= 60
        assert square_to_token(sq) == tok


def test_total_mass_not_equal_to_mean():
    p = _hand_probs()
    legal = [0, 1]
    mass = total_legal_mass(p, legal)
    mean = mean_prob_per_move(p, legal)
    assert np.isclose(mass, 0.70)
    assert np.isclose(mean, 0.35)
    assert np.isclose(mean_prob_on(p, legal, square_to_token), 0.35)


def test_empty_legal_set_is_undefined_with_zero_cardinality():
    p = _hand_probs()
    q = p.copy()
    row = intervention_probability_metrics(p, q, [], [])
    assert row["n_became_legal"] == 0
    assert row["n_became_illegal"] == 0
    assert np.isnan(row["became_legal_d_mass"])
    assert np.isnan(row["became_illegal_d_mass"])
    assert np.isnan(total_legal_mass(p, []))
    assert np.isnan(mean_prob_per_move(p, []))


def test_single_move_set():
    p = _hand_probs()
    assert np.isclose(total_legal_mass(p, [0]), 0.50)
    assert np.isclose(mean_prob_per_move(p, [0]), 0.50)


def test_multiple_move_set():
    p = _hand_probs()
    assert np.isclose(total_legal_mass(p, [0, 1, 2]), 0.80)
    assert np.isclose(mean_prob_per_move(p, [0, 1, 2]), 0.80 / 3.0)


def test_became_legal_only_uses_recipient_illegal_target_legal():
    before = _hand_probs()
    after = before.copy()
    after[square_to_token(2)] += 0.08
    after[square_to_token(0)] -= 0.08
    rec_legal = [0, 1]
    target_legal = [1, 2]
    row = intervention_probability_metrics(before, after, rec_legal, target_legal)
    assert row["n_became_legal"] == 1
    assert row["n_became_illegal"] == 1
    assert row["n_stayed_legal"] == 1
    assert np.isclose(row["became_legal_d_mass"], 0.08)
    assert np.isclose(row["became_illegal_d_mass"], -0.08)
    assert np.isclose(row["stayed_legal_d_mass"], 0.0)


def test_unchanged_illegal_cardinality_is_playable_minus_union():
    p = _hand_probs()
    row = intervention_probability_metrics(p, p, [0], [0, 1])
    assert row["n_stayed_illegal"] == 60 - 2
    assert np.isclose(row["became_legal_d_mass"], 0.0)
    assert np.isclose(row["donor_legal_mass_lift"], 0.0)


def test_historical_mean_metric_is_kept_beside_mass():
    p = _hand_probs()
    q = p.copy()
    q[square_to_token(2)] += 0.08
    q[square_to_token(0)] -= 0.08
    row = intervention_probability_metrics(p, q, [0, 1], [1, 2])
    assert "old_mean_donor_legal_lift" in row
    assert "donor_legal_mass_lift" in row
    assert not np.isclose(row["old_mean_donor_legal_lift"], row["donor_legal_mass_lift"])
    assert np.isclose(row["old_mean_donor_legal_before"], mean_prob_per_move(p, [1, 2]))


def test_probability_conservation_of_hand_shift():
    before = _hand_probs()
    after = before.copy()
    after[square_to_token(2)] += 0.08
    after[square_to_token(0)] -= 0.08
    assert np.isclose(before.sum(), 1.0)
    assert np.isclose(after.sum(), 1.0)
    row = intervention_probability_metrics(before, after, [0, 1], [1, 2])
    parts = (
        row["became_legal_d_mass"]
        + row["became_illegal_d_mass"]
        + row["stayed_legal_d_mass"]
        + row["stayed_illegal_d_mass"]
    )
    assert np.isclose(parts, 0.0)
