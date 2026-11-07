"""08R2 metric definitions. Does not use model outputs from Colab."""

from __future__ import annotations

import numpy as np

from src.othello_r2 import decoder_state_effects, LinearBoardDecoder
from tests.test_othello_r2_probe_parity import _pack


def test_changed_and_unchanged_square_groups():
    pack = _pack()
    dec = LinearBoardDecoder(pack)
    h = np.zeros(dec.d_model)
    rec = np.zeros(64, dtype=np.int64)
    tgt = rec.copy()
    tgt[:5] = 1
    pred = dec.predict_np(h).reshape(-1)
    out = decoder_state_effects(dec, h, rec, tgt)
    change = tgt != rec
    keep = ~change
    assert out["n_changed"] == 5
    assert out["n_unchanged"] == 59
    assert np.isclose(out["changed_acc"], float((pred[change] == tgt[change]).mean()))
    assert np.isclose(out["unchanged_acc"], float((pred[keep] == tgt[keep]).mean()))
    assert out["board_exact"] in {0.0, 1.0}


def test_s_legal_cannot_hide_a_one_sided_failure():
    from src.othello_r2 import causal_scores
    from src.toy_othello import square_to_token

    before = np.zeros(61)
    before[square_to_token(0)] = 0.5
    before[square_to_token(1)] = 0.5
    after = before.copy()
    after[square_to_token(2)] = 0.2
    after[square_to_token(1)] = 0.3
    row = causal_scores(before, after, [0, 1], [1, 2])
    assert row["dP_gain"] > 0
    assert abs(row["dP_loss"]) < 1e-12
    assert row["joint_metric_eligible"] == 1
    assert row["S_legal"] == row["S_legal"]
    assert not (row["dP_gain"] > 0 and row["dP_loss"] < 0)


def test_empty_became_illegal_does_not_zero_fill_s_legal():
    from src.othello_r2 import causal_scores
    from src.toy_othello import square_to_token

    before = np.zeros(61)
    before[square_to_token(0)] = 1.0
    after = before.copy()
    row = causal_scores(before, after, [0], [0])
    assert row["n_became_legal"] == 0
    assert row["n_became_illegal"] == 0
    assert row["dP_gain_defined"] == 0
    assert row["dP_loss_defined"] == 0
    assert row["joint_metric_eligible"] == 0
    assert row["S_legal"] != row["S_legal"]
