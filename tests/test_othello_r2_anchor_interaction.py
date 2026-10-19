"""Engineering tests for the read-only anchor-interaction audit."""

from __future__ import annotations

import numpy as np
import pytest

from src.othello_r2_anchor_interaction import (
    PUBLISHED,
    contrast_from_ys,
    isolate_post_dir,
    mean_log_label_prob,
    published_match,
    switching_cell,
)


def test_isolate_requires_fresh_parent():
    out = isolate_post_dir("/tmp/results/08r2/symmetric_target_fresh")
    assert out["results_dir"].endswith("symmetric_target_fresh/postvalidation")
    with pytest.raises(ValueError):
        isolate_post_dir("/tmp/results/08r2/null_state_transfer")


def test_switching_cells_and_published_match():
    assert switching_cell(0.02, 0.01) == "both"
    assert switching_cell(0.02, -0.01) == "t_only"
    assert switching_cell(-0.02, 0.01) == "w_only"
    assert switching_cell(-0.02, -0.01) == "neither"
    xo = contrast_from_ys(0.1, 0.05, 0.08, 0.07)
    assert xo["A_T"] == pytest.approx(0.05)
    assert xo["A_W"] == pytest.approx(0.01)
    assert xo["both_positive"] == 1
    got = {**PUBLISHED}
    assert published_match(got)["ok"] == 1
    bad = {**PUBLISHED, "A_T": 0.0}
    assert published_match(bad)["ok"] == 0


def test_visible_logprob_clip():
    p = np.full((64, 3), 1e-20)
    p[:, 0] = 1.0
    lab = np.zeros(64, dtype=int)
    assert mean_log_label_prob(p, lab, 1e-12) == pytest.approx(0.0)
    lab2 = np.ones(64, dtype=int)
    assert mean_log_label_prob(p, lab2, 1e-12) == pytest.approx(np.log(1e-12))
