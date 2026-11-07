"""Intervention metric and baseline-label tests. No frozen-model claims."""

from __future__ import annotations

import numpy as np

from src.othello_r2 import causal_scores, outcome_bucket, pca_patch, random_same_norm_edit
from src.toy_othello import square_to_token


def _shift():
    p = np.zeros(61, dtype=np.float64)
    p[square_to_token(0)] = 0.40
    p[square_to_token(1)] = 0.20
    p[square_to_token(2)] = 0.10
    p[0] = 0.05
    p[square_to_token(3)] = 1.0 - p.sum()
    q = p.copy()
    q[square_to_token(2)] += 0.12
    q[square_to_token(0)] -= 0.12
    return p, q


def test_gain_and_loss_are_reported_separately():
    p, q = _shift()
    row = causal_scores(p, q, [0, 1], [1, 2])
    assert np.isclose(row["dP_gain"], 0.12)
    assert np.isclose(row["dP_loss"], -0.12)
    assert np.isclose(row["S_legal"], 0.24)
    assert "became_legal_d_mass" in row
    assert "became_illegal_d_mass" in row


def test_outcome_buckets_separate_probe_fooling():
    assert outcome_bucket(1.0, 0.2, -0.2) == "A_decode_and_behavior"
    assert outcome_bucket(1.0, 0.0, 0.0) == "B_probe_fooling"
    assert outcome_bucket(0.0, 0.2, -0.1) == "C_behavior_without_decode"
    assert outcome_bucket(0.0, 0.0, 0.0) == "D_neither"


def test_random_edit_matches_requested_norm():
    h = np.zeros(16)
    out = random_same_norm_edit(h, 5.0, seed=7)
    assert np.isclose(np.linalg.norm(out - h), 5.0)


def test_pca_patch_uses_donor_and_complement_is_orthogonal():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(8, 3)))[0]
    rec = rng.normal(size=8)
    don = rng.normal(size=8)
    state = pca_patch(rec, don, basis, complement=False)
    comp = pca_patch(rec, don, basis, complement=True)
    d_state = state["edited"] - rec
    d_comp = comp["edited"] - rec
    assert state["valid_edit"] == 1
    assert np.allclose(d_state, ((don - rec) @ basis) @ basis.T)
    assert np.abs(d_state @ d_comp) < 1e-8
    assert np.isclose(state["delta_norm"], np.linalg.norm(d_state))
