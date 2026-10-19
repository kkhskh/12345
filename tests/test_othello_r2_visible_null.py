"""Engineering tests for 08R2-VISIBLE-NULL-FACTORIAL. No real checkpoint."""

from __future__ import annotations

import numpy as np
import pytest

from src.othello_r2_visible_null import (
    VERDICT_NOT,
    VERDICT_SUPPORTED,
    decide_vnf_verdict,
    factorial_cells,
    interaction_ok,
    interaction_scores,
    isolate_vnf_dir,
    project_check,
    unit,
)


def test_seeds_and_isolate():
    out = isolate_vnf_dir("/tmp/toy_othello")
    assert out["results_dir"].endswith("results/08r2/visible_null_factorial")
    with pytest.raises(ValueError):
        isolate_vnf_dir("/tmp/results/08r2/symmetric_target_fresh")


def test_orthogonal_budget_and_interaction():
    rng = np.random.default_rng(0)
    h = rng.normal(size=8)
    r = np.zeros(8)
    r[:4] = rng.normal(size=4)
    n = np.zeros(8)
    n[4:] = rng.normal(size=4)
    P_R = np.diag([1, 1, 1, 1, 0, 0, 0, 0]).astype(np.float64)
    P_N = np.eye(8) - P_R
    assert project_check(r, P_R, P_N, "row")["ok"]
    assert project_check(n, P_R, P_N, "null")["ok"]
    built = factorial_cells(h, r, r, n, n, 3.5, 3.5)
    assert built["ok"]
    assert built["l2"]["TT"] == pytest.approx(np.sqrt(2 * 3.5 ** 2), abs=1e-6)
    assert unit(np.zeros(3))["ok"] == 0
    y_t = {"TT": 0.2, "TW": 0.1, "WT": 0.11, "WW": 0.10}
    y_w = {"WW": 0.18, "WT": 0.09, "TW": 0.10, "TT": 0.08}
    xo = interaction_scores(y_t, y_w)
    assert xo["I_T"] == pytest.approx(0.09)
    assert xo["I_W"] == pytest.approx(0.07)


def test_gate():
    ok = {"point": 0.02, "ci95_low": 0.01}
    gate = {k: 1 for k in ("split_ok", "integrity_ok", "coverage_ok", "projection_ok", "l2_ok")}
    gate["interaction_ok"] = interaction_ok(ok, ok)
    gate["control_reproduces"] = 0
    assert decide_vnf_verdict(gate) == VERDICT_SUPPORTED
    gate["control_reproduces"] = 1
    assert decide_vnf_verdict(gate) == VERDICT_NOT
