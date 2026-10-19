"""Regression tests for the restored fixed-JS COST protocol. No real checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from run_08r2_fiber_cost_fixed_js import calibrate_task, preflight_tasks
from src.othello_r2_fiber_cost_fixed import (
    FIXED_JS_PRIMARY,
    FIXED_JS_TOL,
    VERDICT_INCONCLUSIVE,
    VERDICT_NOT_SUPPORTED,
    VERDICT_SUPPORTED,
    bind_literal_js_target,
    build_direction_tasks,
    calibrate_fixed_js,
    decide_fixed_js_verdict,
    evaluate_fixed_js_completeness,
    isolate_fixed_js_dir,
    recompute_js_from_logits,
    reject_donor_reference_rows,
    scaled_displacement,
)


def _manifest(n: int = 48):
    pairs = []
    wrongs = []
    for i in range(n):
        pairs.append(
            {
                "recip": i,
                "donor": i + 200,
                "recip_game": f"r{i}",
                "target_game": f"t{i}",
                "r2_type": "A_legal_diff" if i < 32 else "B_similar_board_legal_change",
            }
        )
        wrongs.append({"ok": 1, "donors": [{"game_id": f"w{i}_{j}"} for j in range(3)]})
    return {"pairs": pairs, "wrong_donors": wrongs, "n": n, "n_a": 32, "n_b": 16}


def test_a_all_336_tasks_have_literal_primary_target():
    tasks = preflight_tasks(_manifest(), FIXED_JS_PRIMARY)
    assert len(tasks) == 336
    assert all(abs(t["target_js"] - 0.01) < 1e-15 for t in tasks)
    assert sum(1 for t in tasks if t["cell"] == "correct_null") == 48


def test_b_correct_donor_uses_the_same_calibrator_as_controls():
    def ev(t):
        return {"js": 0.04 * t}

    correct = calibrate_task({"cell": "correct_null", "target_js": 0.01}, ev)
    wrong = calibrate_task({"cell": "wrong_null_0", "target_js": 0.01}, ev)
    assert correct["status"] == wrong["status"] == "calibrated"
    assert correct["reason"] != "reference"
    assert abs(correct["js"] - 0.01) <= FIXED_JS_TOL
    assert abs(wrong["js"] - 0.01) <= FIXED_JS_TOL
    assert abs(correct["t"] - wrong["t"]) < 1e-9


def test_c_donor_scale1_js_does_not_change_any_target():
    tasks = build_direction_tasks(_manifest(2)["pairs"], _manifest(2)["wrong_donors"])
    fake_donor_js = {("r0", "t0"): 0.28, ("r1", "t1"): 0.91}
    rebound = bind_literal_js_target(tasks, FIXED_JS_PRIMARY)
    for row in rebound:
        row["donor_reference_js"] = fake_donor_js[(row["recip_game"], row["target_game"])]
    again = bind_literal_js_target(rebound, FIXED_JS_PRIMARY)
    assert all(abs(t["target_js"] - 0.01) < 1e-15 for t in again)
    assert all("donor_reference_js" not in t for t in again)


def test_d_calibration_never_returns_reference():
    out = calibrate_fixed_js(lambda t: {"js": min(0.02, 0.05 * t)})
    assert out["reason"] != "reference"
    assert out["status"] != "reference"
    failed = calibrate_fixed_js(lambda t: {"js": 0.001})
    assert failed["reason"] == "js_unreachable"
    assert failed["status"] != "reference"


def test_e_reported_l2_is_norm_of_scaled_displacement():
    d = np.array([3.0, 4.0, 0.0])
    out = scaled_displacement(d, 0.5)
    assert out["l2"] == pytest.approx(2.5)
    assert out["l2"] == pytest.approx(np.linalg.norm(out["vec"]))


def test_f_completeness_rejects_donor_reference_as_fixed_js(tmp_path: Path):
    d = tmp_path / "results" / "08r2" / "fiber_cost_fixed_js"
    d.mkdir(parents=True)
    (d / "fixed_js_calibrations.csv").write_text(
        "cell,target_js,reason,status\ncorrect_null,0.278,reference,reference\n",
        encoding="utf-8",
    )
    chk = reject_donor_reference_rows(
        [{"cell": "correct_null", "target_js": 0.278, "reason": "reference"}],
        FIXED_JS_PRIMARY,
    )
    assert chk["ok"] == 0
    gate = evaluate_fixed_js_completeness(d, real_model=False)
    assert gate["ok"] == 0
    assert any("donor_reference" in x for x in gate["failures"])


def test_g_recomputed_js_matches_saved_helper():
    rng = np.random.default_rng(0)
    a = rng.normal(size=61)
    b = a + 0.1 * rng.normal(size=61)
    from src.othello_r2_fiber import js_divergence, softmax

    saved = js_divergence(softmax(a), softmax(b))
    got = recompute_js_from_logits(a, b)
    assert abs(got - saved) < 1e-12


def test_h_unreachable_direction_stays_a_failed_attempt():
    attempts = {"js_fail": 0, "valid": 0, "kept_failures": []}
    out = calibrate_task({"cell": "random_null_0", "target_js": 0.01}, lambda t: {"js": 0.0001})
    assert out["ok"] == 0
    assert out["reason"] == "js_unreachable"
    attempts["js_fail"] += 1
    attempts["kept_failures"].append(out)
    assert attempts["js_fail"] == 1
    assert attempts["valid"] == 0
    assert len(attempts["kept_failures"]) == 1


def test_isolate_dir_does_not_write_into_old_cost():
    out = isolate_fixed_js_dir("/tmp/toy_othello")
    assert out["results_dir"].endswith("results/08r2/fiber_cost_fixed_js")
    assert "fiber_spec" not in Path(out["results_dir"]).parts
    diverted = isolate_fixed_js_dir("/tmp/toy_othello/08r2/fiber_cost")
    assert Path(diverted["results_dir"]).name == "fiber_cost_fixed_js"
    assert Path(diverted["results_dir"]).parts[-3:] == ("results", "08r2", "fiber_cost_fixed_js")


def test_gate_incomplete_coverage_is_inconclusive():
    gate = {
        "invariance_ok": 1,
        "js_match_ok": 0,
        "coverage_ok": 0,
        "mean_delta_specific_positive": 1,
        "ci_lower_gt_0": 1,
        "mean_delta_ge_floor": 1,
        "both_components_correct_sign": 1,
        "beats_random": 1,
        "integrity_ok": 1,
    }
    assert decide_fixed_js_verdict(gate) == VERDICT_INCONCLUSIVE
    gate["js_match_ok"] = 1
    gate["coverage_ok"] = 1
    assert decide_fixed_js_verdict(gate) == VERDICT_SUPPORTED
    gate["mean_delta_ge_floor"] = 0
    assert decide_fixed_js_verdict(gate) == VERDICT_NOT_SUPPORTED
