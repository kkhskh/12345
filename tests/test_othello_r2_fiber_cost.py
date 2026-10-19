"""Engineering tests for 08R2-FIBER-COST. No real checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.othello_r2_fiber_cost import (
    VERDICT_ENGINEERING,
    VERDICT_NOT_SUPPORTED,
    VERDICT_SUPPORTED,
    decide_cost_verdict,
    evaluate_cost_completeness,
    isolate_cost_dir,
    js_match_ok,
    load_frozen_spec_manifest,
    match_output_js,
    reconstruct_frozen_pairs,
    resolve_spec_dir,
    unit_direction,
)


def test_js_match_ok_uses_absolute_or_relative_tolerance():
    assert js_match_ok(0.278, 0.279)
    assert js_match_ok(0.265, 0.279) is False
    assert js_match_ok(0.20, 0.21, atol=0.0, rtol=0.05)
    assert js_match_ok(float("nan"), 0.2) is False


def test_js_bisection_recovers_monotone_scale():
    calls = []

    def ev(alpha):
        calls.append(alpha)
        return {"js": 0.02 * alpha, "payload": alpha}

    out = match_output_js(ev, target_js=0.28, alpha_hi=40.0, atol=0.01, rtol=0.0, n_iter=10)
    assert out["ok"] == 1
    assert abs(out["js"] - 0.28) <= 0.01
    assert abs(out["alpha"] - 14.0) <= 0.6
    assert out["payload"] == out["alpha"]
    assert len(calls) <= 12


def test_js_bisection_fails_closed_when_unreachable():
    out = match_output_js(lambda a: {"js": 0.01 * a}, target_js=0.5, alpha_hi=10.0)
    assert out["ok"] == 0
    assert out["reason"] == "js_unreachable"


def test_unit_direction_rejects_degenerate():
    v = unit_direction(np.array([3.0, 4.0, 0.0]))
    assert v is not None
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert unit_direction(np.zeros(4)) is None


def test_verdict_requires_js_match_not_l2():
    gate = {
        "invariance_ok": 1,
        "js_match_ok": 1,
        "mean_delta_specific_positive": 1,
        "ci_lower_gt_0": 1,
        "mean_delta_ge_floor": 1,
        "both_components_correct_sign": 1,
        "beats_random": 1,
        "integrity_ok": 1,
    }
    assert decide_cost_verdict(gate) == VERDICT_SUPPORTED
    gate["js_match_ok"] = 0
    assert decide_cost_verdict(gate) == VERDICT_NOT_SUPPORTED
    gate["js_match_ok"] = 1
    gate["integrity_ok"] = 0
    assert decide_cost_verdict(gate) == VERDICT_ENGINEERING


def test_cost_dir_is_isolated():
    out = isolate_cost_dir("/tmp/results/toy_othello")
    assert out["results_dir"].endswith("08r2/fiber_cost")
    assert "fiber_spec" not in Path(out["results_dir"]).parts


def test_spec_dir_discovery_uses_existing_manifest(tmp_path: Path):
    spec = tmp_path / "08r2" / "fiber_spec"
    spec.mkdir(parents=True)
    (spec / "fiber_spec_pair_manifest.json").write_text("{}", encoding="utf-8")
    found = resolve_spec_dir(None, tmp_path)
    assert Path(found["spec_dir"]) == spec
    assert found["source"] == "discovered"


def test_missing_spec_dir_is_not_fatal(tmp_path: Path):
    found = resolve_spec_dir(tmp_path / "absent", tmp_path)
    assert found["spec_dir"] is None
    assert found["source"] == "missing"


def test_reconstruct_rejects_hash_mismatch():
    with pytest.raises(ValueError, match="reconstructed pair hash"):
        reconstruct_frozen_pairs([], set(), "deadbeef", exclude_fiber=False)


def test_frozen_manifest_rejects_wrong_hash(tmp_path: Path):
    spec = tmp_path / "fiber_spec"
    spec.mkdir()
    (spec / "fiber_spec_pair_manifest.json").write_text(
        '{"n":48,"n_a":32,"n_b":16,"pair_manifest_sha256":"aaa","pairs":[],"wrong_donors":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_frozen_spec_manifest(spec, "bbb")


def test_completeness_fails_on_empty_dir(tmp_path: Path):
    d = tmp_path / "08r2" / "fiber_cost"
    d.mkdir(parents=True)
    gate = evaluate_cost_completeness(d, real_model=True)
    assert gate["ok"] == 0
    assert any(x.startswith("missing_file:") for x in gate["failures"])
