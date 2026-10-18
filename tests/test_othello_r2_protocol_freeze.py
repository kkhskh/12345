"""Frozen A/B/D protocol, failure accounting, and KL gate. No real-model claims."""

from __future__ import annotations

from pathlib import Path

from src.othello_r2 import (
    PROTOCOL_ABD,
    abd_pair_quotas,
    editor_attempt_accounting,
    isolate_r2_results_dir,
    kl_norm_comparisons,
    random_kl_threshold,
    smoke_ab_quotas,
)


def test_abd_quotas_and_smoke_keep_both_categories():
    assert PROTOCOL_ABD == {"n_a": 24, "n_b": 8, "include_c": False, "name": "ABD_v1"}
    assert abd_pair_quotas(32, smoke=False) == (24, 8)
    assert smoke_ab_quotas(4) == (2, 2)
    assert abd_pair_quotas(4, smoke=True) == (2, 2)
    n_a, n_b = smoke_ab_quotas(4)
    assert n_a >= 1 and n_b >= 1


def test_smoke_results_are_isolated():
    out = isolate_r2_results_dir("artifacts/results/toy_othello", smoke=True)
    assert out["results_dir"].endswith("08r2/smoke")
    val = isolate_r2_results_dir("artifacts/results/toy_othello", smoke=False, validation=True)
    assert val["results_dir"].endswith("08r2/validation")
    assert "smoke" not in val["results_dir"].split("/")
    prep = isolate_r2_results_dir("artifacts/results/toy_othello", smoke=False)
    assert prep["results_dir"].endswith("08r2")
    assert not prep["results_dir"].endswith("08r2/validation")


def test_near_zero_random_kl_is_floored():
    missing = random_kl_threshold([])
    assert missing["status"] == "KL_BASELINE_MISSING"
    assert missing["threshold"] is None
    near = random_kl_threshold([0.0, 1e-8, 0.0])
    assert near["status"] == "RANDOM_KL_NEAR_ZERO_FLOOR_APPLIED"
    assert near["threshold"] == 1.5 * 1e-3
    ok = random_kl_threshold([0.2, 0.2])
    assert ok["status"] == "OK"
    assert ok["threshold"] == 1.5 * 0.2


def test_attempt_accounting_keeps_failures_in_denominator():
    rows = [
        {"method": "editor_A_fullspace", "valid_edit": 0, "reason": "nonfinite_loss"},
        {
            "method": "editor_A_fullspace",
            "valid_edit": 1,
            "delta_norm": 2.0,
            "changed_acc": 0.5,
            "unchanged_acc": 0.9,
            "joint_metric_eligible": 1,
            "dP_gain": 0.1,
            "dP_loss": -0.1,
            "S_legal": 0.2,
        },
        {"method": "editor_B_rowspace", "valid_edit": 0, "reason": "nonfinite_grad"},
        {
            "method": "editor_C_reusable",
            "valid_edit": 0,
            "reason": "INSUFFICIENT_TRAIN_PAIRS",
            "baseline_status": "EDITOR_C_FIT_FAILED",
        },
    ]
    acc = editor_attempt_accounting(rows)
    assert acc["editor_A_fullspace"]["attempted"] == 2
    assert acc["editor_A_fullspace"]["executed"] == 1
    assert acc["editor_A_fullspace"]["optimization_failures"] == 1
    assert acc["editor_A_fullspace"]["unconditional_joint_rate"] == 0.5
    assert acc["editor_A_fullspace"]["conditional_mean_S_legal"] == 0.2
    assert acc["editor_C_reusable"]["fit_failures"] == 1
    assert acc["editor_C_reusable"]["executed"] == 0
    assert acc["editor_C_reusable"]["unconditional_joint_rate"] == 0.0


def test_kl_norm_comparisons_flag_donor_access():
    rows = [
        {
            "method": "full_donor_patch",
            "budget": 8.0,
            "r2_type": "A_legal_diff",
            "valid_edit": 1,
            "S_legal": 1.0,
            "kl": 2.0,
            "delta_norm": 10.0,
            "joint_metric_eligible": 1,
            "used_donor_activation": 1,
            "used_model_logits": 0,
        },
        {
            "method": "random_same_norm",
            "budget": 8.0,
            "r2_type": "A_legal_diff",
            "valid_edit": 1,
            "S_legal": 0.0,
            "kl": 0.2,
            "delta_norm": 8.0,
            "joint_metric_eligible": 1,
            "used_donor_activation": 0,
            "used_model_logits": 0,
        },
    ]
    cmp_rows = kl_norm_comparisons(rows)
    donor = next(r for r in cmp_rows if r["method"] == "full_donor_patch" and r["slice"] == "A")
    assert donor["used_donor_activation"] == 1
    assert "donor access" in donor["note"]


def test_amendment_is_git_tracked_path():
    path = Path(__file__).resolve().parents[1] / "configs/08r2_preregistration_amendment_abd.md"
    original = Path(__file__).resolve().parents[1] / "configs/08r2_preregistration.md"
    text = path.read_text(encoding="utf-8")
    assert original.is_file()
    assert "ABD_v1" in text
    assert "cannot establish same-legal-state invariance" in text
    assert "Category C is therefore **unavailable**" in text
    assert "Do not relabel" in text or "does not relabel" in text
