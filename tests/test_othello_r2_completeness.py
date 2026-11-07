"""Completeness validator must fail closed on empty or invalid outputs."""

from __future__ import annotations

import json
from pathlib import Path

from src.othello_r2_eval import evaluate_r2_completeness, readiness_decision


def _write(dir: Path, name: str, text: str) -> None:
    (dir / name).write_text(text, encoding="utf-8")


def test_empty_csv_is_incomplete(tmp_path: Path):
    for name in (
        "08r2_probe_parity.csv",
        "08r2_optimization_traces.csv",
        "08r2_validation_pairs.json",
        "08r2_validation_state_effects.csv",
        "08r2_validation_causal_effects.csv",
        "08r2_validation_baselines.csv",
        "08r2_validation_bootstrap.csv",
        "08r2_validation_trial_ledger.csv",
        "08r2_validation_summary.json",
        "08r2_reproducibility_manifest.json",
        "08r2_confirmation_protocol.json",
        "08r2_scientific_report.md",
    ):
        if name.endswith(".json"):
            _write(tmp_path, name, "{}\n")
        elif name.endswith(".md"):
            _write(tmp_path, name, "empty\n")
        else:
            _write(tmp_path, name, "method\n")
    out = evaluate_r2_completeness(tmp_path)
    assert out["ok"] == 0
    assert any("empty_or_header_only" in f or "insufficient" in f or "parity" in f or "checkpoint" in f for f in out["failures"])


def test_missing_baseline_and_confirmation_leak_fail(tmp_path: Path):
    _write(tmp_path, "08r2_validation_summary.json", json.dumps({
        "nanda": {"nanda_argmax_match": True},
        "nanda_ok": 1,
        "probe_site": "blocks.6.hook_resid_mid",
        "torch_parity_status": "PASS",
        "grad_checks": [{"grad_finite": True}],
        "editor_c_status": "EDITOR_C_FIT_FAILED",
    }))
    _write(tmp_path, "08r2_optimization_traces.csv", "step,loss\n0,1.0\n1,0.9\n")
    _write(
        tmp_path,
        "08r2_validation_causal_effects.csv",
        "method,valid_edit,recip_game,target_game,dP_gain,dP_loss,kl,delta_norm,changed_acc,unchanged_acc,joint_metric_eligible,dP_gain_defined,dP_loss_defined,baseline_status,synthetic\n"
        "editor_A_fullspace,1,confirm_00000,val_1,0.1,-0.1,0.2,1.0,0.5,0.9,1,1,1,EXECUTED,0\n"
        "editor_B_rowspace,1,val_0,val_1,0.1,-0.1,0.2,1.0,0.5,0.9,1,1,1,EXECUTED,0\n",
    )
    for name in (
        "08r2_probe_parity.csv",
        "08r2_validation_pairs.json",
        "08r2_validation_state_effects.csv",
        "08r2_validation_baselines.csv",
        "08r2_validation_bootstrap.csv",
        "08r2_validation_trial_ledger.csv",
        "08r2_reproducibility_manifest.json",
        "08r2_confirmation_protocol.json",
        "08r2_scientific_report.md",
    ):
        _write(tmp_path, name, "x\n1\n" if name.endswith(".csv") else "{}\n")
    out = evaluate_r2_completeness(tmp_path, smoke=True)
    assert out["ok"] == 0
    assert "confirmation_leak" in out["failures"]


def test_empty_bootstrap_fails(tmp_path: Path):
    _write(tmp_path, "08r2_validation_bootstrap.csv", "")
    out = evaluate_r2_completeness(tmp_path)
    assert "empty_or_header_only_bootstrap" in out["failures"] or "empty_file:08r2_validation_bootstrap.csv" in out["failures"]


def test_readiness_gates_are_distinct():
    assert readiness_decision(imports_ok=True, tests_passed=False, implementation_ok=False, checkpoint=False, probe=False, smoke_passed=False, validation_complete=False) == "ENGINEERING_IMPORT_READY"
    assert readiness_decision(imports_ok=True, tests_passed=True, implementation_ok=True, checkpoint=False, probe=False, smoke_passed=False, validation_complete=False) == "IMPLEMENTATION_TESTS_PASSED"
    assert readiness_decision(imports_ok=True, tests_passed=True, implementation_ok=True, checkpoint=True, probe=True, smoke_passed=False, validation_complete=False) == "READY_FOR_REAL_MODEL_SMOKE_TEST"
    assert readiness_decision(imports_ok=True, tests_passed=True, implementation_ok=True, checkpoint=True, probe=True, smoke_passed=True, validation_complete=False) == "REAL_MODEL_SMOKE_TEST_PASSED"
    assert readiness_decision(imports_ok=True, tests_passed=True, implementation_ok=True, checkpoint=True, probe=True, smoke_passed=True, validation_complete=True) == "08R2_VALIDATION_COMPLETE"
