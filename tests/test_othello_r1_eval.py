from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.othello_r1_eval import (
    REQUIRED_REAL_FILES,
    REQUIRED_REAL_STAGES,
    choose_wrong_donors,
    evaluate_completion,
    intended_disagreed_squares,
    isolate_r1_results_dir,
    local_source_readiness,
    nonempty_csv,
    prefix_overlap_by_length,
    real_outputs_scientifically_usable,
    resolve_probe_path,
    score_hook_event,
    score_written_outputs,
    wrong_square_linear_edit,
)


def _all_pass_stages() -> dict[str, str]:
    return {s: "PASS" for s in REQUIRED_REAL_STAGES}


def _all_files_ok() -> dict[str, bool]:
    return {k: True for k in REQUIRED_REAL_FILES}


def test_score_hook_event_fails_on_name_mismatch():
    out = score_hook_event(
        "blocks.6.hook_resid_mid",
        ["blocks.6.hook_resid_post"],
        1,
        (2, 512),
        (2, 512),
        np.zeros((2, 512)),
        np.zeros((2, 512)),
        True,
    )
    assert out["ok"] == 0
    assert "name_mismatch" in out["fail_reason"]


def test_score_hook_event_fails_if_hook_does_not_fire():
    out = score_hook_event("blocks.7.hook_resid_post", [], 0, None, (2, 512), None, None, True)
    assert out["ok"] == 0
    assert "hook_did_not_fire" in out["fail_reason"]


def test_score_hook_event_fails_if_other_tokens_change():
    donor = np.ones((2, 4))
    out = score_hook_event(
        "blocks.6.hook_resid_mid",
        ["blocks.6.hook_resid_mid"],
        2,
        (2, 4),
        (2, 4),
        donor,
        donor,
        False,
    )
    assert out["ok"] == 0
    assert "unrelated_tokens_changed" in out["fail_reason"]


def test_score_hook_event_fails_on_readback_mismatch():
    donor = np.ones((2, 4))
    readback = donor + 0.2
    out = score_hook_event(
        "blocks.6.hook_resid_post",
        ["blocks.6.hook_resid_post"],
        1,
        (2, 4),
        (2, 4),
        readback,
        donor,
        True,
    )
    assert out["ok"] == 0
    assert "readback_ne_donor" in out["fail_reason"]


def test_score_hook_event_passes_only_when_all_conditions_hold():
    donor = np.arange(8, dtype=float).reshape(2, 4)
    out = score_hook_event(
        "blocks.6.hook_resid_post",
        ["blocks.6.hook_resid_post", "blocks.6.hook_resid_post"],
        2,
        (2, 4),
        (2, 4),
        donor,
        donor,
        True,
    )
    assert out["ok"] == 1
    assert out["csv_hook"] == "blocks.6.hook_resid_post"


def test_required_hooks_are_scored_separately():
    names = (
        "blocks.6.hook_resid_mid",
        "blocks.6.hook_resid_post",
        "blocks.7.hook_resid_post",
    )
    donor = np.zeros((1, 3))
    for expected in names:
        ok = score_hook_event(expected, [expected], 1, (1, 3), (1, 3), donor, donor, True)
        bad = score_hook_event(expected, [n for n in names if n != expected][:1], 1, (1, 3), (1, 3), donor, donor, True)
        assert ok["ok"] == 1
        assert bad["ok"] == 0


def test_choose_wrong_donor_is_real_other_state():
    rows = [
        {"player": 1, "occupied": 12, "legal": [1, 2], "terminal": False},
        {"player": 1, "occupied": 13, "legal": [3, 4], "terminal": False},
        {"player": 1, "occupied": 14, "legal": [5, 6], "terminal": False},
        {"player": 1, "occupied": 10, "legal": [7, 8], "terminal": False},
    ]
    pairs = [{"recip": 0, "donor": 1}]
    wrong = choose_wrong_donors(rows, pairs, seed=0)
    assert wrong[0] in {2, 3}
    assert set(rows[wrong[0]]["legal"]) != set(rows[1]["legal"])


def test_choose_wrong_donor_none_when_no_natural_match():
    rows = [
        {"player": 1, "occupied": 12, "legal": [1, 2], "terminal": False},
        {"player": 1, "occupied": 13, "legal": [3, 4], "terminal": False},
    ]
    assert choose_wrong_donors(rows, [{"recip": 0, "donor": 1}], seed=1) == [None]


class _Coef:
    def __init__(self):
        self.coef_ = np.eye(3, 8)


def test_wrong_square_is_not_an_intended_square():
    rec = np.array([0, 1] + [0] * 62)
    don = np.array([1, 1] + [0] * 62)
    assert intended_disagreed_squares(rec, don) == [0]
    rng = np.random.default_rng(0)
    models = [_Coef() for _ in range(64)]
    out = wrong_square_linear_edit(np.zeros(8), rec, don, models, 2.0, rng)
    assert out is not None
    assert out["wrong_square"] != 0
    assert out["natural_state"] == 0
    assert out["control_kind"] == "target_specificity"
    assert np.isclose(out["edit_norm"], 2.0)


def test_prefix_overlap_reports_by_length():
    existing = [[1, 2, 3, 4], [9, 8, 7]]
    hold = [[1, 2, 9, 9], [0, 0, 0]]
    rows = prefix_overlap_by_length(hold, existing, max_len=3)
    assert rows[0]["prefix_length"] == 1
    assert rows[0]["n_overlap"] == 1
    assert rows[1]["n_overlap"] == 1
    assert rows[2]["n_overlap"] == 0
    assert rows[0]["reference"] == "existing_games"


def test_evaluate_completion_blocks_on_missing_artifact():
    stages = _all_pass_stages()
    stages["probe_consistency_08d"] = "BLOCKED_MISSING_ARTIFACT"
    stages["wrong_square_control"] = "BLOCKED_MISSING_ARTIFACT"
    out = evaluate_completion(stages, _all_files_ok())
    assert out["decision"] == "08R1_INCOMPLETE"
    assert out["n_blocked"] == 2
    assert out["n_pass"] < len(REQUIRED_REAL_STAGES)


def test_evaluate_completion_fails_closed_on_stage_fail():
    stages = _all_pass_stages()
    stages["noop_identity"] = "FAIL"
    out = evaluate_completion(stages, _all_files_ok())
    assert out["decision"] == "REAL_MODEL_VALIDATION_FAILED"


def test_evaluate_completion_rejects_synthetic_as_pass():
    out = evaluate_completion(_all_pass_stages(), _all_files_ok(), synthetic_counted=True)
    assert out["decision"] == "08R1_INCOMPLETE"


def test_evaluate_completion_rejects_pending_and_missing_files():
    stages = _all_pass_stages()
    stages["untrained_control"] = "PENDING"
    files = _all_files_ok()
    files["08r1_real_bootstrap_results.csv"] = False
    out = evaluate_completion(stages, files)
    assert out["decision"] == "08R1_INCOMPLETE"
    assert "08r1_real_bootstrap_results.csv" in out["files_missing"]


def test_evaluate_completion_rejects_missing_numeric_results():
    out = evaluate_completion(_all_pass_stages(), _all_files_ok(), numeric_ok=False)
    assert out["decision"] == "08R1_INCOMPLETE"


def test_evaluate_completion_complete_only_when_all_real_passes():
    out = evaluate_completion(_all_pass_stages(), _all_files_ok())
    assert out["decision"] == "08R1_COMPLETE"


def test_score_written_outputs_requires_status_file_already_present(tmp_path: Path):
    for name in REQUIRED_REAL_FILES:
        if name.endswith(".json"):
            continue
        with (tmp_path / name).open("w", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "became_legal_delta_mass",
                    "total_legal_mass_target_after",
                    "ci95_low",
                    "ci95_high",
                    "ok",
                    "mean",
                    "natural_state",
                    "provenance",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "became_legal_delta_mass": "0.1",
                    "total_legal_mass_target_after": "0.2",
                    "ci95_low": "0.05",
                    "ci95_high": "0.15",
                    "ok": "1",
                    "mean": "0.1",
                    "natural_state": "1",
                    "provenance": "real_model",
                }
            )
    missing = score_written_outputs(tmp_path, _all_pass_stages())
    assert missing["decision"] == "08R1_INCOMPLETE"
    assert "08r1_real_model_status.json" in missing["files_missing"]
    (tmp_path / "08r1_real_model_status.json").write_text('{"stages":{}}', encoding="utf-8")
    present = score_written_outputs(tmp_path, _all_pass_stages())
    assert present["decision"] == "08R1_COMPLETE"
    assert present["numeric_ok"] is True


def test_local_source_readiness_ready_when_paths_exist():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run_08r1_real_model_verification.py").read_text(encoding="utf-8")
    helper = (root / "src/othello_r1_eval.py").read_text(encoding="utf-8")
    assert 'check": "row_labels_match_edited_sites"' not in runner
    out = local_source_readiness(runner, helper, tests_passed=True)
    assert out["decision"] == "READY_FOR_COLAB_REAL_RUN"
    blocked = local_source_readiness(runner, helper, tests_passed=False)
    assert blocked["decision"] == "08R1_INCOMPLETE — IMPLEMENTATION_DEFECTS_REMAIN"


def test_real_outputs_reject_synthetic_or_empty(tmp_path: Path):
    for name in REQUIRED_REAL_FILES:
        if name.endswith(".json"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
            continue
        with (tmp_path / name).open("w", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["became_legal_delta_mass", "total_legal_mass_target_after", "ci95_low", "ci95_high", "ok", "mean", "natural_state", "validation_kind"])
            writer.writeheader()
            writer.writerow(
                {
                    "became_legal_delta_mass": "0.1",
                    "total_legal_mass_target_after": "0.2",
                    "ci95_low": "0.0",
                    "ci95_high": "0.3",
                    "ok": "1",
                    "mean": "0.1",
                    "natural_state": "1",
                    "validation_kind": "SYNTHETIC_UNIT_TEST",
                }
            )
    ok, reason = real_outputs_scientifically_usable(tmp_path)
    assert ok is False
    assert "synthetic" in reason or "numeric" in reason or "schema" in reason
    assert nonempty_csv(tmp_path / "08r1_real_corrected_probability_metrics.csv", ["became_legal_delta_mass"])


def test_resolve_probe_path_uses_explicit_file_not_default(tmp_path: Path):
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    default = ckpt / "08c_selected_probe.pkl"
    default.write_bytes(b"default-probe")
    explicit = tmp_path / "drive_root" / "08c_selected_probe.pkl"
    explicit.parent.mkdir()
    explicit.write_bytes(b"discovered-probe")
    info = resolve_probe_path(explicit, ckpt)
    assert info["source"] == "explicit_probe_path"
    assert info["fallback_used"] == 0
    assert info["exists"] is True
    assert Path(info["probe_path"]).read_bytes() == b"discovered-probe"
    missing = resolve_probe_path(tmp_path / "absent.pkl", ckpt)
    assert missing["exists"] is False
    assert missing["fallback_used"] == 0
    assert Path(missing["probe_path"]).name == "absent.pkl"


def test_results_dir_nests_when_historical_artifacts_present(tmp_path: Path):
    hist = tmp_path / "results"
    hist.mkdir()
    (hist / "08a_dataset_manifest.json").write_text("{}", encoding="utf-8")
    (hist / "08i_decision.json").write_text("{}", encoding="utf-8")
    out = isolate_r1_results_dir(hist)
    assert out["nested"] == 1
    assert Path(out["results_dir"]).name == "08r1_real"
    assert "08a_dataset_manifest.json" in out["historical_artifacts_seen"]
    clean = isolate_r1_results_dir(tmp_path / "empty_results")
    assert clean["nested"] == 0


def test_08d_missing_probe_exits_blocked(tmp_path: Path):
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cfg = {
        "results_dir": str(tmp_path / "results"),
        "figures_dir": str(tmp_path / "figures"),
        "checkpoints_dir": str(tmp_path / "ckpts"),
        "data_dir": str(tmp_path / "data"),
    }
    (tmp_path / "ckpts").mkdir()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(root / "run_08d_othello_state_abstraction.py"), "--config", str(cfg_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "BLOCKED_MISSING_ARTIFACT" in combined
    assert "08c_selected_probe.pkl" in combined
