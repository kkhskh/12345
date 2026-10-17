"""The real-model runner must not silently become a mock."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_08r1_real_model_verification.py"


def test_real_runner_exists_and_compiles():
    assert RUNNER.exists()
    py_compile.compile(str(RUNNER), doraise=True)
    text = RUNNER.read_text(encoding="utf-8")
    assert "verify_nanda_checkpoint" in text
    assert "load_othello_gpt" in text
    assert "FakeResidualForward" not in text
    assert "REAL_MODEL_VERIFICATION_PENDING" in text
    assert "does not mock or substitute" in text
    assert "08r1_real_hook_verification.csv" in text
    assert "hook_obj.name" in text
    assert "BLOCKED_MISSING_ARTIFACT" in text
    assert "n-boot >= 1000" in text or "n_boot >= 1000" in text
    assert 'check": "row_labels_match_edited_sites"' not in text
    assert "evaluate_completion" in text
    assert "choose_wrong_donors" in text
    assert "wrong_square_linear_edit" in text
    assert "--probe-path" in text
    assert "resolve_probe_path" in text
    assert "isolate_r1_results_dir" in text
    assert "def patch_fn(act, hook" in text
    assert "def grab_fn(act, hook" in text
    assert "--validate-only" in text
    assert "score_written_outputs" in text


def test_missing_real_stack_exits_pending_not_passed():
    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
    except Exception:
        proc = subprocess.run(
            [sys.executable, str(RUNNER), "--config", str(ROOT / "configs/toy_othello.json")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0
        assert "REAL_MODEL_VERIFICATION_PENDING" in combined
        assert "VERIFIED_ON_REAL_MODEL" not in combined
        return
    pytest.skip("real stack present; missing-stack guard not exercised")
