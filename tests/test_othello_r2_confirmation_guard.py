"""Confirmation must stay sealed during 08R2 validation."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "run_08r2_confirmation.py"
VALIDATE = ROOT / "run_08r2_train_and_validate.py"
PREPARE = ROOT / "run_08r2_prepare.py"


def test_confirmation_and_validate_runners_compile():
    for path in (CONFIRM, VALIDATE, PREPARE, ROOT / "run_08r2_check_readiness.py", ROOT / "src/othello_r2.py"):
        py_compile.compile(str(path), doraise=True)


def test_validate_runner_is_not_a_fake_model_and_keeps_confirmation_sealed():
    text = VALIDATE.read_text(encoding="utf-8")
    assert "FakeResidualForward" not in text
    assert "does not fall back to a fake model" in text
    assert "iterative_state_edit" in text
    assert "require_no_confirmation" in text
    assert "used_donor_activation" in text
    assert "used_model_logits" in text
    assert "evaluate_r2_completeness" in text
    assert "torch_parity_report" in text
    assert "def patch_fn(act, hook" in text
    assert "def patch_fn(act, hook_obj" not in text


def test_confirmation_script_refuses_by_default():
    proc = subprocess.run(
        [sys.executable, str(CONFIRM), "--config", str(ROOT / "configs/toy_othello_08r2.json")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED" in combined
    assert "08R2_VALIDATION_COMPLETE" not in combined


def test_missing_real_stack_exits_colab_required():
    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
    except Exception:
        proc = subprocess.run(
            [sys.executable, str(VALIDATE), "--config", str(ROOT / "configs/toy_othello_08r2.json")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0
        assert "COLAB_REQUIRED" in combined
        assert "08R2_VALIDATION_COMPLETE" not in combined
        return
    import pytest

    pytest.skip("real stack present; missing-stack guard not exercised")
