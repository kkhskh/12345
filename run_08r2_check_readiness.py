"""08R2 readiness gates. Never writes real-model metric tables."""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

from src.othello_r2 import OFFICIAL_0A_LETTER, confirmation_guard_text
from src.othello_r2_eval import readiness_decision
from src.othello_runutil import write_json


REQUIRED = (
    "src/othello_r2.py",
    "src/othello_r2_eval.py",
    "run_08r2_prepare.py",
    "run_08r2_train_and_validate.py",
    "run_08r2_confirmation.py",
    "run_08r2_check_readiness.py",
    "configs/toy_othello_08r2.json",
    "configs/08r2_preregistration.md",
    "configs/08r2_preregistration_amendment_abd.md",
    "configs/08r2_protocol_option1_full_abcd.md",
    "configs/08r2_protocol_option2_narrow_ad.md",
    "tests/test_othello_r2_protocol_freeze.py",
    "tests/test_othello_r2_probe_parity.py",
    "tests/test_othello_r2_optimizer.py",
    "tests/test_othello_r2_intervention.py",
    "tests/test_othello_r2_pair_splits.py",
    "tests/test_othello_r2_metrics.py",
    "tests/test_othello_r2_confirmation_guard.py",
    "tests/test_othello_r2_editor_c.py",
    "tests/test_othello_r2_baselines.py",
    "tests/test_othello_r2_completeness.py",
)

MARKERS = {
    "run_08r2_train_and_validate.py": (
        "torch_parity_report",
        "valid_edit",
        "EDITOR_C_FIT_FAILED",
        "evaluate_r2_completeness",
        "nanda_argmax_match",
        "FakeResidualForward_ABSENT_SENTINEL",
        "NOT_SELECTED_SMOKE",
        "08r2/smoke",
        "def patch_fn(act, hook",
    ),
    "run_08r2_confirmation.py": (
        "CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED",
    ),
    "src/othello_r2.py": (
        "valid_edit",
        "torch_parity_report",
        "joint_metric_eligible",
        "pair_category_census",
        "probe_direction_edit_safe",
        "editor_attempt_accounting",
        "random_kl_threshold",
        "ABD_v1",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2 readiness gates.")
    p.add_argument("--tests-passed", action="store_true")
    p.add_argument("--out", default="artifacts/results/toy_othello/08r2/08r2_local_readiness.json")
    return p.parse_args()


def _has_real_stack() -> bool:
    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
    except Exception:
        return False
    return True


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    missing = [rel for rel in REQUIRED if not (root / rel).is_file()]
    compile_ok = []
    for rel in REQUIRED:
        path = root / rel
        if path.suffix == ".py" and path.is_file():
            py_compile.compile(str(path), doraise=True)
            compile_ok.append(rel)
    marker_fail = []
    for rel, needles in MARKERS.items():
        text = (root / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle.endswith("_ABSENT_SENTINEL"):
                banned = needle.replace("_ABSENT_SENTINEL", "")
                if banned in text:
                    marker_fail.append(f"{rel} contains banned {banned}")
            elif needle not in text:
                marker_fail.append(f"{rel} missing {needle}")
    ckpt = (root / "artifacts/checkpoints/toy_othello/synthetic_model.pth").is_file()
    probe = (root / "artifacts/checkpoints/toy_othello/08c_selected_probe.pkl").is_file()
    implementation_ok = not missing and not marker_fail
    decision = readiness_decision(
        imports_ok=_has_real_stack(),
        tests_passed=bool(args.tests_passed),
        implementation_ok=implementation_ok,
        checkpoint=ckpt,
        probe=probe,
        smoke_passed=False,
        validation_complete=False,
    )
    payload = {
        "decision": decision,
        "tests_passed": bool(args.tests_passed),
        "real_stack": _has_real_stack(),
        "checkpoint_present": ckpt,
        "probe_present": probe,
        "missing_files": missing,
        "marker_fail": marker_fail,
        "compiled": compile_ok,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "confirmation": "CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED",
        "confirmation_guard": confirmation_guard_text(),
        "note": "Import/implementation readiness only. No real-model smoke or validation claimed.",
    }
    write_json(args.out, payload)
    print(decision)
    print(f"checkpoint_present={ckpt} probe_present={probe}")
    print(f"official letter unchanged: {OFFICIAL_0A_LETTER}")
    if missing or marker_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
