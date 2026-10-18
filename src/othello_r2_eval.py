"""08R2 completeness and readiness gates. No scientific effect sizes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.othello_r2 import REQUIRED_BASELINES, REQUIRED_VALIDATION_FILES, SITE

ESSENTIAL_CAUSAL = ("dP_gain", "dP_loss", "kl", "delta_norm", "changed_acc", "unchanged_acc")
READINESS_GATES = (
    "ENGINEERING_IMPORT_READY",
    "IMPLEMENTATION_TESTS_PASSED",
    "READY_FOR_REAL_MODEL_SMOKE_TEST",
    "REAL_MODEL_SMOKE_TEST_PASSED",
    "08R2_VALIDATION_COMPLETE",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows = list(csv.DictReader(text.splitlines()))
    return rows


def _finite(val) -> bool:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return False
    return x == x


def evaluate_r2_completeness(out_dir: str | Path, *, smoke: bool = False) -> dict:
    out_dir = Path(out_dir)
    failures = []
    files = {name: (out_dir / name).is_file() for name in REQUIRED_VALIDATION_FILES}
    for name, exists in files.items():
        if not exists:
            failures.append(f"missing_file:{name}")
        elif (out_dir / name).stat().st_size == 0:
            failures.append(f"empty_file:{name}")

    summary = {}
    if files.get("08r2_validation_summary.json"):
        summary = json.loads((out_dir / "08r2_validation_summary.json").read_text(encoding="utf-8"))

    if summary.get("nanda_ok") != 1 and summary.get("nanda", {}).get("nanda_argmax_match") is not True:
        failures.append("checkpoint_identity")
    if summary.get("probe_site") not in {SITE, None} and summary.get("probe_site") != SITE:
        failures.append("probe_site_mismatch")
    if summary.get("probe_site") != SITE:
        failures.append("probe_site_not_resid_mid")
    if summary.get("torch_parity_status") != "PASS":
        failures.append("torch_sklearn_parity")
    if not summary.get("grad_checks"):
        failures.append("gradient_checks")
    else:
        for g in summary["grad_checks"]:
            if not g.get("grad_finite"):
                failures.append("gradient_not_finite")

    traces = _read_csv(out_dir / "08r2_optimization_traces.csv")
    if len(traces) < 2:
        failures.append("empty_or_header_only_traces")

    causal = _read_csv(out_dir / "08r2_validation_causal_effects.csv")
    if not causal:
        failures.append("empty_or_header_only_causal")
    valid = [r for r in causal if str(r.get("valid_edit")) in {"1", "1.0"}]
    if len(valid) < (2 if smoke else 8):
        failures.append("insufficient_valid_rows")
    methods = {r.get("method") for r in causal}
    if "editor_A_fullspace" not in methods:
        failures.append("editor_A_missing")
    if "editor_B_rowspace" not in methods:
        failures.append("editor_B_missing")
    c_status = summary.get("editor_c_status")
    if c_status not in {"FIT", "EDITOR_C_FIT_FAILED"}:
        failures.append("editor_C_status_unrecorded")
    if c_status == "FIT" and "editor_C_reusable" not in methods:
        failures.append("editor_C_missing_executed_rows")
    if c_status == "EDITOR_C_FIT_FAILED" and any(r.get("method") == "editor_C_reusable" and str(r.get("valid_edit")) in {"1", "1.0"} for r in causal):
        failures.append("editor_C_identity_claimed_as_success")

    for base in REQUIRED_BASELINES:
        tagged = [r for r in causal if r.get("method") == base]
        if not tagged and base != "wrong_target":
            # wrong-target lives as r2_type rather than method name
            if base == "logit_steer" and summary.get("logit_steer_restriction"):
                continue
            failures.append(f"baseline_missing:{base}")
        if tagged and all(r.get("baseline_status") in {None, ""} for r in tagged):
            failures.append(f"baseline_status_unrecorded:{base}")

    for row in valid:
        if "confirm_" in str(row.get("recip_game", "")) or "confirm_" in str(row.get("target_game", "")):
            failures.append("confirmation_leak")
        if not row.get("recip_game"):
            failures.append("missing_group_id")
        if str(row.get("synthetic")) in {"1", "1.0", "True"}:
            failures.append("synthetic_row_in_real_table")
        if str(row.get("joint_metric_eligible")) in {"1", "1.0"}:
            if not _finite(row.get("S_legal")):
                failures.append("eligible_S_legal_not_finite")
        for key in ESSENTIAL_CAUSAL:
            if key in {"dP_gain", "dP_loss"} and str(row.get(f"{key.replace('dP_', 'dP_')}_defined", "1")) == "0":
                continue
            if key == "dP_gain" and str(row.get("dP_gain_defined")) == "0":
                continue
            if key == "dP_loss" and str(row.get("dP_loss_defined")) == "0":
                continue
            if not _finite(row.get(key)) and key not in {"changed_acc", "unchanged_acc"}:
                if key in {"dP_gain", "dP_loss"}:
                    continue
                failures.append(f"nonfinite:{key}")

    boot = _read_csv(out_dir / "08r2_validation_bootstrap.csv")
    if not boot:
        failures.append("empty_or_header_only_bootstrap")
    finite_boot = [r for r in boot if _finite(r.get("point")) or _finite(r.get("boot_mean"))]
    if boot and not finite_boot:
        failures.append("bootstrap_all_nonfinite")

    parity = _read_csv(out_dir / "08r2_probe_parity.csv")
    if parity and not any(r.get("backend") == "torch_vs_sklearn" for r in parity):
        failures.append("torch_parity_row_missing")

    ok = not failures
    return {
        "ok": int(ok),
        "status": "08R2_VALIDATION_COMPLETE" if ok and not smoke else ("REAL_MODEL_SMOKE_TEST_PASSED" if ok and smoke else "08R2_INCOMPLETE"),
        "failures": sorted(set(failures)),
        "n_causal_rows": len(causal),
        "n_valid_edits": len(valid),
        "n_traces": len(traces),
        "n_boot": len(boot),
    }


def readiness_decision(*, imports_ok: bool, tests_passed: bool, implementation_ok: bool, checkpoint: bool, probe: bool, smoke_passed: bool, validation_complete: bool) -> str:
    if validation_complete:
        return "08R2_VALIDATION_COMPLETE"
    if smoke_passed:
        return "REAL_MODEL_SMOKE_TEST_PASSED"
    if implementation_ok and checkpoint and probe:
        return "READY_FOR_REAL_MODEL_SMOKE_TEST"
    if tests_passed and implementation_ok:
        return "IMPLEMENTATION_TESTS_PASSED"
    if imports_ok:
        return "ENGINEERING_IMPORT_READY"
    return "08R2_IMPLEMENTATION_STILL_INCOMPLETE"
