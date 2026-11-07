"""08R2-FIBER-COST fixed-JS restoration. Independent of the donor-reference COST run.

Writes only under results/08r2/fiber_cost_fixed_js/. Does not overwrite fiber_cost/.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.othello_r2_fiber import js_divergence, softmax


FIXED_JS_DIRNAME = "fiber_cost_fixed_js"
FIXED_JS_PRIMARY = 0.01
FIXED_JS_SECONDARY = (0.005, 0.02)
FIXED_JS_TOL = 0.0002
SCALE_GRID = (0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 4.0, 8.0)
N_BISECT = 16
BEHAVIOR_FLOOR = 0.02
COVERAGE_MIN = 44
N_BOOT = 2000
BOOT_SEED = 82
CELLS = ("correct_null", "wrong_null_0", "wrong_null_1", "wrong_null_2", "random_null_0", "random_null_1", "random_null_2")
FORBIDDEN_STATUSES = frozenset({"reference", "donor_reference", "unscaled_reference"})

VERDICT_SUPPORTED = "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED"
VERDICT_NOT_SUPPORTED = "FIXED_JS_TARGET_SPECIFICITY_NOT_SUPPORTED"
VERDICT_INCONCLUSIVE = "FIXED_JS_TARGET_SPECIFICITY_INCONCLUSIVE"
VERDICT_ENGINEERING = "FIXED_JS_ENGINEERING_FAILED"
STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
STATUS_COMPLETE = "FIXED_JS_DIAGNOSTIC_COMPLETE"

REQUIRED_FIXED_FILES = (
    "fixed_js_preregistration.json",
    "fixed_js_pair_manifest.json",
    "fixed_js_preflight.csv",
    "fixed_js_calibrations.csv",
    "fixed_js_interventions.csv",
    "fixed_js_dose_search.csv",
    "fixed_js_full_logits.npz",
    "fixed_js_probe_probabilities.npz",
    "fixed_js_bootstrap.csv",
    "fixed_js_attempt_accounting.json",
    "fixed_js_dependency_manifest.json",
    "fixed_js_completion_status.json",
    "fixed_js_scientific_report.md",
)


def isolate_fixed_js_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == FIXED_JS_DIRNAME:
        out = requested
    elif requested.name == "fiber_cost":
        out = requested.parent.parent / "results" / "08r2" / FIXED_JS_DIRNAME
    elif requested.name == "08r2":
        out = requested.parent / "results" / "08r2" / FIXED_JS_DIRNAME
    else:
        out = requested / "results" / "08r2" / FIXED_JS_DIRNAME
    if "fiber_cost" in out.parts and out.name != FIXED_JS_DIRNAME:
        raise ValueError(f"fixed-JS outputs must not write under fiber_cost/: {out}")
    if "fiber_spec" in out.parts or "fiber_cost_forensic" in out.parts:
        raise ValueError(f"fixed-JS outputs must not write under preserved dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out), "reason": "restored fixed-JS COST writes only under results/08r2/fiber_cost_fixed_js/"}


def fixed_js_ok(achieved: float, target: float = FIXED_JS_PRIMARY, tol: float = FIXED_JS_TOL) -> bool:
    if achieved != achieved or target != target:
        return False
    return abs(float(achieved) - float(target)) <= float(tol)


def bind_literal_js_target(tasks: list[dict], target_js: float) -> list[dict]:
    if float(target_js) != float(FIXED_JS_PRIMARY) and float(target_js) not in {float(x) for x in FIXED_JS_SECONDARY}:
        raise ValueError(f"unregistered JS target {target_js}")
    out = []
    for task in tasks:
        row = dict(task)
        row["target_js"] = float(target_js)
        row["calibration"] = "fixed_js"
        row.pop("donor_reference_js", None)
        out.append(row)
    return out


def build_direction_tasks(pairs: list[dict], wrongs: list[dict], n_random: int = 3) -> list[dict]:
    tasks = []
    for pair, wrep in zip(pairs, wrongs):
        donors = [pair["target_game"]]
        donors.extend(d.get("game_id") for d in (wrep.get("donors") or []))
        while len(donors) < 4:
            donors.append(None)
        donors.extend(["random"] * int(n_random))
        kinds = ("correct", "wrong", "wrong", "wrong", "random", "random", "random")
        for cell, kind, donor in zip(CELLS, kinds, donors[:7]):
            tasks.append(
                {
                    "recip_game": pair["recip_game"],
                    "target_game": pair["target_game"],
                    "r2_type": pair.get("r2_type"),
                    "recip": pair.get("recip"),
                    "donor": pair.get("donor"),
                    "cell": cell,
                    "direction_type": kind,
                    "donor_id": donor,
                    "target_js": float(FIXED_JS_PRIMARY),
                    "calibration": "fixed_js",
                }
            )
    return bind_literal_js_target(tasks, FIXED_JS_PRIMARY)


def calibrate_fixed_js(eval_t, *, target_js: float = FIXED_JS_PRIMARY, grid=SCALE_GRID, tol: float = FIXED_JS_TOL, n_bisect: int = N_BISECT) -> dict:
    """First-crossing on the frozen scale grid, then bisection. Does not use Y."""
    target_js = float(target_js)
    cache: dict[float, dict] = {}

    def ev(t: float) -> dict:
        key = round(float(t), 12)
        if key not in cache:
            out = dict(eval_t(float(t)))
            out["t"] = float(t)
            cache[key] = out
        return cache[key]

    trace = []
    lo = 0.0
    hi = None
    for t in grid:
        f = ev(float(t))
        trace.append({"t": float(t), "js": f.get("js")})
        if f.get("js") != f.get("js"):
            return {**f, "ok": 0, "status": "nonfinite_js", "reason": "nonfinite_js", "target_js": target_js, "trace": trace}
        if f["js"] >= target_js:
            hi = float(t)
            break
        lo = float(t)
    if hi is None:
        last = ev(float(grid[-1]))
        return {
            **last,
            "ok": 0,
            "status": "js_unreachable",
            "reason": "js_unreachable",
            "target_js": target_js,
            "js_abs_err": abs(float(last.get("js", np.nan)) - target_js),
            "trace": trace,
        }
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        f = ev(mid)
        trace.append({"t": mid, "js": f.get("js")})
        if f.get("js") != f.get("js"):
            return {**f, "ok": 0, "status": "nonfinite_js", "reason": "nonfinite_js", "target_js": target_js, "trace": trace}
        if f["js"] < target_js:
            lo = mid
        else:
            hi = mid
    best = ev(hi)
    ok = fixed_js_ok(best.get("js", float("nan")), target_js, tol)
    return {
        **best,
        "ok": int(ok),
        "status": "calibrated" if ok else "js_mismatch",
        "reason": "ok" if ok else "js_mismatch",
        "target_js": target_js,
        "js_abs_err": abs(float(best.get("js", np.nan)) - target_js),
        "trace": trace,
    }


def scaled_displacement(direction: np.ndarray, t: float) -> dict:
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    vec = float(t) * d
    return {"vec": vec, "t": float(t), "l2": float(np.linalg.norm(vec)), "base_l2": float(np.linalg.norm(d))}


def reject_donor_reference_rows(rows: list[dict], expected_target: float = FIXED_JS_PRIMARY) -> dict:
    failures = []
    for i, row in enumerate(rows):
        if row.get("cell") not in CELLS:
            continue
        if row.get("reason") in FORBIDDEN_STATUSES or row.get("status") in FORBIDDEN_STATUSES:
            failures.append(f"row{i}:forbidden_status:{row.get('reason') or row.get('status')}")
        tgt = row.get("target_js")
        try:
            if tgt is not None and abs(float(tgt) - float(expected_target)) > 1e-12:
                failures.append(f"row{i}:non_literal_target:{tgt}")
        except (TypeError, ValueError):
            failures.append(f"row{i}:non_numeric_target")
    return {"ok": int(not failures), "failures": failures}


def evaluate_fixed_js_completeness(out_dir: str | Path, *, real_model: bool = True, expected_target: float = FIXED_JS_PRIMARY) -> dict:
    out_dir = Path(out_dir)
    failures = []
    files = {name: (out_dir / name).is_file() for name in REQUIRED_FIXED_FILES}
    for name, exists in files.items():
        if not exists:
            failures.append(f"missing_file:{name}")
        elif (out_dir / name).stat().st_size == 0:
            failures.append(f"empty_file:{name}")
    if "fiber_cost_fixed_js" not in Path(out_dir).parts:
        failures.append("fixed_js_outputs_not_isolated")
    if "fiber_cost" in Path(out_dir).parts and Path(out_dir).name != FIXED_JS_DIRNAME:
        failures.append("wrote_into_original_fiber_cost")
    status = {}
    if files.get("fixed_js_completion_status.json"):
        status = json.loads((out_dir / "fixed_js_completion_status.json").read_text(encoding="utf-8"))
    if status.get("used_confirmation"):
        failures.append("confirmation_accessed")
    if files.get("fixed_js_calibrations.csv"):
        import csv

        with (out_dir / "fixed_js_calibrations.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        chk = reject_donor_reference_rows(rows, expected_target)
        if not chk["ok"]:
            failures.extend([f"donor_reference:{x}" for x in chk["failures"][:8]])
            failures.append("donor_reference_matching_is_not_fixed_js_compliance")
    if real_model and status.get("status") == STATUS_COMPLETE:
        for key in ("checkpoint_loaded", "probe_loaded", "pairs_ok", "preflight_ok", "full_outputs_saved"):
            if status.get(key) != 1:
                failures.append(f"complete_missing_{key}")
    return {"ok": int(not failures), "failures": failures, "files": files}


def decide_fixed_js_verdict(gate: dict) -> str:
    needed = (
        "invariance_ok",
        "js_match_ok",
        "coverage_ok",
        "mean_delta_specific_positive",
        "ci_lower_gt_0",
        "mean_delta_ge_floor",
        "both_components_correct_sign",
        "beats_random",
        "integrity_ok",
    )
    if any(gate.get(k) is None for k in needed):
        return VERDICT_INCONCLUSIVE
    if not gate.get("integrity_ok"):
        return VERDICT_ENGINEERING
    if not gate.get("coverage_ok") or not gate.get("js_match_ok") or not gate.get("invariance_ok"):
        return VERDICT_INCONCLUSIVE
    if all(bool(gate[k]) for k in needed):
        return VERDICT_SUPPORTED
    return VERDICT_NOT_SUPPORTED


def build_fixed_js_gate(attempts: dict, n_pairs: int, *, hooks_ok: int, parity_ok: int, slices: dict, behavior_floor: float = BEHAVIOR_FLOOR, coverage_min: int = COVERAGE_MIN) -> dict:
    pooled = slices.get("pooled") or {}
    ci = pooled.get("delta_specific_ci") or {}
    rci = pooled.get("delta_random_ci") or {}
    valid = int(attempts.get("valid", 0))
    return {
        "invariance_ok": int(int(attempts.get("invariance_fail", 0)) == 0 and valid > 0),
        "js_match_ok": int(int(attempts.get("js_fail", 0)) == 0 and valid == int(n_pairs)),
        "coverage_ok": int(valid >= int(coverage_min)),
        "mean_delta_specific_positive": int(pooled.get("mean_delta_specific", 0) > 0),
        "ci_lower_gt_0": int(ci.get("ci95_low", 0) > 0) if ci else 0,
        "mean_delta_ge_floor": int(pooled.get("mean_delta_specific", 0) >= float(behavior_floor)),
        "both_components_correct_sign": int(pooled.get("mean_dG", 0) > 0 and pooled.get("mean_dQ", 0) < 0),
        "beats_random": int(pooled.get("mean_delta_random", 0) > 0 and (rci.get("ci95_low", 0) > 0 if rci else 0)),
        "integrity_ok": int(bool(hooks_ok) and bool(parity_ok) and int(attempts.get("editor_fail", 0)) == 0 and int(attempts.get("pairs", 0)) == int(n_pairs)),
    }


def recompute_js_from_logits(editor_logits: np.ndarray, intervention_logits: np.ndarray) -> float:
    return float(js_divergence(softmax(editor_logits), softmax(intervention_logits)))
