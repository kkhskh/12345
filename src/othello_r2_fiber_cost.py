"""08R2-FIBER-COST: target specificity at matched output-JS, not matched residual L2.

Does not replace 08R2-FIBER-SPEC. Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.othello_r2_fiber import js_divergence


COST_DIRNAME = "fiber_cost"
JS_ATOL = 0.01
JS_RTOL = 0.05
N_BISECT = 10
ALPHA_HI_WRONG_MULT = 4.0
ALPHA_HI_RANDOM_MULT = 8.0
BEHAVIOR_FLOOR = 0.05
N_BOOT = 2000
BOOT_SEED = 82

VERDICT_SUPPORTED = "TARGET_SPECIFIC_AT_EQUAL_JS_SUPPORTED"
VERDICT_NOT_SUPPORTED = "TARGET_SPECIFIC_AT_EQUAL_JS_NOT_SUPPORTED"
VERDICT_INCONCLUSIVE = "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE"
VERDICT_ENGINEERING = "FIBER_COST_ENGINEERING_FAILED"
STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
STATUS_COMPLETE = "FIBER_COST_DIAGNOSTIC_COMPLETE"

REQUIRED_COST_FILES = (
    "fiber_cost_preregistration.json",
    "fiber_cost_pair_manifest.json",
    "fiber_cost_dependency_manifest.json",
    "fiber_cost_js_match.csv",
    "fiber_cost_interventions.csv",
    "fiber_cost_dose_search.csv",
    "fiber_cost_full_logits.npz",
    "fiber_cost_probe_probabilities.npz",
    "fiber_cost_bootstrap.csv",
    "fiber_cost_attempt_accounting.json",
    "fiber_cost_completion_status.json",
    "fiber_cost_scientific_report.md",
)


def isolate_cost_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == COST_DIRNAME and requested.parent.name == "08r2":
        out = requested
    elif requested.name == "08r2":
        out = requested / COST_DIRNAME
    else:
        out = requested / "08r2" / COST_DIRNAME
    return {"requested": str(requested), "results_dir": str(out), "reason": "08R2-FIBER-COST writes only under 08r2/fiber_cost/"}


def js_match_ok(achieved: float, target: float, atol: float = JS_ATOL, rtol: float = JS_RTOL) -> bool:
    if achieved != achieved or target != target:
        return False
    return abs(float(achieved) - float(target)) <= max(float(atol), float(rtol) * max(abs(float(target)), 1e-12))


def match_output_js(eval_alpha, target_js: float, alpha_hi: float, atol: float = JS_ATOL, rtol: float = JS_RTOL, n_iter: int = N_BISECT) -> dict:
    """Bisection on scale α≥0 so JS(P_E, P(h_E+αn)) matches target_js.

    eval_alpha(alpha) must return {"js": float, ...}. Search assumes JS is
    nondecreasing in α. If the high endpoint cannot reach the target, fail closed.
    """
    target_js = float(target_js)
    hi = float(alpha_hi)
    if hi <= 0 or target_js < 0:
        return {"ok": 0, "reason": "invalid_bracket", "alpha": 0.0, "js": float("nan"), "trace": []}
    cache: dict[float, dict] = {}

    def ev(alpha: float) -> dict:
        key = round(float(alpha), 12)
        if key not in cache:
            out = dict(eval_alpha(float(alpha)))
            out["alpha"] = float(alpha)
            cache[key] = out
        return cache[key]

    trace = []
    f_hi = ev(hi)
    trace.append({"alpha": hi, "js": f_hi.get("js")})
    if f_hi.get("js") != f_hi.get("js"):
        return {"ok": 0, "reason": "nonfinite_js", "alpha": hi, "js": float("nan"), "trace": trace}
    if f_hi["js"] < target_js and not js_match_ok(f_hi["js"], target_js, atol, rtol):
        return {**f_hi, "ok": 0, "reason": "js_unreachable", "alpha": hi, "trace": trace}
    lo = 0.0
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        f = ev(mid)
        trace.append({"alpha": mid, "js": f.get("js")})
        if f.get("js") != f.get("js"):
            return {**f, "ok": 0, "reason": "nonfinite_js", "alpha": mid, "trace": trace}
        if f["js"] < target_js:
            lo = mid
        else:
            hi = mid
    best = ev(hi)
    if {"alpha": hi, "js": best.get("js")} not in trace:
        trace.append({"alpha": hi, "js": best.get("js")})
    candidates = [v for v in cache.values() if js_match_ok(v.get("js", float("nan")), target_js, atol, rtol)]
    if candidates:
        best = min(candidates, key=lambda v: (abs(float(v["js"]) - target_js), float(v["alpha"])))
    ok = js_match_ok(best.get("js", float("nan")), target_js, atol, rtol)
    return {
        **best,
        "ok": int(ok),
        "reason": "ok" if ok else "js_mismatch",
        "target_js": target_js,
        "js_abs_err": abs(float(best.get("js", np.nan)) - target_js),
        "trace": trace,
    }


def load_frozen_spec_manifest(spec_dir: str | Path, expected_hash: str | None) -> dict:
    path = Path(spec_dir) / "fiber_spec_pair_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen SPEC pair manifest: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    got = doc.get("pair_manifest_sha256")
    if expected_hash and got != expected_hash:
        raise ValueError(f"pair hash {got} != frozen {expected_hash}")
    if int(doc.get("n", 0)) != 48 or int(doc.get("n_a", 0)) != 32 or int(doc.get("n_b", 0)) != 16:
        raise ValueError(f"unexpected frozen pair counts {doc.get('n_a')}/{doc.get('n_b')}/{doc.get('n')}")
    if len(doc.get("pairs") or []) != 48 or len(doc.get("wrong_donors") or []) != 48:
        raise ValueError("frozen manifest missing 48 pairs or 48 wrong-donor records")
    return doc


def unit_direction(vec: np.ndarray, min_norm: float = 1e-8) -> np.ndarray | None:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm < float(min_norm):
        return None
    return v / nrm


def decide_cost_verdict(gate: dict) -> str:
    needed = (
        "invariance_ok",
        "js_match_ok",
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
    if all(bool(gate[k]) for k in needed):
        return VERDICT_SUPPORTED
    return VERDICT_NOT_SUPPORTED


def evaluate_cost_completeness(out_dir: str | Path, *, real_model: bool = True) -> dict:
    import json

    out_dir = Path(out_dir)
    failures = []
    files = {name: (out_dir / name).is_file() for name in REQUIRED_COST_FILES}
    for name, exists in files.items():
        if not exists:
            failures.append(f"missing_file:{name}")
        elif (out_dir / name).stat().st_size == 0:
            failures.append(f"empty_file:{name}")
    status = {}
    if files.get("fiber_cost_completion_status.json"):
        status = json.loads((out_dir / "fiber_cost_completion_status.json").read_text(encoding="utf-8"))
    if status.get("used_confirmation"):
        failures.append("confirmation_accessed")
    if real_model and status.get("status") == STATUS_COMPLETE:
        for key in ("checkpoint_loaded", "probe_loaded", "pairs_ok", "js_matched_controls", "full_outputs_saved"):
            if status.get(key) != 1:
                failures.append(f"complete_missing_{key}")
    if "fiber_cost" not in Path(out_dir).parts:
        failures.append("cost_outputs_not_isolated")
    return {"ok": int(not failures), "failures": failures, "files": files}
