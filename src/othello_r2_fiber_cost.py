"""08R2-FIBER-COST: target specificity at matched output-JS, not matched residual L2.

Does not replace 08R2-FIBER-SPEC. Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.othello_r2_fiber import js_divergence
from src.othello_r2_fiber_spec import (
    N_WRONG,
    PAIR_SEED,
    WANT_A,
    WANT_B,
    WRONG_SEED,
    enumerate_candidates,
    fiber_excluded_games,
    isolate_spec_dir,
    match_wrong_donors,
    select_disjoint_pairs,
)


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


COMMON_SPEC_DIRS = (
    "/content/drive/MyDrive/ml_research_artifacts/toy_othello/08r2/fiber_spec",
    "/content/12345/artifacts/results/toy_othello/08r2/fiber_spec",
    "/content/ml_research/artifacts/results/toy_othello/08r2/fiber_spec",
    "artifacts/results/toy_othello/08r2/fiber_spec",
)
DRIVE_ROOT = Path("/content/drive/MyDrive/ml_research_artifacts/toy_othello")


def resolve_spec_dir(explicit: str | Path | None, results_dir: str | Path) -> dict:
    """Return an existing SPEC folder if present. Missing is not fatal; pairs can be rebuilt."""
    tried = []
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(isolate_spec_dir(results_dir)["results_dir"]),
            Path(results_dir) / "08r2" / "fiber_spec",
            Path(results_dir).parent / "08r2" / "fiber_spec",
        ]
    )
    candidates.extend(Path(p) for p in COMMON_SPEC_DIRS)
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if (path / "fiber_spec_pair_manifest.json").is_file():
            source = "explicit" if explicit and Path(explicit) == path else "discovered"
            return {"spec_dir": path, "source": source, "tried": tried}
    return {"spec_dir": None, "source": "missing", "tried": tried}


def reconstruct_frozen_pairs(
    val_rows: list[dict],
    holdout_ids: set[str],
    expected_hash: str,
    *,
    n_a: int = WANT_A,
    n_b: int = WANT_B,
    pair_seed: int = PAIR_SEED,
    n_wrong: int = N_WRONG,
    wrong_seed: int = WRONG_SEED,
    exclude_fiber: bool = True,
) -> dict:
    excl = fiber_excluded_games(val_rows, holdout_ids) if exclude_fiber else set()
    enum = enumerate_candidates(val_rows, holdout_ids, excl)
    selected = select_disjoint_pairs(enum["found"], int(n_a), int(n_b), int(pair_seed))
    got = selected["pair_manifest_sha256"]
    if expected_hash and got != expected_hash:
        raise ValueError(f"reconstructed pair hash {got} != frozen {expected_hash}")
    if not selected.get("feasible") or selected["n_a"] != int(n_a) or selected["n_b"] != int(n_b):
        raise ValueError(f"reconstructed pair set infeasible A={selected['n_a']} B={selected['n_b']}")
    wrongs = [match_wrong_donors(enum["labeled"], p, holdout_ids, int(n_wrong), int(wrong_seed)) for p in selected["pairs"]]
    if any(not w.get("ok") for w in wrongs):
        raise ValueError("reconstructed wrong-donor matching incomplete")
    return {
        "n": len(selected["pairs"]),
        "n_a": selected["n_a"],
        "n_b": selected["n_b"],
        "feasible": selected["feasible"],
        "n_recipient_games": selected["n_recipient_games"],
        "n_target_games": selected["n_target_games"],
        "recip_target_overlap": selected["recip_target_overlap"],
        "pair_manifest_sha256": got,
        "expected_sha256": expected_hash,
        "hash_ok": 1,
        "seed": selected["seed"],
        "exclude_fiber_games": 1,
        "pairs": selected["pairs"],
        "wrong_donors": wrongs,
        "source": "reconstructed_from_frozen_seeds",
    }


def adopt_drive_defaults(cfg: dict, args) -> dict:
    adopted = {}
    if not DRIVE_ROOT.is_dir():
        return adopted
    if getattr(args, "results_dir", None) is None and Path(cfg.get("results_dir", "")).as_posix().startswith("artifacts/"):
        cfg["results_dir"] = str(DRIVE_ROOT)
        adopted["results_dir"] = cfg["results_dir"]
    if getattr(args, "data_dir", None) is None and not (Path(cfg["data_dir"]) / "prefixes_val.jsonl").is_file():
        if (DRIVE_ROOT / "data" / "prefixes_val.jsonl").is_file():
            cfg["data_dir"] = str(DRIVE_ROOT / "data")
            adopted["data_dir"] = cfg["data_dir"]
    if getattr(args, "checkpoints_dir", None) is None:
        ckpt_name = cfg.get("checkpoint_file", "synthetic_model.pth")
        if not (Path(cfg["checkpoints_dir"]) / ckpt_name).is_file() and (DRIVE_ROOT / "checkpoints" / ckpt_name).is_file():
            cfg["checkpoints_dir"] = str(DRIVE_ROOT / "checkpoints")
            adopted["checkpoints_dir"] = cfg["checkpoints_dir"]
    return adopted


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
