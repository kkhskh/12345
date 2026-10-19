"""08R2-VISIBLE-NULL-FACTORIAL helpers.

Does not retrain the frozen nullspace map. Not obstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r2_null_state import ridge_fit
from src.othello_r2_state_semantics import predict_zhat


VNF_DIRNAME = "visible_null_factorial"
COHORT_SEED = 10182
PREFIX_SEED = 10282
TRIPLET_SEED = 10382
BOOT_SEED = 10482
RANDOM_SEED = 10582
SHUFFLE_SEED = 10682
RHO = 3.5
NU = 3.5
I_FLOOR = 0.005
COVERAGE_MIN = 29
MIN_DIR = 1e-6
STATUS_INFEASIBLE = "VISIBLE_NULL_CENSUS_INFEASIBLE"
STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
VERDICT_SUPPORTED = "TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_SUPPORTED"
VERDICT_NOT = "TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED"
VERDICT_INCONCLUSIVE = "VISIBLE_NULL_FACTORIAL_INCONCLUSIVE"
VERDICT_ENGINEERING = "VISIBLE_NULL_FACTORIAL_ENGINEERING_FAILED"
BANNED = {
    "null_state_transfer",
    "state_semantics",
    "fiber_spec",
    "fiber_cost",
    "fiber_cost_forensic",
    "fiber_cost_fixed_js",
    "symmetric_target",
    "symmetric_target_census_audit",
    "symmetric_target_fresh",
}


def jsonable_prefix(row: dict) -> dict:
    skip = {"abs_lab", "rel_lab", "legal_mask", "board_np"}
    out = {}
    for key, val in row.items():
        if key in skip:
            continue
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            out[key] = val.item()
        else:
            out[key] = val
    return out


def isolate_vnf_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == VNF_DIRNAME:
        out = requested
    elif requested.name == "08r2":
        out = requested / VNF_DIRNAME
    else:
        out = requested / "results" / "08r2" / VNF_DIRNAME
    if any(p in BANNED for p in out.parts):
        raise ValueError(f"visible-null factorial must not write under preserved dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out)}


def unit(vec: np.ndarray, min_norm: float = MIN_DIR) -> dict:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < float(min_norm):
        return {"ok": 0, "unit": None, "norm": n}
    return {"ok": 1, "unit": v / n, "norm": n}


def project_check(vec: np.ndarray, P_R: np.ndarray, P_N: np.ndarray, expect: str, atol: float = 1e-5) -> dict:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    r_res = float(np.linalg.norm(P_N @ v)) if expect == "row" else float(np.linalg.norm(P_R @ v))
    return {"ok": int(r_res <= float(atol)), "leak": r_res, "expect": expect}


def rhat(b: np.ndarray, pack: dict) -> np.ndarray:
    x = np.asarray(b, dtype=np.float64).reshape(-1)
    pred = x @ np.asarray(pack["W"], dtype=np.float64) + np.asarray(pack["c"], dtype=np.float64)
    return np.asarray(pack["P_R"], dtype=np.float64) @ pred


def factorial_cells(h_r, r_t, r_w, n_t, n_w, rho: float = RHO, nu: float = NU, min_norm: float = MIN_DIR) -> dict:
    ut, uw, nt, nw = unit(r_t, min_norm), unit(r_w, min_norm), unit(n_t, min_norm), unit(n_w, min_norm)
    if not all(u["ok"] for u in (ut, uw, nt, nw)):
        return {"ok": 0, "reason": "degenerate_direction", "units": {"r_T": ut, "r_W": uw, "n_T": nt, "n_W": nw}}
    h = np.asarray(h_r, dtype=np.float64).reshape(-1)
    rho, nu = float(rho), float(nu)
    cells = {
        "TT": h + rho * ut["unit"] + nu * nt["unit"],
        "TW": h + rho * ut["unit"] + nu * nw["unit"],
        "WT": h + rho * uw["unit"] + nu * nt["unit"],
        "WW": h + rho * uw["unit"] + nu * nw["unit"],
        "vis_T": h + rho * ut["unit"],
        "vis_W": h + rho * uw["unit"],
        "nul_T": h + nu * nt["unit"],
        "nul_W": h + nu * nw["unit"],
        "noop": h.copy(),
    }
    target = float(np.sqrt(rho * rho + nu * nu))
    l2 = {k: float(np.linalg.norm(cells[k] - h)) for k in cells}
    four = all(abs(l2[k] - target) <= 1e-6 for k in ("TT", "TW", "WT", "WW"))
    return {"ok": int(four), "cells": cells, "l2": l2, "target_l2": target, "units": {"r_T": ut["unit"], "r_W": uw["unit"], "n_T": nt["unit"], "n_W": nw["unit"]}}


def interaction_scores(y_t: dict, y_w: dict) -> dict:
    d_t_t = float(y_t["TT"] - y_t["TW"])
    d_t_w = float(y_t["WT"] - y_t["WW"])
    d_w_w = float(y_w["WW"] - y_w["WT"])
    d_w_t = float(y_w["TW"] - y_w["TT"])
    return {
        "d_T_T": d_t_t,
        "d_T_W": d_t_w,
        "d_W_W": d_w_w,
        "d_W_T": d_w_t,
        "I_T": d_t_t - d_t_w,
        "I_W": d_w_w - d_w_t,
    }


def random_null_unit(P_N: np.ndarray, rng: np.random.Generator, min_norm: float = MIN_DIR) -> np.ndarray | None:
    v = rng.normal(size=int(np.asarray(P_N).shape[0]))
    v = np.asarray(P_N, dtype=np.float64) @ v
    u = unit(v, min_norm)
    return u["unit"] if u["ok"] else None


def fit_rowspace_ridge(B: np.ndarray, R: np.ndarray, P_R: np.ndarray, alpha: float = 10.0) -> dict:
    ridge = ridge_fit(B, R, alpha)
    return {"W": ridge["W"], "c": ridge["c"], "alpha": float(alpha), "P_R": np.asarray(P_R, dtype=np.float64), "refit_on_causal_y": 0}


def decide_vnf_verdict(gate: dict) -> str:
    needed = ("split_ok", "integrity_ok", "coverage_ok", "projection_ok", "l2_ok")
    if any(gate.get(k) is None for k in needed):
        return VERDICT_INCONCLUSIVE
    if not gate.get("integrity_ok") or not gate.get("split_ok") or not gate.get("projection_ok"):
        return VERDICT_ENGINEERING
    if not gate.get("coverage_ok") or not gate.get("l2_ok"):
        return VERDICT_INCONCLUSIVE
    if gate.get("interaction_ok") and not gate.get("control_reproduces"):
        return VERDICT_SUPPORTED
    return VERDICT_NOT


def interaction_ok(i_t: dict, i_w: dict, floor: float = I_FLOOR) -> int:
    return int(
        i_t.get("point", 0) >= float(floor)
        and i_w.get("point", 0) >= float(floor)
        and i_t.get("ci95_low", 0) > 0
        and i_w.get("ci95_low", 0) > 0
    )
