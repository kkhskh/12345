"""08R2-STATE-SEMANTICS: exploratory ablation of the frozen donor-free map.

Does not overwrite NST/SPEC/COST/fixed-JS. Official transfer letter is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r2_null_state import FEATURE_NAMES, pca_coords, predict_z, r2_score, ridge_fit
from src.toy_othello import board_hamming


SEM_DIRNAME = "state_semantics"
NST_PAIR_HASH = "39abdc79090211ab4a69e3e5f79f14319231705640f231dcc146233a9f5092e6"
FROZEN_K = 64
FROZEN_ALPHA = 10.0
FIXED_JS = 0.01
FIXED_JS_TOL = 0.0002
BEHAVIOR_FLOOR = 0.01
COVERAGE_MIN = 44
N_BOOT = 5000
BOOT_SEED = 82
N_FOLDS = 5
CV_SEED = 6182
DIR_ATOL = 1e-4
Y_RECON_ATOL = 1e-6

STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
STATUS_STAGE0_FAIL = "STATE_SEMANTICS_STAGE0_FAIL"
STATUS_STAGE1_FAIL = "STATE_SEMANTICS_RECONSTRUCTION_FAIL"
VERDICT_BOARD = "BOARD_GEOMETRY_INCREMENT_SUPPORTED"
VERDICT_LEGAL = "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY"
VERDICT_MIXED = "FEATURE_SEMANTICS_MIXED"
VERDICT_INCONCLUSIVE = "STATE_SEMANTICS_INCONCLUSIVE"
VERDICT_ENGINEERING = "STATE_SEMANTICS_ENGINEERING_FAILED"

ABLATIONS = ("FULL", "BOARD_ONLY", "LEGAL_ONLY", "STAGE_ONLY")
JS_METHODS = ("FULL", "BOARD_ONLY", "LEGAL_ONLY", "STAGE_ONLY", "TARGET_COMP")
PRIMARY_JS = ("FULL", "BOARD_ONLY", "LEGAL_ONLY", "STAGE_ONLY")
BANNED_DIRS = {
    "null_state_transfer",
    "fiber_spec",
    "fiber_cost",
    "fiber_cost_forensic",
    "fiber_cost_fixed_js",
}


def isolate_sem_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == SEM_DIRNAME:
        out = requested
    elif requested.name == "08r2":
        out = requested / SEM_DIRNAME
    else:
        out = requested / "results" / "08r2" / SEM_DIRNAME
    if any(p in BANNED_DIRS for p in out.parts):
        raise ValueError(f"state-semantics must not write under preserved dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out)}


def feature_indices(name: str) -> np.ndarray:
    name = str(name)
    if name == "FULL":
        return np.arange(len(FEATURE_NAMES))
    if name == "BOARD_ONLY":
        return np.concatenate([np.arange(192), [256, 257]])
    if name == "LEGAL_ONLY":
        return np.concatenate([np.arange(192, 256), [256, 257]])
    if name == "STAGE_ONLY":
        return np.array([256, 257])
    raise ValueError(name)


def feature_names_for(name: str) -> list[str]:
    idx = feature_indices(name)
    return [FEATURE_NAMES[i] for i in idx]


def slice_features(B: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(B, dtype=np.float64)[..., feature_indices(name)]


def predict_zhat(b_full: np.ndarray, pack: dict) -> np.ndarray:
    b = np.asarray(b_full, dtype=np.float64).reshape(-1)
    idx = np.asarray(pack["idx"], dtype=np.int64)
    z = predict_z(b[idx].reshape(1, -1), {"mu": pack["mu"], "V": pack["V"]}, {"W": pack["W"], "c": pack["c"]})[0]
    return np.asarray(pack["P_N"], dtype=np.float64) @ z


def ablation_delta(h_e: np.ndarray, b_full: np.ndarray, pack: dict) -> np.ndarray:
    z_e = np.asarray(pack["P_N"], dtype=np.float64) @ np.asarray(h_e, dtype=np.float64).reshape(-1)
    return predict_zhat(b_full, pack) - z_e


def decompose_full_stage(h_e: np.ndarray, b_full: np.ndarray, full: dict, stage: dict) -> dict:
    z_e = np.asarray(full["P_N"], dtype=np.float64) @ np.asarray(h_e, dtype=np.float64).reshape(-1)
    z_full = predict_zhat(b_full, full)
    z_stage = predict_zhat(b_full, stage)
    generic = z_stage - z_e
    target = z_full - z_stage
    return {
        "delta_full": z_full - z_e,
        "delta_generic": generic,
        "delta_target": target,
        "sum_err": float(np.linalg.norm((generic + target) - (z_full - z_e))),
    }


def fit_frozen_ridge(Z: np.ndarray, B: np.ndarray, games: list[str], pca: dict, name: str, alpha: float = FROZEN_ALPHA, n_folds: int = N_FOLDS, seed: int = CV_SEED) -> dict:
    Z = np.asarray(Z, dtype=np.float64)
    Bsub = slice_features(B, name)
    games = [str(g) for g in games]
    coords = pca_coords(Z, pca)
    from src.othello_r2_null_state import grouped_folds

    folds = grouped_folds(games, n_folds, seed)
    scores = []
    for held in folds:
        held = set(map(str, held))
        tr = np.array([g not in held for g in games])
        te = ~tr
        if tr.sum() < 8 or te.sum() < 4:
            continue
        ridge = ridge_fit(Bsub[tr], coords[tr], float(alpha))
        pred = predict_z(Bsub[te], pca, ridge)
        scores.append(r2_score(Z[te], pred))
    ridge = ridge_fit(Bsub, coords, float(alpha))
    pred = predict_z(Bsub, pca, ridge)
    return {
        "name": name,
        "idx": feature_indices(name),
        "ridge": ridge,
        "cv_r2": float(np.nanmean(scores)) if scores else float("nan"),
        "n_scores": len(scores),
        "train_r2": r2_score(Z, pred),
        "n_features": int(Bsub.shape[1]),
        "k": int(pca["k"]),
        "alpha": float(alpha),
    }


def pack_ablation(fit: dict, pca: dict, P_N: np.ndarray) -> dict:
    return {
        "name": fit["name"],
        "mu": pca["mu"],
        "V": pca["V"],
        "W": fit["ridge"]["W"],
        "c": fit["ridge"]["c"],
        "idx": np.asarray(fit["idx"], dtype=np.int64),
        "P_N": np.asarray(P_N, dtype=np.float64),
        "k": int(pca["k"]),
        "alpha": float(fit["alpha"]),
        "feature_names": feature_names_for(fit["name"]),
    }


def pack_full_from_nst(nst: dict) -> dict:
    return {
        "name": "FULL",
        "mu": np.asarray(nst["mu"], dtype=np.float64),
        "V": np.asarray(nst["V"], dtype=np.float64),
        "W": np.asarray(nst["W"], dtype=np.float64),
        "c": np.asarray(nst["c"], dtype=np.float64),
        "idx": feature_indices("FULL"),
        "P_N": np.asarray(nst["P_N"], dtype=np.float64),
        "k": int(nst["k"]) if np.ndim(nst["k"]) == 0 else int(nst["k"].reshape(-1)[0]),
        "alpha": float(nst["alpha"]) if np.ndim(nst["alpha"]) == 0 else float(nst["alpha"].reshape(-1)[0]),
        "feature_names": list(FEATURE_NAMES),
    }


def direction_parity(computed: np.ndarray, saved: np.ndarray, atol: float = DIR_ATOL) -> dict:
    a = np.asarray(computed, dtype=np.float64).reshape(-1)
    b = np.asarray(saved, dtype=np.float64).reshape(-1)
    err = float(np.linalg.norm(a - b))
    return {"ok": int(err <= float(atol)), "l2_err": err, "atol": float(atol)}


def probe_image_residual(delta: np.ndarray, W_probe: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(W_probe, dtype=np.float64) @ np.asarray(delta, dtype=np.float64).reshape(-1)))


def paired_boot(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[x == x]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "point": float("nan"), "median": float("nan"), "boot_mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "win_frac": float("nan")}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        means[i] = x[rng.integers(0, n, n)].mean()
    return {
        "n": n,
        "point": float(x.mean()),
        "median": float(np.median(x)),
        "boot_mean": float(means.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "win_frac": float((x > 0).mean()),
        "n_boot": int(n_boot),
    }


def match_controls(rec: dict, tgt: dict, wrong: dict) -> dict:
    from src.othello_runutil import board_arr

    lr = set(int(s) for s in rec["legal"])
    lt = set(int(s) for s in tgt["legal"])
    lw = set(int(s) for s in wrong["legal"])
    union = lt | lw
    return {
        "player_match": int(int(rec["player"]) == int(tgt["player"]) == int(wrong["player"])),
        "occ_diff_tw": abs(int(tgt["occupied"]) - int(wrong["occupied"])),
        "legal_count_diff_tw": abs(len(lt) - len(lw)),
        "prefix_diff_tw": abs(len(tgt["tokens"]) - len(wrong["tokens"])),
        "hamming_tw": int(board_hamming(board_arr(tgt), board_arr(wrong))),
        "legal_jaccard_tw": float(len(lt & lw) / len(union)) if union else float("nan"),
        "n_became_T": int(len(lt - lr)),
        "n_became_W": int(len(lw - lr)),
        "n_left_T": int(len(lr - lt)),
        "n_left_W": int(len(lr - lw)),
        "occ_slack_ok": int(abs(int(wrong["occupied"]) - int(tgt["occupied"])) <= 4),
        "legal_slack_ok": int(abs(len(lw) - len(lt)) <= 4),
        "prefix_slack_ok": int(abs(len(wrong["tokens"]) - len(tgt["tokens"])) <= 4),
    }


def crossover_scores(y_t_ht, y_t_hw, y_w_hw, y_w_ht) -> dict:
    a_t = float(y_t_ht - y_t_hw)
    a_w = float(y_w_hw - y_w_ht)
    return {"A_T": a_t, "A_W": a_w, "C": a_t + a_w, "both_positive": int(a_t > 0 and a_w > 0)}


def increment_ok(boot: dict, floor: float = BEHAVIOR_FLOOR) -> int:
    return int(boot.get("point", 0) >= float(floor) and boot.get("ci95_low", 0) > 0)


def decide_semantics_verdict(gate: dict) -> str:
    needed = ("reconstruction_ok", "split_ok", "integrity_ok", "invariance_ok", "js_match_ok", "coverage_ok")
    if any(gate.get(k) is None for k in needed):
        return VERDICT_INCONCLUSIVE
    if not gate.get("integrity_ok") or not gate.get("reconstruction_ok") or not gate.get("split_ok"):
        return VERDICT_ENGINEERING
    if not gate.get("coverage_ok") or not gate.get("js_match_ok") or not gate.get("invariance_ok"):
        return VERDICT_INCONCLUSIVE
    if gate.get("board_increment_ok"):
        if gate.get("legal_increment_ok"):
            return VERDICT_MIXED
        return VERDICT_BOARD
    if gate.get("legal_matches_full"):
        return VERDICT_LEGAL
    return VERDICT_MIXED
