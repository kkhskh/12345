"""08R2-NULL-STATE-TRANSFER: donor-free probe-null state candidate.

Does not overwrite SPEC/COST/fixed-JS. Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.othello_r2_fiber import softmax
from src.othello_r2_fiber_spec import y_components
from src.toy_othello import legal_squares, square_to_token


NST_DIRNAME = "null_state_transfer"
SITE = "blocks.6.hook_resid_mid"
KS = (8, 16, 32, 64)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
PAIR_SEED = 4182
WRONG_SEED = 4282
RANDOM_SEED = 5182
CV_SEED = 6182
N_FOLDS = 5
WANT_A = 32
WANT_B = 16
FIXED_JS = 0.01
FIXED_JS_TOL = 0.0002
BEHAVIOR_FLOOR = 0.01
COVERAGE_MIN = 44
N_BOOT = 2000
BOOT_SEED = 82
Y_RECON_ATOL = 1e-6
FIXED_JS_PAIR_HASH = "2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3"

VERDICT_SUPPORTED = "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED"
VERDICT_NOT_SUPPORTED = "DONOR_FREE_NULL_STATE_TRANSFER_NOT_SUPPORTED"
VERDICT_INCONCLUSIVE = "DONOR_FREE_NULL_STATE_TRANSFER_INCONCLUSIVE"
VERDICT_ENGINEERING = "NULL_STATE_TRANSFER_ENGINEERING_FAILED"
STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
STATUS_STAGE0_FAIL = "STAGE0_SCORING_INCONSISTENT"
STATUS_INFEASIBLE = "NULL_STATE_CENSUS_INFEASIBLE"
STATUS_COMPLETE = "NULL_STATE_TRANSFER_COMPLETE"

PRIMARY_CELLS = ("learned", "wrong_target", "mean_state", "random")
SECONDARY_CELLS = ("oracle", "gradient")
FEATURE_NAMES = (
    [f"rel_{i}_{c}" for i in range(64) for c in ("empty", "mine", "theirs")]
    + [f"legal_{i}" for i in range(64)]
    + ["player_is_black", "stage_occ64"]
)


def isolate_nst_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == NST_DIRNAME:
        out = requested
    elif requested.name == "08r2":
        out = requested / NST_DIRNAME
    else:
        out = requested / "results" / "08r2" / NST_DIRNAME
    banned = {"fiber_spec", "fiber_cost", "fiber_cost_forensic", "fiber_cost_fixed_js"}
    if any(p in banned for p in out.parts):
        raise ValueError(f"NST must not write under preserved experiment dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out)}


def stage_mean_name(key: tuple[int, int]) -> str:
    return f"{int(key[0])}_{int(key[1])}"


def parse_stage_mean_name(name: str) -> tuple[int, int]:
    a, b = str(name).split("_")
    return int(a), int(b)


def legal_record(row: dict) -> dict:
    from src.toy_othello import square_to_token

    v = verify_simulator_legal(row)
    legal = [int(s) for s in (row.get("legal") or [])]
    board = np.asarray(row.get("board_np", np.asarray(row.get("board", [])).reshape(-1))).reshape(-1)
    rel = np.asarray(row.get("rel_lab", [])).reshape(-1)
    return {
        "game_id": row.get("game_id"),
        "player": int(row["player"]),
        "occupied": int(row.get("occupied", 0)),
        "board": [int(x) for x in board],
        "rel_lab": [int(x) for x in rel],
        "legal_squares": legal,
        "legal_tokens": [square_to_token(s) for s in legal if square_to_token(s) is not None],
        "sim_ok": v["ok"],
        "token_ok": v["token_ok"],
    }


def donor_separation_audit() -> dict:
    import inspect

    params = tuple(inspect.signature(learned_null_delta).parameters)
    return {
        "signature": params,
        "allowed": learned_uses_only_allowed_inputs(),
        "has_donor_argument": int("h_d" in params or "h_D" in params),
        "uses_donor_activation": 0,
        "ok": int(params == learned_uses_only_allowed_inputs()),
    }


def symbolic_state(row: dict) -> np.ndarray:
    rel = np.asarray(row["rel_lab"], dtype=np.int64).reshape(-1)
    if rel.size != 64:
        raise ValueError("rel_lab must be 64 squares")
    board_oh = np.eye(3, dtype=np.float64)[np.clip(rel, 0, 2)].reshape(-1)
    legal = np.zeros(64, dtype=np.float64)
    for s in row.get("legal") or []:
        if 0 <= int(s) < 64:
            legal[int(s)] = 1.0
    player = np.array([1.0 if int(row["player"]) == 1 else 0.0], dtype=np.float64)
    occ = float(row.get("occupied", int(np.count_nonzero(np.asarray(row.get("board", [])).reshape(-1)))))
    stage = np.array([occ / 64.0], dtype=np.float64)
    out = np.concatenate([board_oh, legal, player, stage])
    if out.size != len(FEATURE_NAMES):
        raise ValueError(f"feature size {out.size} != {len(FEATURE_NAMES)}")
    return out


def stage_key(row: dict) -> tuple[int, int]:
    return int(row["player"]), int(row["occupied"]) // 4


def verify_simulator_legal(row: dict) -> dict:
    board = np.asarray(row["board_np"] if "board_np" in row else np.asarray(row["board"]).reshape(8, 8))
    got = set(legal_squares(board, int(row["player"])))
    stored = set(int(s) for s in (row.get("legal") or []))
    toks = [square_to_token(s) for s in sorted(stored) if square_to_token(s) is not None]
    return {
        "game_id": row.get("game_id"),
        "ok": int(got == stored),
        "n_sim": len(got),
        "n_stored": len(stored),
        "token_ok": int(all(t is not None and t >= 0 for t in toks)),
        "became_tokens_defined": 1,
    }


def rescore_y_from_logits(logits: np.ndarray, rec_legal, tgt_legal, probs=None) -> dict:
    if probs is None:
        probs = softmax(np.asarray(logits, dtype=np.float64))
    return y_components(np.asarray(probs, dtype=np.float64), rec_legal, tgt_legal)


def saved_y_present(saved_row: dict) -> bool:
    return all(saved_row.get(k) not in (None, "") for k in ("Y", "G", "Q"))


def audit_saved_y(saved_row: dict, recomputed: dict, atol: float = Y_RECON_ATOL) -> dict:
    if not saved_y_present(saved_row):
        return {"ok": None, "mismatches": [], "compared": 0, "missing_saved": [k for k in ("Y", "G", "Q") if saved_row.get(k) in (None, "")]}
    out = {"ok": 1, "mismatches": [], "compared": 1, "abs_Y": 0.0, "abs_G": 0.0, "abs_Q": 0.0}
    for key in ("Y", "G", "Q"):
        a, b = saved_row.get(key), recomputed.get(key)
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            out["ok"] = 0
            out["mismatches"].append(key)
            continue
        err = abs(fa - fb) if fa == fa and fb == fb else float("nan")
        out[f"abs_{key}"] = err
        if err != err or err > float(atol):
            if not (fa != fa and fb != fb):
                out["ok"] = 0
                out["mismatches"].append(key)
    return out


def resolve_pair_rows(val_rows: list[dict], pair: dict) -> dict:
    rec = val_rows[int(pair["recip"])]
    tgt = val_rows[int(pair["donor"])]
    if rec["game_id"] != pair["recip_game"] or tgt["game_id"] != pair["target_game"]:
        raise ValueError(f"pair index/game mismatch {pair['recip_game']}|{pair['target_game']}")
    last_rec = None
    last_tgt = None
    for row in val_rows:
        if row["game_id"] == pair["recip_game"]:
            last_rec = row
        if row["game_id"] == pair["target_game"]:
            last_tgt = row
    return {
        "rec": rec,
        "tgt": tgt,
        "last_prefix_same_as_pair": int(last_rec is rec and last_tgt is tgt),
    }


def fit_pca(Z: np.ndarray, k: int) -> dict:
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    X = Z - mu
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = int(min(k, Vt.shape[0], Z.shape[0] - 1))
    V = Vt[:k].T
    ev = (S**2)
    evr = float(ev[:k].sum() / ev.sum()) if ev.sum() else 0.0
    return {"mu": mu, "V": V, "explained_variance_ratio": evr, "singular_values": S[:k], "k": k}


def pca_coords(Z: np.ndarray, pca: dict) -> np.ndarray:
    return (np.asarray(Z, dtype=np.float64) - pca["mu"]) @ pca["V"]


def ridge_fit(B: np.ndarray, A: np.ndarray, alpha: float) -> dict:
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B1 = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
    xtx = B1.T @ B1
    d = xtx.shape[0]
    xtx = xtx + float(alpha) * np.eye(d)
    xtx[-1, -1] -= float(alpha)
    w1 = np.linalg.solve(xtx, B1.T @ A)
    return {"W": w1[:-1], "c": w1[-1], "alpha": float(alpha)}


def predict_z(B: np.ndarray, pca: dict, ridge: dict) -> np.ndarray:
    Ahat = np.asarray(B, dtype=np.float64) @ ridge["W"] + ridge["c"]
    return pca["mu"] + Ahat @ pca["V"].T


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    num = float(np.sum((y_true - y_pred) ** 2))
    den = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - num / den if den > 0 else float("nan")


def grouped_folds(games: list[str], n_folds: int = N_FOLDS, seed: int = CV_SEED) -> list[np.ndarray]:
    uniq = np.array(sorted(set(games)))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)
    return np.array_split(uniq, int(n_folds))


def select_ridge_pca(Z: np.ndarray, B: np.ndarray, games: list[str], ks=KS, alphas=ALPHAS, n_folds: int = N_FOLDS, seed: int = CV_SEED) -> dict:
    Z = np.asarray(Z, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    games = [str(g) for g in games]
    folds = grouped_folds(games, n_folds, seed)
    rows = []
    best = None
    for k in ks:
        for alpha in alphas:
            scores = []
            for held in folds:
                held = set(map(str, held))
                tr = np.array([g not in held for g in games])
                te = ~tr
                if tr.sum() < 8 or te.sum() < 4:
                    continue
                pca = fit_pca(Z[tr], int(k))
                coords = pca_coords(Z[tr], pca)
                ridge = ridge_fit(B[tr], coords, float(alpha))
                pred = predict_z(B[te], pca, ridge)
                scores.append(r2_score(Z[te], pred))
            mean_r2 = float(np.nanmean(scores)) if scores else float("nan")
            rec = {"k": int(k), "alpha": float(alpha), "cv_r2": mean_r2, "n_scores": len(scores)}
            rows.append(rec)
            if best is None or best["cv_r2"] != best["cv_r2"] or (mean_r2 == mean_r2 and mean_r2 > best["cv_r2"]):
                best = rec
    if best is None:
        raise ValueError("CV selection produced no valid fold")
    pca = fit_pca(Z, int(best["k"]))
    ridge = ridge_fit(B, pca_coords(Z, pca), float(best["alpha"]))
    return {"best": best, "table": rows, "pca": pca, "ridge": ridge, "train_r2": r2_score(Z, predict_z(B, pca, ridge))}


def pack_model(pca: dict, ridge: dict, P_N: np.ndarray, stage_means: dict) -> dict:
    return {
        "mu": pca["mu"],
        "V": pca["V"],
        "W": ridge["W"],
        "c": ridge["c"],
        "k": int(pca["k"]),
        "alpha": float(ridge["alpha"]),
        "P_N": np.asarray(P_N, dtype=np.float64),
        "stage_means": {stage_mean_name(k): v for k, v in stage_means.items()},
        "feature_names": list(FEATURE_NAMES),
        "uses_donor_activation": 0,
    }


def learned_null_delta(h_e: np.ndarray, b_t: np.ndarray, model: dict) -> np.ndarray:
    """Donor-free. Signature forbids h_D."""
    h_e = np.asarray(h_e, dtype=np.float64).reshape(-1)
    b_t = np.asarray(b_t, dtype=np.float64).reshape(-1)
    z_e = model["P_N"] @ h_e
    zhat = predict_z(b_t.reshape(1, -1), {"mu": model["mu"], "V": model["V"]}, {"W": model["W"], "c": model["c"]})[0]
    zhat = model["P_N"] @ zhat
    return zhat - z_e


def oracle_null_delta(h_e: np.ndarray, h_d: np.ndarray, P_N: np.ndarray) -> np.ndarray:
    return np.asarray(P_N, dtype=np.float64) @ (np.asarray(h_d, dtype=np.float64) - np.asarray(h_e, dtype=np.float64))


def mean_state_delta(h_e: np.ndarray, key: tuple[int, int], model: dict) -> np.ndarray | None:
    name = stage_mean_name(key)
    if name not in model["stage_means"]:
        return None
    z_e = model["P_N"] @ np.asarray(h_e, dtype=np.float64).reshape(-1)
    mu = model["P_N"] @ np.asarray(model["stage_means"][name], dtype=np.float64).reshape(-1)
    return mu - z_e


def learned_uses_only_allowed_inputs() -> tuple:
    return ("h_e", "b_t", "model")


def decide_nst_verdict(gate: dict) -> str:
    needed = (
        "split_ok",
        "no_donor_leak",
        "invariance_ok",
        "js_match_ok",
        "coverage_ok",
        "mean_vs_wrong_ge_floor",
        "ci_wrong_gt_0",
        "beats_random",
        "beats_mean_state",
        "beats_editor",
        "components_vs_editor_ok",
        "integrity_ok",
    )
    if any(gate.get(k) is None for k in needed):
        return VERDICT_INCONCLUSIVE
    if not gate.get("integrity_ok") or not gate.get("split_ok") or not gate.get("no_donor_leak"):
        return VERDICT_ENGINEERING
    if not gate.get("coverage_ok") or not gate.get("js_match_ok") or not gate.get("invariance_ok"):
        return VERDICT_INCONCLUSIVE
    if all(bool(gate[k]) for k in needed):
        return VERDICT_SUPPORTED
    return VERDICT_NOT_SUPPORTED


def split_leakage_audit(train_games, select_games, recip_games, target_games, wrong_games, prior_games, holdout_ids) -> dict:
    train_games = set(map(str, train_games))
    select_games = set(map(str, select_games))
    recip_games = set(map(str, recip_games))
    target_games = set(map(str, target_games))
    wrong_games = set(map(str, wrong_games))
    prior_games = set(map(str, prior_games))
    holdout_ids = set(map(str, holdout_ids))
    eval_games = recip_games | target_games
    leaks = {
        "train_vs_eval": sorted(train_games & eval_games),
        "select_vs_eval": sorted(select_games & eval_games),
        "recip_vs_target": sorted(recip_games & target_games),
        "eval_vs_prior": sorted(eval_games & prior_games),
        "eval_vs_holdout": sorted(eval_games & holdout_ids),
        "wrong_vs_holdout": sorted(wrong_games & holdout_ids),
        "wrong_vs_prior": sorted(wrong_games & prior_games),
    }
    return {
        "ok": int(all(len(v) == 0 for v in leaks.values())),
        "n_train": len(train_games),
        "n_select": len(select_games),
        "n_recip": len(recip_games),
        "n_target": len(target_games),
        "n_wrong": len(wrong_games),
        "n_prior": len(prior_games),
        "leaks": leaks,
    }


def prior_eval_games(val_rows: list[dict], holdout_ids: set[str]) -> dict:
    from src.othello_r2_fiber_cost import reconstruct_frozen_pairs
    from src.othello_r2_fiber_spec import fiber_excluded_games

    fiber = fiber_excluded_games(val_rows, holdout_ids)
    spec = reconstruct_frozen_pairs(val_rows, holdout_ids, "2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3")
    games = set(map(str, fiber))
    for p in spec["pairs"]:
        games.add(str(p["recip_game"]))
        games.add(str(p["target_game"]))
    for w in spec["wrong_donors"]:
        for d in w.get("donors") or []:
            games.add(str(d["game_id"]))
    return {
        "games": games,
        "n_fiber": len(fiber),
        "n_spec_pairs": len(spec["pairs"]),
        "n_total": len(games),
        "spec_hash": spec["pair_manifest_sha256"],
    }
