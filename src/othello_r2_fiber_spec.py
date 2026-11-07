"""08R2-FIBER-SPEC: L2-matched target-specificity of probe-null residuals.

No new editor. No confirmation. Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r1 import cluster_bootstrap, kl_row, legality_sets, total_legal_mass
from src.othello_r2 import pair_manifest_hash, r2_pair_categories, require_no_confirmation
from src.othello_r2_fiber import js_divergence, softmax
from src.othello_runutil import add_labels, board_arr
from src.toy_othello import board_hamming, square_to_token


SPEC_DIRNAME = "fiber_spec"
PRIMARY = {"A_legal_diff", "B_similar_board_legal_change"}
WANT_A = 32
WANT_B = 16
PAIR_SEED = 2182
WRONG_SEED = 2282
RANDOM_SEEDS = (3182, 3282, 3382)
N_WRONG = 3
N_RANDOM = 3
OCC_SLACK = 4
LEGAL_SIZE_SLACK = 4
PREFIX_SLACK = 4
L2_RTOL = 1e-6
L2_MIN_NORM = 1e-8
PROBE_PROBA_ATOL = 1e-5
BEHAVIOR_FLOOR = 0.05
N_BOOT = 2000
BOOT_SEED = 82
RANK_TOL = 1e-8
FIBER_PAIR_SEED = 1082

REQUIRED_SPEC_FILES = (
    "fiber_spec_preregistration.json",
    "fiber_spec_pair_manifest.json",
    "fiber_spec_pair_census.csv",
    "fiber_spec_dependency_manifest.json",
    "fiber_spec_probe_parity.csv",
    "fiber_spec_interventions.csv",
    "fiber_spec_full_logits.npz",
    "fiber_spec_full_probe_probabilities.npz",
    "fiber_spec_nullspace_vectors.npz",
    "fiber_spec_target_legal_sets.json",
    "fiber_spec_attempt_accounting.json",
    "fiber_spec_bootstrap.csv",
    "fiber_spec_completion_status.json",
    "fiber_spec_scientific_report.md",
)

VERDICT_SUPPORTED = "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED"
VERDICT_NOT_SUPPORTED = "TARGET_SPECIFIC_NULL_EFFECT_NOT_SUPPORTED"
VERDICT_INCONCLUSIVE = "TARGET_SPECIFIC_NULL_EFFECT_INCONCLUSIVE"
VERDICT_ENGINEERING = "FIBER_SPEC_ENGINEERING_FAILED"
STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
STATUS_INFEASIBLE = "FIBER_SPEC_CENSUS_INFEASIBLE"
STATUS_COMPLETE = "FIBER_SPEC_DIAGNOSTIC_COMPLETE"


def isolate_spec_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == SPEC_DIRNAME and requested.parent.name == "08r2":
        out = requested
    elif requested.name == "08r2":
        out = requested / SPEC_DIRNAME
    else:
        out = requested / "08r2" / SPEC_DIRNAME
    return {"requested": str(requested), "results_dir": str(out), "reason": "08R2-FIBER-SPEC writes only under 08r2/fiber_spec/"}


def fiber_excluded_games(val_rows: list[dict], holdout_ids: set[str]) -> set[str]:
    pairs = r2_pair_categories(val_rows, 32, FIBER_PAIR_SEED, holdout_ids, n_a=24, n_b=8, include_c=False)
    games = set()
    for p in pairs:
        if p.get("r2_type") in PRIMARY:
            games.add(str(p["recip_game"]))
            games.add(str(p["target_game"]))
    return games


def _kind(rec: dict, don: dict) -> tuple[str | None, int]:
    if rec.get("terminal") or don.get("terminal"):
        return None, -1
    if int(rec["player"]) != int(don["player"]):
        return None, -1
    if rec["game_id"] == don["game_id"]:
        return None, -1
    lr, ld = set(rec["legal"]), set(don["legal"])
    if not (ld - lr) or not (lr - ld):
        return None, -1
    ham = int(board_hamming(board_arr(rec), board_arr(don)))
    if 2 <= ham <= 8:
        return "B_similar_board_legal_change", ham
    if ham > 8:
        return "A_legal_diff", ham
    return None, ham


def enumerate_candidates(val_rows: list[dict], holdout_ids: set[str], exclude_games: set[str]) -> dict:
    require_no_confirmation(val_rows, holdout_ids)
    labeled = add_labels(list(val_rows))
    elig = [i for i, r in enumerate(labeled) if r.get("legal") and not r.get("terminal")]
    buckets: dict[tuple, list[int]] = {}
    for i in elig:
        r = labeled[i]
        buckets.setdefault((int(r["player"]), int(r["occupied"]) // 4), []).append(i)
    rej = {
        "holdout": 0,
        "fiber_excluded": 0,
        "same_game": 0,
        "player_or_legal": 0,
        "hamming_1_or_0": 0,
        "empty_became_or_left": 0,
        "cross_bucket": 0,
    }
    found = {"A_legal_diff": [], "B_similar_board_legal_change": []}
    for key, idxs in buckets.items():
        for i in idxs:
            rec = labeled[i]
            if rec["game_id"] in holdout_ids:
                rej["holdout"] += 1
                continue
            if rec["game_id"] in exclude_games:
                rej["fiber_excluded"] += 1
                continue
            for j in idxs:
                if i == j:
                    continue
                don = labeled[j]
                if don["game_id"] in holdout_ids:
                    continue
                if don["game_id"] in exclude_games:
                    continue
                if rec["game_id"] == don["game_id"]:
                    rej["same_game"] += 1
                    continue
                kind, ham = _kind(rec, don)
                if kind is None:
                    if int(rec["player"]) != int(don["player"]):
                        rej["player_or_legal"] += 1
                    elif not (set(don["legal"]) - set(rec["legal"])) or not (set(rec["legal"]) - set(don["legal"])):
                        rej["empty_became_or_left"] += 1
                    else:
                        rej["hamming_1_or_0"] += 1
                    continue
                lr, ld = set(rec["legal"]), set(don["legal"])
                found[kind].append(
                    {
                        "recip": i,
                        "donor": j,
                        "r2_type": kind,
                        "hamming": ham,
                        "recip_game": rec["game_id"],
                        "target_game": don["game_id"],
                        "player": int(rec["player"]),
                        "occupied_recip": int(rec["occupied"]),
                        "occupied_donor": int(don["occupied"]),
                        "n_legal_recip": int(len(rec["legal"])),
                        "n_legal_donor": int(len(don["legal"])),
                        "n_became_legal": int(len(ld - lr)),
                        "n_became_illegal": int(len(lr - ld)),
                        "prefix_len_recip": int(len(rec["tokens"])),
                        "prefix_len_donor": int(len(don["tokens"])),
                        "same_player": 1,
                        "used_confirmation": 0,
                    }
                )
    return {"labeled": labeled, "found": found, "rejections": rej, "n_eligible_rows": len(elig)}


def _unique_games(edges: list[dict], field: str) -> int:
    return len({e[field] for e in edges})


def select_disjoint_pairs(found: dict, n_a: int, n_b: int, seed: int, n_trials: int = 400) -> dict:
    a = list(found["A_legal_diff"])
    b = list(found["B_similar_board_legal_change"])
    rng = np.random.default_rng(int(seed))
    best = {"pairs": [], "n_a": 0, "n_b": 0}

    def take(pool, limit, used):
        out = []
        for e in pool:
            if len(out) >= limit:
                break
            rg, tg = e["recip_game"], e["target_game"]
            if rg in used or tg in used or rg == tg:
                continue
            out.append(e)
            used.add(rg)
            used.add(tg)
        return out

    for _ in range(int(n_trials)):
        used: set[str] = set()
        order_b = b.copy()
        order_a = a.copy()
        rng.shuffle(order_b)
        rng.shuffle(order_a)
        got_b = take(order_b, n_b, used)
        got_a = take(order_a, n_a, used)
        if len(got_b) > best["n_b"] or (len(got_b) == best["n_b"] and len(got_a) > best["n_a"]):
            best = {"pairs": got_a + got_b, "n_a": len(got_a), "n_b": len(got_b)}
        if len(got_a) == n_a and len(got_b) == n_b:
            best["feasible"] = True
            best["trial_found"] = True
            return finalize_selection(best, seed)

    best["feasible"] = False
    return finalize_selection(best, seed)


def finalize_selection(best: dict, seed: int) -> dict:
    pairs = best["pairs"]
    recip = {p["recip_game"] for p in pairs}
    tgt = {p["target_game"] for p in pairs}
    return {
        "pairs": pairs,
        "n_a": best["n_a"],
        "n_b": best["n_b"],
        "feasible": bool(best.get("feasible")),
        "n_recipient_games": len(recip),
        "n_target_games": len(tgt),
        "recip_target_overlap": sorted(recip & tgt),
        "n_games_used": len(recip | tgt),
        "pair_manifest_sha256": pair_manifest_hash(pairs),
        "seed": int(seed),
    }


def match_wrong_donors(labeled: list[dict], pair: dict, holdout_ids: set[str], k: int, seed: int) -> dict:
    rec = labeled[int(pair["recip"])]
    tgt = labeled[int(pair["donor"])]
    rng = np.random.default_rng(int(seed) + int(pair["recip"]))
    cands = []
    for i, row in enumerate(labeled):
        if i == pair["recip"] or i == pair["donor"]:
            continue
        if row["game_id"] in {rec["game_id"], tgt["game_id"]}:
            continue
        if row["game_id"] in holdout_ids:
            continue
        if row.get("terminal") or not row.get("legal"):
            continue
        if int(row["player"]) != int(rec["player"]):
            continue
        occ_d = abs(int(row["occupied"]) - int(tgt["occupied"]))
        legal_d = abs(len(row["legal"]) - len(tgt["legal"]))
        pref_d = abs(len(row["tokens"]) - len(tgt["tokens"]))
        if occ_d > OCC_SLACK or legal_d > LEGAL_SIZE_SLACK or pref_d > PREFIX_SLACK:
            continue
        cands.append((occ_d, legal_d, pref_d, int(row["game_id"].split("_")[-1]) if "_" in str(row["game_id"]) else i, i))
    cands.sort()
    # Deterministic shuffle of equally ranked prefixes, then pick distinct games.
    order = list(cands)
    rng.shuffle(order)
    order.sort(key=lambda t: (t[0], t[1], t[2]))
    picked = []
    seen = set()
    for item in order:
        i = item[-1]
        gid = labeled[i]["game_id"]
        if gid in seen:
            continue
        seen.add(gid)
        picked.append(
            {
                "index": i,
                "game_id": gid,
                "occupied": int(labeled[i]["occupied"]),
                "n_legal": int(len(labeled[i]["legal"])),
                "prefix_len": int(len(labeled[i]["tokens"])),
                "occ_diff_vs_correct": abs(int(labeled[i]["occupied"]) - int(tgt["occupied"])),
                "legal_size_diff_vs_correct": abs(len(labeled[i]["legal"]) - len(tgt["legal"])),
                "prefix_diff_vs_correct": abs(len(labeled[i]["tokens"]) - len(tgt["tokens"])),
            }
        )
        if len(picked) >= int(k):
            break
    return {
        "n_candidates": int(len({labeled[c[-1]]["game_id"] for c in cands})),
        "n_selected": int(len(picked)),
        "ok": int(len(picked) >= int(k)),
        "donors": picked,
    }


def match_l2(vec: np.ndarray, target_norm: float, min_norm: float = L2_MIN_NORM, rtol: float = L2_RTOL) -> dict:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm < float(min_norm) or float(target_norm) < float(min_norm):
        return {"ok": 0, "reason": "degenerate_direction", "vec": None, "l2": nrm, "target_l2": float(target_norm)}
    out = v * (float(target_norm) / nrm)
    got = float(np.linalg.norm(out))
    rel = abs(got - float(target_norm)) / max(float(target_norm), 1e-15)
    return {
        "ok": int(rel <= float(rtol)),
        "reason": "ok" if rel <= float(rtol) else "l2_mismatch",
        "vec": out,
        "l2": got,
        "target_l2": float(target_norm),
        "rel_err": rel,
    }


def y_components(probs: np.ndarray, rec_legal, tgt_legal) -> dict:
    parts = legality_sets(rec_legal, tgt_legal)
    g = total_legal_mass(probs, parts["became_legal"])
    q = total_legal_mass(probs, parts["became_illegal"])
    return {
        "G": float(g) if g == g else float("nan"),
        "Q": float(q) if q == q else float("nan"),
        "Y": float(g - q) if g == g and q == q else float("nan"),
        "became_legal_squares": parts["became_legal"],
        "became_illegal_squares": parts["became_illegal"],
        "became_legal_tokens": [square_to_token(s) for s in parts["became_legal"] if square_to_token(s) is not None],
        "became_illegal_tokens": [square_to_token(s) for s in parts["became_illegal"] if square_to_token(s) is not None],
        "stayed_legal_mass": float(total_legal_mass(probs, parts["stayed_legal"])),
        "stayed_illegal_mass": float(total_legal_mass(probs, parts["stayed_illegal"])),
    }


def pair_contrasts(y_correct: float, y_wrongs: list[float], y_rands: list[float]) -> dict:
    wmean = float(np.mean(y_wrongs)) if y_wrongs else float("nan")
    rmean = float(np.mean(y_rands)) if y_rands else float("nan")
    return {
        "delta_specific": float(y_correct - wmean) if y_correct == y_correct and wmean == wmean else float("nan"),
        "delta_random": float(y_correct - rmean) if y_correct == y_correct and rmean == rmean else float("nan"),
        "y_correct": float(y_correct),
        "y_wrong_mean": wmean,
        "y_random_mean": rmean,
        "n_wrong": int(len(y_wrongs)),
        "n_random": int(len(y_rands)),
    }


def decide_verdict(gate: dict) -> str:
    needed = (
        "invariance_ok",
        "l2_ok",
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


def evaluate_spec_completeness(out_dir: str | Path, *, real_model: bool = True) -> dict:
    out_dir = Path(out_dir)
    failures = []
    files = {name: (out_dir / name).is_file() for name in REQUIRED_SPEC_FILES}
    for name, exists in files.items():
        if not exists:
            failures.append(f"missing_file:{name}")
        elif (out_dir / name).stat().st_size == 0:
            failures.append(f"empty_file:{name}")
    status = {}
    if files.get("fiber_spec_completion_status.json"):
        status = json_load(out_dir / "fiber_spec_completion_status.json")
    if status.get("used_confirmation"):
        failures.append("confirmation_accessed")
    if real_model and status.get("status") == STATUS_COMPLETE:
        for key in ("checkpoint_loaded", "probe_loaded", "parity_ok", "pairs_ok", "controls_complete", "full_outputs_saved"):
            if status.get(key) != 1:
                failures.append(f"complete_missing_{key}")
    if "fiber_spec" not in Path(out_dir).parts:
        failures.append("spec_outputs_not_isolated")
    return {"ok": int(not failures), "failures": failures, "files": files}


def json_load(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def census_tables(enum: dict, selected: dict, wrong_reports: list[dict]) -> dict:
    found = enum["found"]
    return {
        "n_candidate_A": int(len(found["A_legal_diff"])),
        "n_candidate_B": int(len(found["B_similar_board_legal_change"])),
        "n_unique_recip_A": _unique_games(found["A_legal_diff"], "recip_game"),
        "n_unique_target_A": _unique_games(found["A_legal_diff"], "target_game"),
        "n_unique_recip_B": _unique_games(found["B_similar_board_legal_change"], "recip_game"),
        "n_unique_target_B": _unique_games(found["B_similar_board_legal_change"], "target_game"),
        "rejections": enum["rejections"],
        "selected_n_a": selected["n_a"],
        "selected_n_b": selected["n_b"],
        "feasible_32_16": int(selected["n_a"] >= WANT_A and selected["n_b"] >= WANT_B and not selected["recip_target_overlap"]),
        "n_wrong_ok": int(sum(r["ok"] for r in wrong_reports)),
        "n_wrong_fail": int(sum(not r["ok"] for r in wrong_reports)),
        "pair_manifest_sha256": selected["pair_manifest_sha256"],
    }


def slice_boot(values, clusters, n_boot: int, seed: int) -> dict:
    out = cluster_bootstrap(np.asarray(values, dtype=np.float64), list(clusters), n_boot, seed)
    return {k: out[k] for k in ("n", "n_clusters", "point", "boot_mean", "ci95_low", "ci95_high")}
