"""08R2-SYMMETRIC-TARGET: three-anchor two-target switching test.

Does not overwrite NST/SEM/SPEC/COST. Official letters unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r2 import pair_manifest_hash, require_no_confirmation
from src.othello_r2_fiber_spec import _kind
from src.othello_r2_null_state import prior_eval_games
from src.othello_r2_state_semantics import ABLATIONS, crossover_scores, paired_boot
from src.othello_runutil import add_labels, board_arr
from src.toy_othello import sha256_json


SYM_DIRNAME = "symmetric_target"
NST_PAIR_HASH = "39abdc79090211ab4a69e3e5f79f14319231705640f231dcc146233a9f5092e6"
SPEC_HASH = "2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3"
TRIPLET_SEED = 7182
WANT_A = 16
WANT_B = 16
LEGAL_COUNT_SLACK = 4
FIXED_JS = 0.01
FIXED_JS_TOL = 0.0002
AT_FLOOR = 0.005
AW_FLOOR = 0.005
BOTH_POS_MIN = 0.60
COVERAGE_MIN = 29
N_BOOT = 5000
BOOT_SEED = 82
DIR_ATOL = 1e-4

STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
STATUS_STAGE0_FAIL = "SYMMETRIC_TARGET_STAGE0_FAIL"
STATUS_INFEASIBLE = "SYMMETRIC_TARGET_CENSUS_INFEASIBLE"
VERDICT_SUPPORTED = "SYMMETRIC_TARGET_SWITCHING_SUPPORTED"
VERDICT_ANCHOR = "ANCHOR_DEPENDENT_TARGET_SWITCHING"
VERDICT_NOT = "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED"
VERDICT_INCONCLUSIVE = "SYMMETRIC_TARGET_SWITCHING_INCONCLUSIVE"
VERDICT_ENGINEERING = "SYMMETRIC_TARGET_ENGINEERING_FAILED"

BANNED = {
    "null_state_transfer",
    "state_semantics",
    "fiber_spec",
    "fiber_cost",
    "fiber_cost_forensic",
    "fiber_cost_fixed_js",
}
ANCHORS = ("N", "T", "W")
PRIMARY_MODELS = ("FULL",)
SECONDARY_MODELS = ("LEGAL_ONLY", "BOARD_ONLY", "STAGE_ONLY")


def isolate_sym_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == SYM_DIRNAME:
        out = requested
    elif requested.name == "08r2":
        out = requested / SYM_DIRNAME
    else:
        out = requested / "results" / "08r2" / SYM_DIRNAME
    if any(p in BANNED for p in out.parts):
        raise ValueError(f"symmetric-target must not write under preserved dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out)}


def nst_eval_games(val_rows: list[dict], holdout_ids: set[str]) -> dict:
    from src.othello_r2_fiber_spec import enumerate_candidates, match_wrong_donors, select_disjoint_pairs

    prior = prior_eval_games(val_rows, holdout_ids)
    enum = enumerate_candidates(val_rows, holdout_ids, prior["games"])
    selected = select_disjoint_pairs(enum["found"], 32, 16, 4182)
    games = set(prior["games"])
    for p in selected["pairs"]:
        games.add(str(p["recip_game"]))
        games.add(str(p["target_game"]))
    used = {p["recip_game"] for p in selected["pairs"]} | {p["target_game"] for p in selected["pairs"]}
    wrongs = [match_wrong_donors(enum["labeled"], p, holdout_ids | prior["games"] | used, 1, 4282) for p in selected["pairs"]]
    for w in wrongs:
        for d in w.get("donors") or []:
            games.add(str(d["game_id"]))
    return {"games": games, "n_total": len(games), "nst_hash": selected["pair_manifest_sha256"], "prior": prior}


def triplet_hash(triplets: list[dict]) -> str:
    payload = [
        {
            "recip": t.get("recip"),
            "target_t": t.get("target_t"),
            "target_w": t.get("target_w"),
            "r2_type": t.get("r2_type"),
            "recip_game": t.get("recip_game"),
            "target_t_game": t.get("target_t_game"),
            "target_w_game": t.get("target_w_game"),
            "hamming_t": t.get("hamming_t"),
            "hamming_w": t.get("hamming_w"),
        }
        for t in triplets
    ]
    return sha256_json(payload)


def _legal_differs(a: dict, b: dict) -> bool:
    la, lb = set(a["legal"]), set(b["legal"])
    return bool(la - lb) and bool(lb - la)


def _vs_recip_ok(rec: dict, tgt: dict) -> bool:
    lr, lt = set(rec["legal"]), set(tgt["legal"])
    return bool(lt - lr) and bool(lr - lt)


def enumerate_triplets(val_rows: list[dict], holdout_ids: set[str], exclude_games: set[str]) -> dict:
    require_no_confirmation(val_rows, holdout_ids)
    labeled = add_labels(list(val_rows))
    elig = [
        i
        for i, r in enumerate(labeled)
        if r.get("legal") and not r.get("terminal") and str(r["game_id"]) not in holdout_ids and str(r["game_id"]) not in exclude_games
    ]
    found = {"A_legal_diff": [], "B_similar_board_legal_change": []}
    for i in elig:
        rec = labeled[i]
        cands = {"A_legal_diff": [], "B_similar_board_legal_change": []}
        for j in elig:
            if i == j:
                continue
            tgt = labeled[j]
            if rec["game_id"] == tgt["game_id"]:
                continue
            if int(rec["player"]) != int(tgt["player"]):
                continue
            if not _vs_recip_ok(rec, tgt):
                continue
            kind, ham = _kind(rec, tgt)
            if kind not in cands:
                continue
            cands[kind].append((j, ham, int(tgt["occupied"]), len(tgt["legal"])))
        for kind, pool in cands.items():
            by_occ: dict[int, list] = {}
            for item in pool:
                by_occ.setdefault(item[2], []).append(item)
            for occ, group in by_occ.items():
                if len(group) < 2:
                    continue
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ja, ham_t, _, nt = group[a]
                        jb, ham_w, _, nw = group[b]
                        ta, tb = labeled[ja], labeled[jb]
                        if ta["game_id"] == tb["game_id"]:
                            continue
                        if abs(nt - nw) > LEGAL_COUNT_SLACK:
                            continue
                        if not _legal_differs(ta, tb):
                            continue
                        found[kind].append(
                            {
                                "recip": i,
                                "target_t": ja,
                                "target_w": jb,
                                "r2_type": kind,
                                "recip_game": rec["game_id"],
                                "target_t_game": ta["game_id"],
                                "target_w_game": tb["game_id"],
                                "player": int(rec["player"]),
                                "occupied_recip": int(rec["occupied"]),
                                "occupied_t": int(ta["occupied"]),
                                "occupied_w": int(tb["occupied"]),
                                "n_legal_t": int(len(ta["legal"])),
                                "n_legal_w": int(len(tb["legal"])),
                                "hamming_t": int(ham_t),
                                "hamming_w": int(ham_w),
                                "used_confirmation": 0,
                            }
                        )
    return {"labeled": labeled, "found": found, "n_eligible": len(elig)}


def select_triplets(found: dict, n_a: int, n_b: int, seed: int, n_trials: int = 400) -> dict:
    a = list(found["A_legal_diff"])
    b = list(found["B_similar_board_legal_change"])
    rng = np.random.default_rng(int(seed))
    best = {"triplets": [], "n_a": 0, "n_b": 0}

    def take(pool, limit, used):
        out = []
        for e in pool:
            if len(out) >= limit:
                break
            games = {e["recip_game"], e["target_t_game"], e["target_w_game"]}
            if len(games) < 3 or games & used:
                continue
            out.append(e)
            used |= games
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
            best = {"triplets": got_a + got_b, "n_a": len(got_a), "n_b": len(got_b)}
        if len(got_a) == n_a and len(got_b) == n_b:
            best["feasible"] = True
            return finalize_triplets(best, seed)
    best["feasible"] = False
    return finalize_triplets(best, seed)


def finalize_triplets(best: dict, seed: int) -> dict:
    trips = best["triplets"]
    recip = {t["recip_game"] for t in trips}
    targets = {t["target_t_game"] for t in trips} | {t["target_w_game"] for t in trips}
    return {
        "triplets": trips,
        "n_a": best["n_a"],
        "n_b": best["n_b"],
        "feasible": bool(best.get("feasible")),
        "n_recipients": len(recip),
        "n_targets": len(targets),
        "recip_target_overlap": sorted(recip & targets),
        "triplet_sha256": triplet_hash(trips),
        "seed": int(seed),
    }


def stage_only_identical(b_t: np.ndarray, b_w: np.ndarray) -> bool:
    return bool(np.allclose(np.asarray(b_t).reshape(-1)[256:258], np.asarray(b_w).reshape(-1)[256:258]))


def arithmetic_levels(y_full, y_stage, y_editor) -> dict:
    full_e = float(y_full - y_editor)
    stage_e = float(y_stage - y_editor)
    full_s = float(y_full - y_stage)
    share = float(stage_e / full_e) if full_e != 0 else float("nan")
    return {
        "FULL_minus_EditorA": full_e,
        "STAGE_ONLY_minus_EditorA": stage_e,
        "FULL_minus_STAGE_ONLY": full_s,
        "stage_share_of_full_editor": share,
        "note": "Arithmetic of mean response levels, not independent causal mechanisms.",
    }


def decide_sym_verdict(gate: dict) -> str:
    needed = ("split_ok", "integrity_ok", "invariance_ok", "js_match_ok", "coverage_ok", "reconstruction_ok")
    if any(gate.get(k) is None for k in needed):
        return VERDICT_INCONCLUSIVE
    if not gate.get("integrity_ok") or not gate.get("split_ok") or not gate.get("reconstruction_ok"):
        return VERDICT_ENGINEERING
    if not gate.get("coverage_ok") or not gate.get("js_match_ok") or not gate.get("invariance_ok"):
        return VERDICT_INCONCLUSIVE
    if gate.get("natural_symmetric_ok"):
        return VERDICT_SUPPORTED
    if gate.get("anchor_dependence_ok"):
        return VERDICT_ANCHOR
    return VERDICT_NOT


def natural_symmetric_ok(a_t: dict, a_w: dict, c: dict, both_frac: float, floor: float = AT_FLOOR, both_min: float = BOTH_POS_MIN) -> int:
    return int(
        a_t.get("point", 0) >= float(floor)
        and a_w.get("point", 0) >= float(floor)
        and a_t.get("ci95_low", 0) > 0
        and a_w.get("ci95_low", 0) > 0
        and c.get("point", 0) > 0
        and both_frac >= float(both_min)
    )


def anchor_dependence_ok(nat: dict, at: dict, aw: dict) -> int:
    """T-anchor favors T more than W; W-anchor favors W more than T; natural is weaker on at least one arm."""
    try:
        t_bias = at["A_T"]["point"] - at["A_W"]["point"]
        w_bias = aw["A_W"]["point"] - aw["A_T"]["point"]
        nat_weak = (nat["A_W"]["ci95_low"] <= 0) or (nat["A_T"]["ci95_low"] <= 0)
        return int(t_bias > 0 and w_bias > 0 and nat_weak)
    except (KeyError, TypeError):
        return 0
