"""Read-only 08R2-SYMMETRIC-TARGET census audit.

Does not overwrite SYM/NST/SEM results. No model. No confirmation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r2_fiber_spec import _kind
from src.othello_r2_symmetric_target import (
    LEGAL_COUNT_SLACK,
    NST_PAIR_HASH,
    TRIPLET_SEED,
    WANT_A,
    WANT_B,
    enumerate_triplets,
    select_triplets,
)
from src.othello_runutil import add_labels, board_arr
from src.toy_othello import board_hamming, generate_game, legal_squares, occupancy, prefixes_from_game, subsample_prefixes


AUDIT_DIRNAME = "symmetric_target_census_audit"
FRESH_SEED = 8182
FRESH_N_GAMES = 400
FRESH_PREFIX_SEED = 8282
BANNED = {
    "null_state_transfer",
    "state_semantics",
    "fiber_spec",
    "fiber_cost",
    "fiber_cost_forensic",
    "fiber_cost_fixed_js",
    "symmetric_target",
}


def isolate_audit_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == AUDIT_DIRNAME:
        out = requested
    elif requested.name == "08r2":
        out = requested / AUDIT_DIRNAME
    else:
        out = requested / "results" / "08r2" / AUDIT_DIRNAME
    if any(p in BANNED for p in out.parts):
        raise ValueError(f"census audit must not write under preserved dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out)}


def eligible_indices(labeled: list[dict], holdout_ids: set[str], exclude_games: set[str]) -> list[int]:
    return [
        i
        for i, r in enumerate(labeled)
        if r.get("legal") and not r.get("terminal") and str(r["game_id"]) not in holdout_ids and str(r["game_id"]) not in exclude_games
    ]


def integrity_row(row: dict) -> dict:
    board = board_arr(row)
    occ_board = occupancy(board)
    occ_row = int(row.get("occupied", -1))
    sim_legal = set(legal_squares(board, int(row["player"])))
    rec_legal = set(int(s) for s in (row.get("legal") or []))
    return {
        "occ_match": int(occ_board == occ_row),
        "legal_match": int(sim_legal == rec_legal),
        "occ_board": occ_board,
        "occ_row": occ_row,
        "n_legal_sim": len(sim_legal),
        "n_legal_row": len(rec_legal),
    }


def max_triplet_packing(triplets: list[dict]) -> dict:
    """Exact 3-game set packing. Empty candidate set is a certificate of zero."""
    if not triplets:
        return {"n_triplets": 0, "max_packing": 0, "method": "empty_candidate_certificate", "upper_bound": 0, "selected": []}
    games = set()
    for t in triplets:
        games |= {t["recip_game"], t["target_t_game"], t["target_w_game"]}
    upper = min(len(triplets), len(games) // 3, len({t["recip_game"] for t in triplets}))
    edges = []
    for i, t in enumerate(triplets):
        g = (t["recip_game"], t["target_t_game"], t["target_w_game"])
        if len(set(g)) < 3:
            continue
        edges.append((i, set(g)))
    best: list[int] = []

    def rec(start: int, used: set[str], chosen: list[int]) -> None:
        nonlocal best
        remain = 0
        for j in range(start, len(edges)):
            if edges[j][1].isdisjoint(used):
                remain += 1
        if len(chosen) + remain <= len(best):
            return
        if len(chosen) > len(best):
            best = chosen.copy()
        for j in range(start, len(edges)):
            idx, gs = edges[j]
            if not gs.isdisjoint(used):
                continue
            chosen.append(idx)
            rec(j + 1, used | gs, chosen)
            chosen.pop()

    if len(edges) <= 80:
        rec(0, set(), [])
        method = "exact_backtrack"
        packed = len(best)
    else:
        used: set[str] = set()
        greedy: list[int] = []
        for idx, gs in edges:
            if gs.isdisjoint(used):
                greedy.append(idx)
                used |= gs
        packed = len(greedy)
        best = greedy
        method = "greedy_only_large_instance"
    return {
        "n_triplets": len(triplets),
        "n_usable_hyperedges": len(edges),
        "n_unique_games": len(games),
        "upper_bound": int(upper),
        "max_packing": int(packed),
        "method": method,
        "selected": [triplets[i] for i in best],
    }


def progressive_funnel(labeled: list[dict], elig: list[int], legal_slack: int = LEGAL_COUNT_SLACK, expand_a: bool = False) -> dict:
    n_elig = len(elig)
    pair = {
        "eligible_recipients": n_elig,
        "eligible_targets": n_elig,
        "distinct_game": 0,
        "same_player": 0,
        "ham_B_2_8": 0,
        "ham_A_gt_8": 0,
        "ham_0_or_1": 0,
        "bidir_legal_vs_recip": 0,
        "B_pairs": 0,
        "A_pairs": 0,
        "B_pairs_loose_pair_kind": 0,
    }
    b_targets: dict[int, list[tuple[int, int, int, int]]] = {}
    a_targets: dict[int, list[tuple[int, int, int, int]]] = {}
    for i in elig:
        rec = labeled[i]
        for j in elig:
            if i == j:
                continue
            tgt = labeled[j]
            if rec["game_id"] == tgt["game_id"]:
                continue
            pair["distinct_game"] += 1
            if int(rec["player"]) != int(tgt["player"]):
                continue
            pair["same_player"] += 1
            ham = int(board_hamming(board_arr(rec), board_arr(tgt)))
            if 2 <= ham <= 8:
                pair["ham_B_2_8"] += 1
            elif ham > 8:
                pair["ham_A_gt_8"] += 1
            else:
                pair["ham_0_or_1"] += 1
            lr, lt = set(rec["legal"]), set(tgt["legal"])
            bidir = bool(lt - lr) and bool(lr - lt)
            if bidir:
                pair["bidir_legal_vs_recip"] += 1
            if lr != lt and ham <= 8:
                pair["B_pairs_loose_pair_kind"] += 1
            kind, kind_ham = _kind(rec, tgt)
            if kind == "B_similar_board_legal_change":
                pair["B_pairs"] += 1
                b_targets.setdefault(i, []).append((j, kind_ham, int(tgt["occupied"]), len(tgt["legal"])))
            elif kind == "A_legal_diff":
                pair["A_pairs"] += 1
                a_targets.setdefault(i, []).append((j, kind_ham, int(tgt["occupied"]), len(tgt["legal"])))

    def triplets_from(by_recip: dict[int, list], kind: str, store: bool) -> dict:
        recips_ge2 = 0
        recips_eq_occ_ge2 = 0
        after_occ = 0
        after_legal_slack = 0
        after_distinct_tw = 0
        after_legal_differ = 0
        trips = []
        pack_games: set[str] = set()
        pack_recip: set[str] = set()
        for i, pool in by_recip.items():
            if len(pool) < 2:
                continue
            recips_ge2 += 1
            by_occ: dict[int, list] = {}
            for item in pool:
                by_occ.setdefault(item[2], []).append(item)
            if any(len(g) >= 2 for g in by_occ.values()):
                recips_eq_occ_ge2 += 1
            rec = labeled[i]
            for occ, group in by_occ.items():
                if len(group) < 2:
                    continue
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ja, ham_t, _, nt = group[a]
                        jb, ham_w, _, nw = group[b]
                        after_occ += 1
                        ta, tb = labeled[ja], labeled[jb]
                        if abs(nt - nw) <= legal_slack:
                            after_legal_slack += 1
                        else:
                            continue
                        if ta["game_id"] == tb["game_id"]:
                            continue
                        after_distinct_tw += 1
                        la, lb = set(ta["legal"]), set(tb["legal"])
                        if not (la - lb and lb - la):
                            continue
                        after_legal_differ += 1
                        pack_games |= {rec["game_id"], ta["game_id"], tb["game_id"]}
                        pack_recip.add(rec["game_id"])
                        if store:
                            trips.append(
                                {
                                    "recip": i,
                                    "target_t": ja,
                                    "target_w": jb,
                                    "r2_type": kind,
                                    "recip_game": rec["game_id"],
                                    "target_t_game": ta["game_id"],
                                    "target_w_game": tb["game_id"],
                                    "hamming_t": int(ham_t),
                                    "hamming_w": int(ham_w),
                                    "occupied_t": int(ta["occupied"]),
                                    "occupied_w": int(tb["occupied"]),
                                    "n_legal_t": int(len(ta["legal"])),
                                    "n_legal_w": int(len(tb["legal"])),
                                }
                            )
        pack = max_triplet_packing(trips) if store else {
            "n_triplets": after_legal_differ,
            "max_packing": None,
            "method": "count_only_not_packed",
            "upper_bound": min(after_legal_differ, len(pack_games) // 3, len(pack_recip)) if after_legal_differ else 0,
            "selected": [],
        }
        return {
            "recipients_with_ge2_targets": recips_ge2,
            "recipients_with_ge2_equal_occupancy_targets": recips_eq_occ_ge2,
            "tw_same_occupancy": after_occ,
            "tw_legal_count_slack": after_legal_slack,
            "tw_distinct_games": after_distinct_tw,
            "tw_legal_sets_differ": after_legal_differ,
            "n_triplets": after_legal_differ,
            "packing": pack,
        }

    trip_b = triplets_from(b_targets, "B_similar_board_legal_change", store=True)
    trip_a = (
        triplets_from(a_targets, "A_legal_diff", store=False)
        if expand_a
        else {
            "recipients_with_ge2_targets": sum(1 for pool in a_targets.values() if len(pool) >= 2),
            "recipients_with_ge2_equal_occupancy_targets": None,
            "tw_same_occupancy": None,
            "tw_legal_count_slack": None,
            "tw_distinct_games": None,
            "tw_legal_sets_differ": None,
            "n_triplets": None,
            "packing": {"max_packing": None, "upper_bound": None, "method": "a_triplets_not_expanded"},
            "n_pair_pools": len(a_targets),
        }
    )
    return {"pairs": pair, "B": trip_b, "A": trip_a}


def exclusion_breakdown(val_rows: list[dict], holdout_ids: set[str]) -> dict:
    from src.othello_r2_fiber_cost import reconstruct_frozen_pairs
    from src.othello_r2_fiber_spec import enumerate_candidates, fiber_excluded_games, match_wrong_donors, select_disjoint_pairs
    from src.othello_r2_null_state import prior_eval_games

    prior = prior_eval_games(val_rows, holdout_ids)
    fiber = fiber_excluded_games(val_rows, holdout_ids)
    spec = reconstruct_frozen_pairs(val_rows, holdout_ids, "2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3")
    enum = enumerate_candidates(val_rows, holdout_ids, prior["games"])
    selected = select_disjoint_pairs(enum["found"], 32, 16, 4182)
    nst_pair_games = set()
    for p in selected["pairs"]:
        nst_pair_games.add(str(p["recip_game"]))
        nst_pair_games.add(str(p["target_game"]))
    used = {p["recip_game"] for p in selected["pairs"]} | {p["target_game"] for p in selected["pairs"]}
    wrongs = [match_wrong_donors(enum["labeled"], p, holdout_ids | prior["games"] | used, 1, 4282) for p in selected["pairs"]]
    nst_wrong_games = set()
    for w in wrongs:
        for d in w.get("donors") or []:
            nst_wrong_games.add(str(d["game_id"]))
    all_games = set(map(str, prior["games"])) | nst_pair_games | nst_wrong_games
    return {
        "n_fiber": len(set(map(str, fiber))),
        "n_prior_fiber_spec": prior["n_total"],
        "n_spec_pairs": len(spec["pairs"]),
        "spec_hash": spec["pair_manifest_sha256"],
        "nst_hash": selected["pair_manifest_sha256"],
        "nst_hash_expected": NST_PAIR_HASH,
        "nst_hash_ok": int(selected["pair_manifest_sha256"] == NST_PAIR_HASH),
        "n_nst_pair_games": len(nst_pair_games),
        "n_nst_wrong_games": len(nst_wrong_games),
        "n_total_excluded": len(all_games),
        "games": all_games,
    }


def prefix_record_to_row(rec) -> dict:
    return {
        "game_id": rec.game_id,
        "split": rec.split,
        "t": rec.t,
        "tokens": list(rec.tokens),
        "squares": list(rec.squares),
        "board": np.asarray(rec.board).reshape(-1).tolist(),
        "player": int(rec.player),
        "raw_player": int(rec.raw_player),
        "legal": list(rec.legal),
        "next_square": rec.next_square,
        "terminal": bool(rec.terminal),
        "must_pass": bool(rec.must_pass),
        "occupied": int(rec.occupied),
        "stage": rec.stage,
    }


def fresh_cohort_rows(n_games: int, game_seed: int, prefix_seed: int, used_ids: set[str]) -> list[dict]:
    rng = np.random.default_rng(int(game_seed))
    recs = []
    i = 0
    while i < int(n_games):
        gid = f"fresh_{i:05d}"
        if gid in used_ids:
            i += 1
            continue
        squares = generate_game(rng, li_compatible=True, max_moves=60)
        recs.extend(prefixes_from_game(gid, squares, "fresh", li_compatible=True))
        i += 1
    sampled = subsample_prefixes(recs, k=6, seed=int(prefix_seed), min_prefix=5, max_from_end=5)
    return [prefix_record_to_row(r) for r in sampled]


def synthetic_b_triplet_rows() -> list[dict]:
    """Three occupancy-matched boards with Hamming 4/6 and distinct legal sets."""
    base = np.zeros((8, 8), dtype=np.int8)
    base[3, 3] = -1
    base[3, 4] = 1
    base[4, 3] = 1
    base[4, 4] = -1
    extra = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for i, (r, c) in enumerate(extra):
        base[r, c] = 1 if i % 2 == 0 else -1
    boards = [base.copy(), base.copy(), base.copy()]
    boards[1][5, 0] = 1
    boards[1][0, 0] = 0
    boards[1][5, 1] = -1
    boards[1][0, 1] = 0
    boards[2][6, 0] = 1
    boards[2][0, 2] = 0
    boards[2][6, 1] = -1
    boards[2][1, 0] = 0
    rows = []
    legals = ([10, 11, 12], [10, 11, 20], [10, 12, 21])
    for i, (board, legal) in enumerate(zip(boards, legals)):
        rows.append(
            {
                "game_id": f"syn_{i}",
                "split": "synthetic",
                "t": 8,
                "tokens": [1, 2, 3],
                "board": board.reshape(-1).tolist(),
                "player": 1,
                "legal": list(legal),
                "terminal": False,
                "must_pass": False,
                "occupied": int(occupancy(board)),
            }
        )
    return rows


def compare_to_production(val_rows: list[dict], holdout_ids: set[str], exclude: set[str]) -> dict:
    enum = enumerate_triplets(val_rows, holdout_ids, exclude)
    selected = select_triplets(enum["found"], WANT_A, WANT_B, TRIPLET_SEED)
    return {
        "n_eligible": enum["n_eligible"],
        "n_candidate_A": len(enum["found"]["A_legal_diff"]),
        "n_candidate_B": len(enum["found"]["B_similar_board_legal_change"]),
        "selected_A": selected["n_a"],
        "selected_B": selected["n_b"],
        "feasible": int(bool(selected.get("feasible"))),
        "triplet_sha256": selected["triplet_sha256"],
        "n_recipients": selected["n_recipients"],
        "n_targets": selected["n_targets"],
    }
