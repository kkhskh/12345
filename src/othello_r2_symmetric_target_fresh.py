"""08R2-SYMMETRIC-TARGET-FRESH: new independently seeded cohort.

Does not rewrite leftover-val SYMMETRIC_TARGET_CENSUS_INFEASIBLE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r2_fiber_spec import _kind
from src.othello_r2_symmetric_target import (
    LEGAL_COUNT_SLACK,
    WANT_A,
    WANT_B,
    _legal_differs,
    _vs_recip_ok,
    finalize_triplets,
)
from src.othello_runutil import add_labels
from src.othello_r2_symmetric_target_census_audit import prefix_record_to_row
from src.toy_othello import generate_game, prefixes_from_game, sha256_json, subsample_prefixes


FRESH_DIRNAME = "symmetric_target_fresh"
COHORT_SEED = 9182
PREFIX_SEED = 9282
TRIPLET_SEED = 9382
BOOT_SEED = 9482
PILOT_COHORT_SEED = 8182
PILOT_PREFIX_SEED = 8282
N_GAMES = 400
WANT_UNION = 96
PACK_TRIALS = 2000
STATUS_INFEASIBLE = "FRESH_TRIPLET_CENSUS_INFEASIBLE"
STATUS_AWAITING = "IMPLEMENTED_AWAITING_REAL_MODEL_EXECUTION"
BANNED = {
    "null_state_transfer",
    "state_semantics",
    "fiber_spec",
    "fiber_cost",
    "fiber_cost_forensic",
    "fiber_cost_fixed_js",
    "symmetric_target",
    "symmetric_target_census_audit",
}


def isolate_fresh_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == FRESH_DIRNAME:
        out = requested
    elif requested.name == "08r2":
        out = requested / FRESH_DIRNAME
    else:
        out = requested / "results" / "08r2" / FRESH_DIRNAME
    if any(p in BANNED for p in out.parts):
        raise ValueError(f"fresh study must not write under preserved dirs: {out}")
    return {"requested": str(requested), "results_dir": str(out)}


def sequence_hash(squares) -> str:
    return sha256_json([int(s) for s in squares])


def board_state_hash(row: dict) -> str:
    return sha256_json({"board": [int(x) for x in np.asarray(row["board"]).reshape(-1)], "player": int(row["player"])})


def generate_game_records(n_games: int, game_seed: int, id_prefix: str) -> list[dict]:
    rng = np.random.default_rng(int(game_seed))
    out = []
    for i in range(int(n_games)):
        squares = generate_game(rng, li_compatible=True, max_moves=60)
        gid = f"{id_prefix}_{i:05d}"
        out.append({"game_id": gid, "squares": [int(s) for s in squares], "sequence_sha256": sequence_hash(squares), "seed": int(game_seed)})
    return out


def prefixes_from_records(records: list[dict], prefix_seed: int, split: str, k: int = 6) -> list[dict]:
    recs = []
    for g in records:
        recs.extend(prefixes_from_game(g["game_id"], g["squares"], split, li_compatible=True))
    sampled = subsample_prefixes(recs, k=k, seed=int(prefix_seed), min_prefix=5, max_from_end=5)
    rows = [prefix_record_to_row(r) for r in sampled]
    seq = {g["game_id"]: g["sequence_sha256"] for g in records}
    for r in rows:
        r["sequence_sha256"] = seq.get(r["game_id"])
        r["board_state_sha256"] = board_state_hash(r)
    return rows


def longest_squares_by_game(rows: list[dict]) -> dict[str, list[int]]:
    best: dict[str, list[int]] = {}
    for r in rows:
        gid = str(r["game_id"])
        sq = [int(s) for s in (r.get("squares") or [])]
        if gid not in best or len(sq) > len(best[gid]):
            best[gid] = sq
    return best


def banned_sequence_hashes(split_rows: dict[str, list[dict]], pilot_records: list[dict]) -> dict:
    hashes: set[str] = set()
    by_source = {}
    for name, rows in split_rows.items():
        got = {sequence_hash(sq) for sq in longest_squares_by_game(rows).values() if sq}
        by_source[name] = len(got)
        hashes |= got
    pilot = {g["sequence_sha256"] for g in pilot_records}
    by_source["pilot_8182"] = len(pilot)
    hashes |= pilot
    return {"hashes": hashes, "by_source": by_source, "n_total": len(hashes)}


def apply_exclusions(records: list[dict], banned_hashes: set[str], banned_ids: set[str]) -> dict:
    kept, dropped = [], []
    for g in records:
        if g["game_id"] in banned_ids or str(g["game_id"]).startswith("confirm_"):
            dropped.append({**g, "reason": "id"})
        elif g["sequence_sha256"] in banned_hashes:
            dropped.append({**g, "reason": "sequence_hash"})
        else:
            kept.append(g)
    return {"kept": kept, "dropped": dropped, "n_kept": len(kept), "n_dropped": len(dropped)}


def witness_ok(selected: dict, n_a: int = WANT_A, n_b: int = WANT_B, n_union: int | None = None) -> dict:
    if n_union is None:
        n_union = 3 * (int(n_a) + int(n_b))
    trips = selected.get("triplets") or []
    recip = {t["recip_game"] for t in trips}
    targets = {t["target_t_game"] for t in trips} | {t["target_w_game"] for t in trips}
    union = recip | targets
    roles_ok = all(len({t["recip_game"], t["target_t_game"], t["target_w_game"]}) == 3 for t in trips)
    return {
        "n_a": selected.get("n_a", 0),
        "n_b": selected.get("n_b", 0),
        "n_recipients": len(recip),
        "n_targets": len(targets),
        "n_union": len(union),
        "recip_target_overlap": sorted(recip & targets),
        "roles_ok": int(roles_ok),
        "ok": int(
            selected.get("n_a") == n_a
            and selected.get("n_b") == n_b
            and len(recip) == n_a + n_b
            and len(targets) == 2 * (n_a + n_b)
            and len(union) == n_union
            and not (recip & targets)
            and roles_ok
        ),
    }


def _take(pool, limit, used):
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


def joint_ab_packing(found: dict, n_a: int, n_b: int, seed: int, n_trials: int = PACK_TRIALS) -> dict:
    """Pack from already-materialized A and B pools."""
    a = list(found.get("A_legal_diff") or [])
    b = list(found.get("B_similar_board_legal_change") or [])
    rng = np.random.default_rng(int(seed))
    best = {"triplets": [], "n_a": 0, "n_b": 0}
    b_games = set()
    for e in b:
        b_games |= {e["recip_game"], e["target_t_game"], e["target_w_game"]}
    upper_b = min(len(b), len(b_games) // 3, len({e["recip_game"] for e in b}))
    for _ in range(int(n_trials)):
        used: set[str] = set()
        order_b = b.copy()
        order_a = a.copy()
        rng.shuffle(order_b)
        rng.shuffle(order_a)
        got_b = _take(order_b, n_b, used)
        got_a = _take(order_a, n_a, used)
        if len(got_b) > best["n_b"] or (len(got_b) == best["n_b"] and len(got_a) > best["n_a"]):
            best = {"triplets": got_a + got_b, "n_a": len(got_a), "n_b": len(got_b)}
        if len(got_a) == n_a and len(got_b) == n_b:
            best["feasible"] = True
            out = finalize_triplets(best, seed)
            out["method"] = "multistart_greedy_joint"
            out["upper_b"] = int(upper_b)
            out["n_trials"] = int(n_trials)
            out["witness"] = witness_ok(out, n_a, n_b)
            out["feasible"] = bool(out["witness"]["ok"])
            return out
    best["feasible"] = False
    out = finalize_triplets(best, seed)
    out["method"] = "multistart_greedy_joint"
    out["upper_b"] = int(upper_b)
    out["n_trials"] = int(n_trials)
    out["witness"] = witness_ok(out, n_a, n_b)
    out["feasible"] = False
    out["infeasible_certificate"] = "b_upper_lt_16" if upper_b < n_b else "no_joint_witness_after_multistart"
    return out


def _a_triplet(rec, i, ta, ja, tb, jb, ham_t, ham_w) -> dict:
    return {
        "recip": i,
        "target_t": ja,
        "target_w": jb,
        "r2_type": "A_legal_diff",
        "recip_game": rec["game_id"],
        "target_t_game": ta["game_id"],
        "target_w_game": tb["game_id"],
        "hamming_t": int(ham_t),
        "hamming_w": int(ham_w),
        "used_confirmation": 0,
    }


def pack_a_online(labeled: list[dict], elig: list[int], used: set[str], n_a: int, rng: np.random.Generator) -> list[dict]:
    remain = [i for i in elig if str(labeled[i]["game_id"]) not in used]
    rng.shuffle(remain)
    got = []
    used_local = set(used)
    for i in remain:
        if len(got) >= n_a:
            break
        rec = labeled[i]
        if rec["game_id"] in used_local:
            continue
        pool = []
        for j in remain:
            if i == j:
                continue
            tgt = labeled[j]
            if tgt["game_id"] in used_local or rec["game_id"] == tgt["game_id"]:
                continue
            if int(rec["player"]) != int(tgt["player"]) or not _vs_recip_ok(rec, tgt):
                continue
            kind, ham = _kind(rec, tgt)
            if kind != "A_legal_diff":
                continue
            pool.append((j, ham, int(tgt["occupied"]), len(tgt["legal"])))
            if len(pool) >= 16:
                break
        by_occ: dict[int, list] = {}
        for item in pool:
            by_occ.setdefault(item[2], []).append(item)
        taken = False
        for group in by_occ.values():
            if taken or len(group) < 2:
                continue
            for a in range(len(group)):
                if taken:
                    break
                for b in range(a + 1, len(group)):
                    ja, ham_t, _, nt = group[a]
                    jb, ham_w, _, nw = group[b]
                    ta, tb = labeled[ja], labeled[jb]
                    games = {rec["game_id"], ta["game_id"], tb["game_id"]}
                    if len(games) < 3 or games & used_local or abs(nt - nw) > LEGAL_COUNT_SLACK:
                        continue
                    if not _legal_differs(ta, tb):
                        continue
                    got.append(_a_triplet(rec, i, ta, ja, tb, jb, ham_t, ham_w))
                    used_local |= games
                    taken = True
                    break
    return got


def enumerate_kind_triplets(rows: list[dict], kinds: set[str]) -> dict:
    labeled = add_labels([{**r} for r in rows])
    elig = [i for i, r in enumerate(labeled) if r.get("legal") and not r.get("terminal")]
    found = {k: [] for k in ("A_legal_diff", "B_similar_board_legal_change")}
    for i in elig:
        rec = labeled[i]
        cands = {k: [] for k in kinds}
        for j in elig:
            if i == j:
                continue
            tgt = labeled[j]
            if rec["game_id"] == tgt["game_id"] or int(rec["player"]) != int(tgt["player"]):
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
            for group in by_occ.values():
                if len(group) < 2:
                    continue
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ja, ham_t, _, nt = group[a]
                        jb, ham_w, _, nw = group[b]
                        ta, tb = labeled[ja], labeled[jb]
                        if ta["game_id"] == tb["game_id"] or abs(nt - nw) > LEGAL_COUNT_SLACK:
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
                                "hamming_t": int(ham_t),
                                "hamming_w": int(ham_w),
                                "used_confirmation": 0,
                            }
                        )
    return {"labeled": labeled, "found": found, "n_eligible": len(elig), "elig": elig}


def joint_pack_from_rows(rows: list[dict], n_a: int, n_b: int, seed: int, n_trials: int = PACK_TRIALS) -> dict:
    enum_b = enumerate_kind_triplets(rows, {"B_similar_board_legal_change"})
    b = enum_b["found"]["B_similar_board_legal_change"]
    labeled, elig = enum_b["labeled"], enum_b["elig"]
    rng = np.random.default_rng(int(seed))
    b_games = set()
    for e in b:
        b_games |= {e["recip_game"], e["target_t_game"], e["target_w_game"]}
    upper_b = min(len(b), len(b_games) // 3, len({e["recip_game"] for e in b}) if b else 0)
    best = {"triplets": [], "n_a": 0, "n_b": 0}
    for _ in range(int(n_trials)):
        used: set[str] = set()
        order_b = b.copy()
        rng.shuffle(order_b)
        got_b = _take(order_b, n_b, used)
        got_a = pack_a_online(labeled, elig, used, n_a, rng) if len(got_b) == n_b else []
        if len(got_b) > best["n_b"] or (len(got_b) == best["n_b"] and len(got_a) > best["n_a"]):
            best = {"triplets": got_a + got_b, "n_a": len(got_a), "n_b": len(got_b)}
        if len(got_a) == n_a and len(got_b) == n_b:
            best["feasible"] = True
            out = finalize_triplets(best, seed)
            out.update({"method": "b_then_online_a", "upper_b": int(upper_b), "n_trials": int(n_trials), "n_b_candidates": len(b)})
            out["witness"] = witness_ok(out, n_a, n_b)
            out["feasible"] = bool(out["witness"]["ok"])
            return out
    best["feasible"] = False
    out = finalize_triplets(best, seed)
    out.update({"method": "b_then_online_a", "upper_b": int(upper_b), "n_trials": int(n_trials), "n_b_candidates": len(b)})
    out["witness"] = witness_ok(out, n_a, n_b)
    out["feasible"] = False
    out["infeasible_certificate"] = "b_upper_lt_16" if upper_b < n_b else "no_joint_witness_after_multistart"
    return out
