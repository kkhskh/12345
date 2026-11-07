"""Shared 0A I/O and pairing helpers. No obstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.toy_othello import (
    board_hamming,
    labels_to_board,
    legal_mask_from_board,
    legal_squares,
    to_absolute_labels,
    to_relative_labels,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_split(data_dir: Path, split: str) -> list[dict]:
    return load_jsonl(Path(data_dir) / f"prefixes_{split}.jsonl")


def board_arr(row: dict) -> np.ndarray:
    return np.asarray(row["board"], dtype=np.int8).reshape(8, 8)


def add_labels(rows: list[dict]) -> list[dict]:
    for r in rows:
        b = board_arr(r)
        r["abs_lab"] = to_absolute_labels(b)
        r["rel_lab"] = to_relative_labels(b, int(r["player"]))
        r["legal_mask"] = legal_mask_from_board(b, int(r["player"]))
        r["board_np"] = b
    return rows


def tokens_of(rows: list[dict]) -> list[list[int]]:
    return [list(r["tokens"]) for r in rows]


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def pair_candidates(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = {i: r for i, r in enumerate(rows) if r.get("legal") and not r.get("terminal")}
    keys = list(idx.keys())
    by_bucket: dict[tuple, list[int]] = {}
    for i in keys:
        r = idx[i]
        occ = int(r["occupied"])
        bucket = (int(r["player"]), occ // 4, min(len(r["legal"]), 8))
        by_bucket.setdefault(bucket, []).append(i)
    pairs = []
    tries = 0
    used = set()
    while len(pairs) < n and tries < n * 80:
        tries += 1
        i = int(rng.choice(keys))
        if i in used:
            continue
        r = idx[i]
        bucket = (int(r["player"]), int(r["occupied"]) // 4, min(len(r["legal"]), 8))
        cands = [j for j in by_bucket.get(bucket, []) if j != i and j not in used]
        if not cands:
            continue
        j = int(rng.choice(cands))
        d = idx[j]
        ham = board_hamming(r["board_np"], d["board_np"])
        lr, ld = set(r["legal"]), set(d["legal"])
        if ham == 0:
            kind = "E_same_state"
        elif lr == ld:
            kind = "D_same_legal"
        elif ham <= 6:
            kind = "C_local_diff"
        elif lr != ld:
            kind = "B_legal_diff"
        else:
            kind = "A_same_side"
        if lr != ld and kind == "A_same_side":
            kind = "B_legal_diff"
        if kind == "A_same_side" and lr != ld:
            kind = "B_legal_diff"
        pairs.append(
            {
                "recip": i,
                "donor": j,
                "type": kind if ham > 0 else "E_same_state",
                "hamming": ham,
                "n_became_legal": len(ld - lr),
                "n_became_illegal": len(lr - ld),
                "same_player": int(r["player"]) == int(d["player"]),
            }
        )
        used.add(i)
        used.add(j)
    return pairs


def legality_partition(rec_legal: list[int], don_legal: list[int]) -> dict[str, list[int]]:
    a, b = set(rec_legal), set(don_legal)
    return {
        "became_legal": sorted(b - a),
        "became_illegal": sorted(a - b),
        "stayed_legal": sorted(a & b),
        "stayed_illegal": [],
    }


def mean_prob_on(probs: np.ndarray, squares: list[int], token_fn) -> float:
    """Historical 0A helper: mean P(token) over a set, not total legal mass.

    Kept unchanged so re-running 08e/08f reproduces the original tables.
    Corrected mass/mean pair is `src.othello_r1.intervention_probability_metrics`.
    """
    if not squares:
        return float("nan")
    toks = [token_fn(s) for s in squares if s != -100]
    if not toks:
        return float("nan")
    return float(probs[toks].mean())


def decode_board_from_rel(pred_rel: np.ndarray, player: int) -> np.ndarray:
    from src.toy_othello import relative_to_absolute

    return labels_to_board(relative_to_absolute(pred_rel, player))


def induced_legal(pred_abs: np.ndarray, player: int) -> list[int]:
    return legal_squares(labels_to_board(pred_abs), player)
