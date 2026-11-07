"""Validation pairing must stay off the sealed confirmation games."""

from __future__ import annotations

from pathlib import Path

import pytest

from collections import Counter

from src.othello_r2 import load_holdout_game_ids, pair_census_public, r2_pair_categories, require_no_confirmation, smoke_ab_quotas
from src.othello_runutil import add_labels, load_split


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "artifacts/toy_othello/data/prefixes_val.jsonl"
HOLDOUT = ROOT / "configs/08r1_fresh_holdout_manifest.json"


def test_confirmation_ids_are_rejected():
    with pytest.raises(RuntimeError, match="confirmation-split"):
        require_no_confirmation([{"game_id": "confirm_00000"}])


def test_holdout_manifest_hash_and_prefix():
    holdout = load_holdout_game_ids(HOLDOUT)
    assert holdout["split"] == "08r2_08r3_confirmation"
    assert holdout["n_games"] == 400
    assert holdout["manifest_sha256"] == "19d9757509256139900dec0647a331c28e549f17b87d7bc8d707ebccad989d24"
    assert all(i.startswith("confirm_") for i in holdout["game_ids"])


@pytest.mark.skipif(not VAL.is_file(), reason="local val prefixes missing")
def test_validation_pairs_are_game_grouped_and_disjoint_from_holdout():
    holdout = load_holdout_game_ids(HOLDOUT)
    rows = add_labels(load_split(VAL.parent, "val"))
    require_no_confirmation(rows, holdout["game_ids"])
    n_a, n_b = smoke_ab_quotas(4)
    pairs = r2_pair_categories(rows, 4, 1082, holdout["game_ids"], n_a=n_a, n_b=n_b, include_c=False)
    assert pairs
    games = {p["recip_game"] for p in pairs} | {p["target_game"] for p in pairs}
    assert not (games & holdout["game_ids"])
    counts = Counter(p["r2_type"] for p in pairs)
    assert counts["A_legal_diff"] == 2
    assert counts["B_similar_board_legal_change"] == 2
    assert counts["D_wrong_target"] == 4
    assert counts.get("C_same_legal_diff_board", 0) == 0
    census = pair_census_public(rows, holdout["game_ids"])
    assert census["b_available"]
    assert not census["c_available_strict"]
    primary = [p for p in pairs if p["r2_type"] != "D_wrong_target"]
    assert all(p["same_player"] == 1 for p in primary)
    assert all(p["recip_game"] != p["target_game"] for p in primary)
    assert all(p["used_confirmation"] == 0 for p in pairs)
    assert not any(p["r2_type"] == "A_legal_diff" and p.get("hamming", 99) <= 8 for p in primary)
