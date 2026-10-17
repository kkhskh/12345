from __future__ import annotations

from pathlib import Path

from run_08a_prepare_othello_data import split_games, write_dataset
from src.toy_othello import generate_game, prefixes_from_game


def test_splits_are_by_full_games(tmp_path: Path):
    games = {"train": [["a"], ["b"]], "val": [["c"]], "test": [["d"]]}
    # uniqueness across splits
    seqs = {tuple(g) for split in games.values() for g in split}
    assert len(seqs) == 4


def test_no_shared_sequences_or_ids():
    splits = split_games(n_train=20, n_val=8, n_test=8, seed=0)
    ids = []
    seqs = []
    for name, games in splits.items():
        for gid, sq in games:
            ids.append(gid)
            seqs.append(tuple(sq))
    assert len(ids) == len(set(ids))
    assert len(seqs) == len(set(seqs))
    by_split = {k: {tuple(sq) for _, sq in v} for k, v in splits.items()}
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])


def test_positions_keep_game_id():
    splits = split_games(n_train=6, n_val=2, n_test=2, seed=1)
    rows = []
    for split, games in splits.items():
        for gid, sq in games:
            rows.extend(prefixes_from_game(gid, sq, split))
    assert all(r.game_id for r in rows)
    games_in_train = {r.game_id for r in rows if r.split == "train"}
    games_in_test = {r.game_id for r in rows if r.split == "test"}
    assert not (games_in_train & games_in_test)
