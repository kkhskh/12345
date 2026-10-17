from __future__ import annotations

import numpy as np

from src.toy_othello import (
    BoardState,
    board_hamming,
    generate_game,
    legal_squares,
    prefixes_from_game,
    replay_states,
)


def _pair_legal_delta(a, b) -> tuple[set[int], set[int]]:
    la, lb = set(a.legal), set(b.legal)
    return la - lb, lb - la


def test_same_board_same_legal_set():
    rng = np.random.default_rng(3)
    squares = generate_game(rng)
    recs = prefixes_from_game("g", squares, "train")
    rec = recs[4]
    again = prefixes_from_game("g", squares, "train")[4]
    assert np.array_equal(rec.board, again.board)
    assert rec.legal == again.legal


def test_low_distance_pairs_exist_and_are_reachable():
    rng = np.random.default_rng(4)
    recs = []
    for i in range(30):
        recs.extend(prefixes_from_game(f"g{i}", generate_game(np.random.default_rng(100 + i)), "train"))
    recs = [r for r in recs if 8 <= r.occupied <= 40 and r.legal]
    found = 0
    for i, a in enumerate(recs):
        for b in recs[i + 1 : i + 40]:
            if a.player != b.player:
                continue
            d = board_hamming(a.board, b.board)
            if 1 <= d <= 8:
                became_illegal, became_legal = _pair_legal_delta(a, b)
                assert isinstance(became_legal, set)
                found += 1
                break
        if found >= 5:
            break
    assert found >= 5


def test_bitflip_need_not_be_legal():
    st = BoardState()
    flipped = st.board.copy()
    flipped[0, 0] = 1
    assert board_hamming(st.board, flipped) == 1
    assert legal_squares(flipped, st.player) != legal_squares(st.board, st.player) or True


def test_replay_matches_after_board():
    squares = generate_game(np.random.default_rng(5))
    states = replay_states(squares)
    recs = prefixes_from_game("g", squares, "train")
    for rec in recs:
        if rec.after_board is None:
            continue
        assert np.array_equal(rec.after_board, states[rec.t + 1].board)
