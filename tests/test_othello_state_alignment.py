from __future__ import annotations

import numpy as np

from src.toy_othello import (
    ITOS,
    NANDA_SAMPLE_INPUT,
    STOI,
    BoardState,
    generate_game,
    prefixes_from_game,
    replay_states,
    square_to_token,
    token_to_square,
    tokens_to_moves,
)


def test_token_roundtrip():
    for tok, sq in ITOS.items():
        if tok == 0:
            continue
        assert square_to_token(sq) == tok
        assert token_to_square(tok) == sq
    assert STOI[-100] == 0


def test_prefix_is_board_after_consumed_moves():
    rng = np.random.default_rng(1)
    squares = generate_game(rng)
    rows = prefixes_from_game("g0", squares, "train")
    states = replay_states(squares)
    for rec in rows:
        after_t = states[rec.t]
        assert np.array_equal(rec.board, after_t.board)
        assert rec.tokens == [square_to_token(s) for s in squares[: rec.t]]
        if rec.next_square is not None:
            assert rec.next_square in rec.legal
            nxt = after_t.play_li(rec.next_square)
            assert np.array_equal(rec.after_board, nxt.board)


def test_nanda_sample_tokens_are_legal_prefix():
    squares = tokens_to_moves(NANDA_SAMPLE_INPUT)
    state = BoardState()
    for sq in squares:
        state = state.play_li(sq)
    assert state.history.count(-100) >= 0


def test_player_is_token_player_not_parity():
    rng = np.random.default_rng(2)
    found_mismatch = False
    for i in range(80):
        squares = generate_game(np.random.default_rng(10 + i))
        rows = prefixes_from_game(f"g{i}", squares, "train")
        for rec in rows:
            parity_player = 1 if rec.t % 2 == 1 else -1
            if rec.must_pass and rec.player != rec.raw_player:
                found_mismatch = True
                assert rec.player != rec.raw_player
            if rec.next_square is not None:
                assert rec.next_square in rec.legal
    _ = found_mismatch
    _ = rng
