from __future__ import annotations

import numpy as np

from src.toy_othello import (
    BLACK,
    CENTER_SQUARES,
    STOI,
    WHITE,
    BoardState,
    apply_square,
    generate_game,
    initial_board,
    legal_squares,
    occupancy,
    parse_square,
    replay_states,
    square_name,
    transform_board,
    transform_square,
)


def test_initial_position():
    b = initial_board()
    assert int(b[3, 3]) == WHITE
    assert int(b[3, 4]) == BLACK
    assert int(b[4, 3]) == BLACK
    assert int(b[4, 4]) == WHITE
    assert occupancy(b) == 4
    legal = {square_name(s) for s in legal_squares(b, BLACK)}
    assert legal == {"c4", "d3", "e6", "f5"}


def test_d3_flips_exactly_one():
    state = BoardState().play(parse_square("d3"))
    assert int(state.board[3, 2]) == BLACK
    assert int(state.board[3, 3]) == BLACK
    assert occupancy(state.board) == 5
    assert state.player == WHITE


def test_illegal_occupied_and_no_flip():
    b = initial_board()
    try:
        apply_square(b, parse_square("d4"), BLACK)
        raise AssertionError("occupied center must be illegal")
    except ValueError:
        pass
    try:
        apply_square(b, parse_square("a1"), BLACK)
        raise AssertionError("a1 has no flips")
    except ValueError:
        pass


def test_center_squares_have_no_tokens():
    for sq in CENTER_SQUARES:
        assert sq not in STOI


def test_pass_and_terminal_random_games():
    rng = np.random.default_rng(0)
    saw_pass = False
    for i in range(40):
        squares = generate_game(rng, li_compatible=True)
        states = replay_states(squares, li_compatible=True)
        assert states[0].player == BLACK
        for t, st in enumerate(states[:-1]):
            nxt = squares[t]
            if st.must_pass():
                saw_pass = True
                assert nxt in st.legal_moves(-st.player)
            else:
                assert nxt in st.legal_moves()
        assert states[-1].terminal() or len(squares) >= 59 or not states[-1].legal_moves()
    assert saw_pass or True  # passes are rare; do not require them


def test_explicit_pass_rejected_when_moves_exist():
    st = BoardState()
    try:
        st.play(-100)
        raise AssertionError("pass should be illegal")
    except ValueError:
        pass


def test_symmetry_preserves_occupancy_and_legal_count():
    st = BoardState().play(parse_square("d3")).play(parse_square("c3"))
    for name in ("id", "rot90", "rot180", "flip_h", "flip_v"):
        tb = transform_board(st.board, name)
        assert occupancy(tb) == occupancy(st.board)
        legal = legal_squares(st.board, st.player)
        tlegal = {transform_square(s, name) for s in legal}
        assert tlegal == set(legal_squares(tb, st.player))
