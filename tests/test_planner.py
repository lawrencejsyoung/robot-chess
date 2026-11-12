import chess
import pytest

from robot_chess.robot.planner import MovePlanner


def plan(fen, uci):
    b = chess.Board(fen)
    return MovePlanner().plan(b, chess.Move.from_uci(uci))


def test_quiet_move():
    (p,) = plan(chess.STARTING_FEN, "g1f3")
    assert (p.src, p.dst, p.piece.symbol()) == ("g1", "f3", "N")


def test_capture_clears_target_first():
    steps = plan("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5")
    assert [(s.src, s.dst, s.reason) for s in steps] == [
        ("d5", "graveyard_black", "capture"),
        ("d1", "d5", "move"),
    ]


@pytest.mark.parametrize("uci,rook", [("e1g1", ("h1", "f1")), ("e1c1", ("a1", "d1"))])
def test_castling(uci, rook):
    steps = plan("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", uci)
    assert steps[0].piece.piece_type == chess.KING
    assert (steps[1].src, steps[1].dst) == rook


def test_en_passant_removes_adjacent_pawn():
    steps = plan("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
    assert (steps[0].src, steps[0].dst) == ("d5", "graveyard_black")
    assert (steps[1].src, steps[1].dst) == ("e5", "d6")


def test_capture_promotion():
    steps = plan("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q")
    assert [(s.src, s.dst) for s in steps] == [
        ("b8", "graveyard_black"),
        ("a7", "graveyard_white"),
        ("reserve", "b8"),
    ]
    assert steps[-1].piece == chess.Piece(chess.QUEEN, chess.WHITE)


def test_illegal_move_rejected():
    with pytest.raises(ValueError):
        plan(chess.STARTING_FEN, "e2e5")
