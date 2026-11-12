import chess

from robot_chess.config import EngineConfig
from robot_chess.engine import ChessEngine


def builtin(depth=2):
    return ChessEngine(EngineConfig(uci_path=None, fallback_depth=depth))


def test_returns_legal_move():
    b = chess.Board()
    r = builtin().choose_move(b)
    assert r.move in b.legal_moves and r.source == "builtin"


def test_finds_mate_in_one():
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert builtin().choose_move(b).move == chess.Move.from_uci("a1a8")


def test_takes_hanging_queen():
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert builtin().choose_move(b).move == chess.Move.from_uci("d1d5")
