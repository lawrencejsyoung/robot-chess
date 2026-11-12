"""The reasoning system: picks the robot's move.

Uses a UCI engine (Stockfish) when available and falls back to a small
built-in alpha-beta searcher so the whole stack runs anywhere (CI, laptops
without Stockfish, the simulator).
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass

import chess
import chess.engine

from .config import EngineConfig

log = logging.getLogger(__name__)

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE = 100_000


@dataclass
class EngineResult:
    move: chess.Move
    think_time_s: float
    score_cp: int | None = None  # from the side-to-move's perspective
    source: str = "builtin"


class ChessEngine:
    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self._uci: chess.engine.SimpleEngine | None = None
        path = self.cfg.uci_path and shutil.which(self.cfg.uci_path)
        if path:
            try:
                self._uci = chess.engine.SimpleEngine.popen_uci(path)
                if self.cfg.skill_level is not None and "Skill Level" in self._uci.options:
                    self._uci.configure({"Skill Level": self.cfg.skill_level})
                log.info("Using UCI engine at %s", path)
            except Exception as exc:  # pragma: no cover - depends on host
                log.warning("Failed to start UCI engine (%s); using built-in", exc)
                self._uci = None
        else:
            log.info("No UCI engine found; using built-in searcher")

    @property
    def source(self) -> str:
        return "uci" if self._uci else "builtin"

    def choose_move(self, board: chess.Board) -> EngineResult:
        if board.is_game_over():
            raise ValueError("Game is over; no move to choose")
        start = time.perf_counter()
        if self._uci:
            res = self._uci.play(
                board,
                chess.engine.Limit(time=self.cfg.think_time_s),
                info=chess.engine.INFO_SCORE,
            )
            score = res.info.get("score")
            cp = score.relative.score(mate_score=MATE_SCORE) if score else None
            assert res.move is not None
            return EngineResult(res.move, time.perf_counter() - start, cp, "uci")
        score, move = self._search(board, self.cfg.fallback_depth)
        return EngineResult(move, time.perf_counter() - start, score, "builtin")

    def close(self) -> None:
        if self._uci:
            self._uci.quit()
            self._uci = None

    def __enter__(self) -> "ChessEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- built-in negamax with alpha-beta ------------------------------------

    @staticmethod
    def evaluate(board: chess.Board) -> int:
        """Material + light mobility, from the side to move's perspective."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        # Small centre bonus for pawns and knights.
        centre = chess.SquareSet(chess.BB_CENTER)
        for sq in centre:
            p = board.piece_at(sq)
            if p and p.piece_type in (chess.PAWN, chess.KNIGHT):
                score += 15 if p.color == chess.WHITE else -15
        return score if board.turn == chess.WHITE else -score

    def _ordered_moves(self, board: chess.Board) -> list[chess.Move]:
        def key(m: chess.Move) -> int:
            k = 0
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                k += 10 * PIECE_VALUES.get(victim.piece_type if victim else chess.PAWN, 100)
                k -= PIECE_VALUES.get(attacker.piece_type if attacker else chess.PAWN, 0) // 10
            if m.promotion:
                k += PIECE_VALUES[m.promotion]
            return -k

        return sorted(board.legal_moves, key=key)

    def _search(self, board: chess.Board, depth: int) -> tuple[int, chess.Move]:
        best_move: chess.Move | None = None
        alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        for move in self._ordered_moves(board):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if best_move is None or score > alpha:
                alpha, best_move = score, move
        assert best_move is not None
        return alpha, best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        if depth <= 0 or board.is_game_over():
            return self.evaluate(board)
        for move in self._ordered_moves(board):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha
