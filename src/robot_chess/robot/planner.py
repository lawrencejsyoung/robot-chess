"""Decompose a chess move into physical pick-and-place primitives.

GR00T is trained on short, language-conditioned manipulation skills, so the
interface between the reasoning system and the action model is deliberately
narrow: one primitive = "move the <colour> <piece> from A to B", where A/B are
squares or named off-board zones.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from ..config import RobotConfig
from ..physical import is_square


@dataclass(frozen=True)
class PickPlace:
    piece: chess.Piece
    src: str
    dst: str
    reason: str = "move"  # move | capture | castle_rook | en_passant | promote_remove | promote_place | correction

    def location_phrase(self, loc: str) -> str:
        if is_square(loc):
            return loc
        return loc.replace("_", " ")

    def describe(self) -> str:
        colour = "white" if self.piece.color == chess.WHITE else "black"
        name = chess.piece_name(self.piece.piece_type)
        return f"{colour} {name} {self.location_phrase(self.src)} -> {self.location_phrase(self.dst)}"


class MovePlanner:
    def __init__(self, cfg: RobotConfig | None = None):
        cfg = cfg or RobotConfig()
        self.graveyard = {chess.WHITE: cfg.graveyard_white, chess.BLACK: cfg.graveyard_black}
        self.reserve = cfg.reserve

    def plan(self, board: chess.Board, move: chess.Move) -> list[PickPlace]:
        """Primitives to physically realise ``move`` from position ``board``.

        Ordering matters: the destination square is always cleared before a
        piece is placed on it.
        """
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {move.uci()} in {board.fen()}")
        mover = board.piece_at(move.from_square)
        assert mover is not None
        fn, tn = chess.square_name(move.from_square), chess.square_name(move.to_square)
        steps: list[PickPlace] = []

        if board.is_en_passant(move):
            victim_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            victim = board.piece_at(victim_sq)
            assert victim is not None
            steps.append(PickPlace(victim, chess.square_name(victim_sq), self.graveyard[victim.color], "en_passant"))
        elif board.is_capture(move):
            victim = board.piece_at(move.to_square)
            assert victim is not None
            steps.append(PickPlace(victim, tn, self.graveyard[victim.color], "capture"))

        if board.is_castling(move):
            # python-chess encodes castling as king e1->g1 / e1->c1.
            rank = chess.square_rank(move.from_square)
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            steps.append(PickPlace(mover, fn, tn, "move"))
            steps.append(PickPlace(
                chess.Piece(chess.ROOK, mover.color),
                chess.square_name(rook_from), chess.square_name(rook_to), "castle_rook",
            ))
        elif move.promotion:
            steps.append(PickPlace(mover, fn, self.graveyard[mover.color], "promote_remove"))
            steps.append(PickPlace(chess.Piece(move.promotion, mover.color), self.reserve, tn, "promote_place"))
        else:
            steps.append(PickPlace(mover, fn, tn, "move"))
        return steps
