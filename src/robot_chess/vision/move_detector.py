"""Infer moves from occupancy grids.

Occupancy alone can't tell a knight from a bishop, but it doesn't need to: we
know the previous position, so we only have to find which *legal* move
produces the observed grid. That makes castling, en passant and captures fall
out naturally. The only ambiguity left is the promotion piece, which defaults
to a queen (override with ``promotion``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import chess
import numpy as np

from ..physical import occupancy_from_board


@dataclass
class Detection:
    move: chess.Move | None
    occupancy: list[int]
    candidates: list[chess.Move]
    changed_squares: list[int]
    latency_s: float = 0.0

    @property
    def ambiguous(self) -> bool:
        return len({(m.from_square, m.to_square) for m in self.candidates}) > 1


def expected_after(board: chess.Board, move: chess.Move) -> list[int]:
    board.push(move)
    try:
        return occupancy_from_board(board)
    finally:
        board.pop()


def diff_squares(a: list[int], b: list[int]) -> list[int]:
    return [i for i in range(64) if a[i] != b[i]]


class MoveDetector:
    def __init__(self, classify: Callable[[np.ndarray], list[int]], stable_frames: int = 3):
        self.classify = classify
        self.stable_frames = stable_frames

    def match(self, board: chess.Board, occ: list[int], promotion: int = chess.QUEEN) -> Detection:
        changed = diff_squares(occupancy_from_board(board), occ)
        cands = [m for m in board.legal_moves if expected_after(board, m) == occ]
        move = None
        if cands:
            promos = [m for m in cands if m.promotion == promotion]
            move = promos[0] if promos else cands[0]
        return Detection(move, occ, cands, changed)

    def read_stable(self, read_frame: Callable[[], np.ndarray], max_frames: int = 50) -> list[int]:
        """Read frames until ``stable_frames`` consecutive grids agree."""
        last, streak = None, 0
        for _ in range(max_frames):
            occ = self.classify(read_frame())
            streak = streak + 1 if occ == last else 1
            last = occ
            if streak >= self.stable_frames:
                return occ
        raise TimeoutError("Board never stabilised (hand in view? lighting flicker?)")

    def wait_for_move(
        self,
        board: chess.Board,
        read_frame: Callable[[], np.ndarray],
        timeout_s: float | None = None,
        poll_s: float = 0.2,
        on_unmatched: Callable[[Detection], None] | None = None,
    ) -> Detection:
        """Block until the observed board matches a legal move from ``board``."""
        start = time.perf_counter()
        baseline = occupancy_from_board(board)
        last_reported: list[int] | None = None
        while True:
            occ = self.read_stable(read_frame)
            if occ != baseline:
                det = self.match(board, occ)
                if det.move is not None:
                    det.latency_s = time.perf_counter() - start
                    return det
                if on_unmatched and occ != last_reported:
                    on_unmatched(det)
                    last_reported = occ
            if timeout_s is not None and time.perf_counter() - start > timeout_s:
                raise TimeoutError("No legal move detected")
            if poll_s:
                time.sleep(poll_s)

    def verify(self, expected: list[int], read_frame: Callable[[], np.ndarray]) -> list[int]:
        """Return squares whose observed occupancy differs from ``expected``."""
        return diff_squares(expected, self.read_stable(read_frame))
