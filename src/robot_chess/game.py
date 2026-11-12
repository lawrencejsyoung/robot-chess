"""The coordinator between the reasoning system and the embodied system.

    human's turn:  vision waits for a stable board that matches a legal move
    robot's turn:  engine picks a move -> planner -> primitives
                   for each primitive: execute -> verify with vision
                   -> on mismatch: plan a correction and retry
                   -> after ``max_retries``: ask the human for help
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol

import chess
import numpy as np

from .engine import ChessEngine
from .metrics import GameSummary, MetricsLogger, MoveRecord
from .physical import EMPTY, WHITE, BLACK, is_square
from .robot.base import RobotController
from .robot.planner import MovePlanner, PickPlace
from .vision.move_detector import Detection, MoveDetector, diff_squares

log = logging.getLogger(__name__)


class HumanPlayer(Protocol):
    def take_turn(self, board: chess.Board) -> chess.Move | None:
        """Make a move on the physical board.

        A real human returns ``None`` (we learn the move from vision). A
        simulated human returns the ground-truth move so detection accuracy
        can be scored.
        """
        ...


class AssistHandler(Protocol):
    def __call__(self, expected: dict[int, chess.Piece], mismatched: list[int]) -> None:
        """Ask a person to make the board match ``expected``; return when done."""
        ...


def _colour_code(piece: chess.Piece) -> int:
    return WHITE if piece.color == chess.WHITE else BLACK


def apply_primitive(pieces: dict[int, chess.Piece], p: PickPlace) -> dict[int, chess.Piece]:
    out = dict(pieces)
    if is_square(p.src):
        out.pop(chess.parse_square(p.src), None)
    if is_square(p.dst):
        out[chess.parse_square(p.dst)] = p.piece
    return out


def occupancy_of(pieces: dict[int, chess.Piece]) -> list[int]:
    occ = [EMPTY] * 64
    for sq, piece in pieces.items():
        occ[sq] = _colour_code(piece)
    return occ


def plan_correction(p: PickPlace, observed: list[int], expected: list[int]) -> PickPlace | None:
    """Work out how to recover from a failed primitive using only occupancy."""
    colour = _colour_code(p.piece)
    if is_square(p.dst) and observed[chess.parse_square(p.dst)] not in (EMPTY, expected[chess.parse_square(p.dst)]):
        return None  # something unexpected is on the target; needs a human
    if is_square(p.src):
        src = chess.parse_square(p.src)
        if observed[src] == colour and expected[src] == EMPTY:
            return p  # grasp missed: piece still at source, try again
    extras = [sq for sq in range(64) if observed[sq] == colour and expected[sq] == EMPTY]
    if not extras and not is_square(p.src):
        return p  # reserve piece never arrived: retry
    if len(extras) == 1:
        return PickPlace(p.piece, chess.square_name(extras[0]), p.dst, "correction")
    return None


class GameOrchestrator:
    def __init__(
        self,
        engine: ChessEngine,
        planner: MovePlanner,
        robot: RobotController,
        detector: MoveDetector,
        read_frame: Callable[[], np.ndarray],
        human: HumanPlayer,
        assist: AssistHandler,
        robot_color: chess.Color = chess.BLACK,
        metrics: MetricsLogger | None = None,
        max_retries: int = 2,
        human_move_timeout_s: float | None = None,
        poll_s: float = 0.2,
        notify: Callable[[str], None] = print,
        board: chess.Board | None = None,
    ):
        self.engine = engine
        self.planner = planner
        self.robot = robot
        self.detector = detector
        self.read_frame = read_frame
        self.human = human
        self.assist = assist
        self.robot_color = robot_color
        self.metrics = metrics or MetricsLogger(log_dir=None)
        self.max_retries = max_retries
        self.human_move_timeout_s = human_move_timeout_s
        self.poll_s = poll_s
        self.notify = notify
        self.board = board or chess.Board()

    # --- turns -----------------------------------------------------------------

    def human_turn(self) -> MoveRecord:
        truth = self.human.take_turn(self.board)
        unmatched = 0

        def on_unmatched(det: Detection) -> None:
            nonlocal unmatched
            unmatched += 1
            squares = ", ".join(chess.square_name(s) for s in det.changed_squares)
            self.notify(f"Board doesn't match a legal move (changed: {squares}). Please check.")

        det = self.detector.wait_for_move(
            self.board, self.read_frame, self.human_move_timeout_s, self.poll_s, on_unmatched
        )
        assert det.move is not None
        rec = MoveRecord(
            ply=self.board.ply(),
            side="human",
            uci=det.move.uci(),
            san=self.board.san(det.move),
            detection_latency_s=round(det.latency_s, 4),
            detection_correct=None if truth is None else det.move == truth,
            detection_ambiguous=det.ambiguous,
            unmatched_readings=unmatched,
        )
        self.board.push(det.move)
        return rec

    def robot_turn(self) -> MoveRecord:
        res = self.engine.choose_move(self.board)
        san = self.board.san(res.move)
        self.notify(f"Robot plays {san}")
        prims = self.planner.plan(self.board, res.move)
        rec = MoveRecord(
            ply=self.board.ply(), side="robot", uci=res.move.uci(), san=san,
            engine_time_s=round(res.think_time_s, 4), engine_source=res.source,
            score_cp=res.score_cp, primitives=len(prims),
        )
        start = time.perf_counter()
        pieces = dict(self.board.piece_map())
        for p in prims:
            pieces = apply_primitive(pieces, p)
            self._execute_verified(p, pieces, rec)
        self.robot.home()
        rec.execution_time_s = round(time.perf_counter() - start, 4)
        self.board.push(res.move)
        final = self.detector.read_stable(self.read_frame)
        rec.verified = not diff_squares(occupancy_of(self.board.piece_map()), final)
        return rec

    def _execute_verified(self, p: PickPlace, expected_pieces: dict[int, chess.Piece], rec: MoveRecord) -> None:
        expected = occupancy_of(expected_pieces)
        current: PickPlace | None = p
        retries = 0
        while True:
            assert current is not None
            result = self.robot.execute(current)
            rec.attempts += 1
            if "failure" in result.info:
                rec.failures.append(result.info["failure"])
            self.robot.home()
            observed = self.detector.read_stable(self.read_frame)
            if observed == expected:
                return
            log.info("Primitive %s not verified (%s)", current.describe(), result.info)
            current = plan_correction(p, observed, expected) if retries < self.max_retries else None
            if current is None:
                mismatched = diff_squares(expected, observed)
                self.notify(
                    "I need a hand: please fix "
                    + ", ".join(chess.square_name(s) for s in mismatched)
                )
                self.assist(expected_pieces, mismatched)
                rec.human_assists += 1
                if self.detector.read_stable(self.read_frame) == expected:
                    return
                raise RuntimeError("Board still inconsistent after human assistance")
            else:
                retries += 1
                rec.corrections += 1

    # --- game loop -------------------------------------------------------------

    def play(self, max_plies: int | None = None) -> GameSummary:
        while not self.board.is_game_over(claim_draw=True):
            if max_plies is not None and self.board.ply() >= max_plies:
                return self.metrics.summary("*", "max_plies")
            if self.board.turn == self.robot_color:
                rec = self.robot_turn()
            else:
                rec = self.human_turn()
            self.metrics.log(rec)
        outcome = self.board.outcome(claim_draw=True)
        assert outcome is not None
        self.notify(f"Game over: {outcome.result()} ({outcome.termination.name.lower()})")
        return self.metrics.summary(outcome.result(), outcome.termination.name.lower())
