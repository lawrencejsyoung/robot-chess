"""Wiring for fully simulated games (no hardware): used by tests and benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from .config import Config, EngineConfig
from .engine import ChessEngine
from .game import GameOrchestrator
from .metrics import MetricsLogger
from .physical import PhysicalBoard
from .robot.planner import MovePlanner
from .robot.sim import SimRobot
from .vision.calibration import BoardCalibration
from .vision.camera import SimCamera
from .vision.move_detector import MoveDetector
from .vision.occupancy import OccupancyClassifier


class SimHuman:
    """A scripted opponent that physically moves pieces on the simulated board.

    ``style``: "random" or "engine" (built-in searcher at ``depth``).
    """

    def __init__(self, physical: PhysicalBoard, planner: MovePlanner, style: str = "random",
                 seed: int | None = None, depth: int = 1):
        self.physical = physical
        self.planner = planner
        self.style = style
        self.rng = random.Random(seed)
        self.engine = ChessEngine(EngineConfig(uci_path=None, fallback_depth=depth)) if style == "engine" else None

    def take_turn(self, board: chess.Board) -> chess.Move:
        if self.engine:
            move = self.engine.choose_move(board).move
        else:
            move = self.rng.choice(list(board.legal_moves))
        for p in self.planner.plan(board, move):
            piece = self.physical.take(p.src, p.piece) or p.piece
            self.physical.put(p.dst, piece)
        return move


class SimAssist:
    """Plays the role of a helpful human who tidies up the robot's mistakes."""

    def __init__(self, physical: PhysicalBoard):
        self.physical = physical
        self.calls = 0

    def __call__(self, expected: dict[int, chess.Piece], mismatched: list[int]) -> None:
        self.calls += 1
        self.physical.squares = {chess.square_name(sq): p for sq, p in expected.items()}


@dataclass
class SimGame:
    orchestrator: GameOrchestrator
    physical: PhysicalBoard
    robot: SimRobot
    camera: SimCamera


def build_sim_game(
    cfg: Config | None = None,
    human_style: str = "random",
    seed: int | None = None,
    log_dir: str | None = None,
    game_name: str | None = None,
    auto_calibrate: bool = True,
) -> SimGame:
    cfg = cfg or Config()
    board = chess.Board()
    physical = PhysicalBoard.from_board(board, cfg.robot.reserve)
    camera = SimCamera(
        lambda: {chess.parse_square(n): p for n, p in physical.squares.items()},
        seed=seed, board_px=cfg.vision.warp_size_px,
    )
    # Calibrate exactly as on hardware: from a frame of the empty board.
    empty = camera.render({})
    cal = (BoardCalibration.from_empty_board(empty, cfg.vision.warp_size_px)
           if auto_calibrate else camera.true_calibration)
    classifier = OccupancyClassifier.from_camera_frame(
        cal, empty, roi_fraction=cfg.vision.roi_fraction, diff_threshold=cfg.vision.occupancy_threshold
    )
    detector = MoveDetector(classifier.classify, stable_frames=cfg.vision.stable_frames)
    planner = MovePlanner(cfg.robot)
    robot = SimRobot(physical, cfg.robot.sim_failure_rate, seed if cfg.robot.sim_seed is None else cfg.robot.sim_seed)
    human = SimHuman(physical, planner, human_style, seed=None if seed is None else seed + 1)
    robot_color = chess.WHITE if cfg.game.robot_color == "white" else chess.BLACK
    orch = GameOrchestrator(
        engine=ChessEngine(cfg.engine),
        planner=planner,
        robot=robot,
        detector=detector,
        read_frame=camera.read,
        human=human,
        assist=SimAssist(physical),
        robot_color=robot_color,
        metrics=MetricsLogger(log_dir, game_name),
        max_retries=cfg.robot.max_retries,
        poll_s=0.0,
        notify=lambda msg: None,
        board=board,
    )
    return SimGame(orch, physical, robot, camera)
