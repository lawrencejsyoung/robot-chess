"""Command-line entry point: ``robot-chess {play,calibrate,benchmark}``."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import chess
import cv2

from .config import Config, load_config


# --- helpers -------------------------------------------------------------------

class ConsoleHuman:
    """Real human at a real board: we only prompt, vision finds the move."""

    def take_turn(self, board: chess.Board) -> None:
        print(f"\n{board}\nYour move ({'white' if board.turn else 'black'}). Make it on the board.")
        return None


class ConsoleAssist:
    def __call__(self, expected, mismatched) -> None:
        target = chess.Board(None)
        for sq, p in expected.items():
            target.set_piece_at(sq, p)
        print("Please set these squares as shown, then press Enter:",
              ", ".join(chess.square_name(s) for s in mismatched))
        print(target)
        input()


class TypedSimHuman:
    """Simulator with a human typing moves (SAN or UCI) in the terminal."""

    def __init__(self, sim_human):
        self.sim = sim_human

    def take_turn(self, board: chess.Board) -> chess.Move:
        print(f"\n{board}")
        while True:
            text = input("Your move: ").strip()
            try:
                move = board.parse_san(text)
            except ValueError:
                try:
                    move = chess.Move.from_uci(text)
                except ValueError:
                    move = None
            if move is not None and move in board.legal_moves:
                break
            print("Illegal or unparseable move, try again.")
        for p in self.sim.planner.plan(board, move):
            piece = self.sim.physical.take(p.src, p.piece) or p.piece
            self.sim.physical.put(p.dst, piece)
        return move


def _load_class(path: str):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _print_summary(d: dict) -> None:
    width = max(len(k) for k in d)
    for k, v in d.items():
        print(f"  {k:<{width}}  {v}")


# --- commands --------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace, cfg: Config) -> int:
    from .metrics import aggregate
    from .simulation import build_sim_game

    cfg.robot.sim_failure_rate = args.failure_rate
    base_color = cfg.game.robot_color
    out_dir = Path(cfg.game.log_dir) / args.name
    summaries = []
    for i in range(args.games):
        flip = {"white": "black", "black": "white"}
        cfg.game.robot_color = flip[base_color] if (args.alternate and i % 2) else base_color
        g = build_sim_game(cfg, human_style=args.human, seed=args.seed + i,
                           log_dir=str(out_dir), game_name=f"game-{i:03d}")
        s = g.orchestrator.play(max_plies=args.max_plies)
        summaries.append(s)
        print(f"game {i}: {s.result:>7} {s.termination:<22} robot first-try "
              f"{s.robot_first_try_rate}  autonomous {s.robot_autonomous_rate}  "
              f"detect {s.detection_accuracy}")
    report = aggregate(summaries)
    report["failure_rate"] = args.failure_rate
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(f"\nBenchmark ({out_dir / 'benchmark.json'}):")
    _print_summary(report)
    return 0


def cmd_calibrate(args: argparse.Namespace, cfg: Config) -> int:
    from .vision.calibration import BoardCalibration

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}", file=sys.stderr)
            return 1
    else:
        from .vision.camera import OpenCVCamera
        cam = OpenCVCamera(cfg.vision.camera_index)
        for _ in range(10):  # let auto-exposure settle
            frame = cam.read()
        cam.close()
    if args.corners:
        import numpy as np
        pts = np.array([float(v) for v in args.corners.split(",")], dtype=np.float32).reshape(4, 2)
        cal = BoardCalibration(pts, cfg.vision.warp_size_px)
    else:
        cal = BoardCalibration.from_empty_board(frame, cfg.vision.warp_size_px, args.rotate)
    cal.save(cfg.vision.calibration_file)
    warped = cal.warp(frame)
    cv2.imwrite(cfg.vision.empty_reference, warped)
    preview = warped.copy()
    s = cal.square_px
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        cv2.putText(preview, chess.square_name(sq), (int(f * s + 4), int((8 - r) * s - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, s / 160, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(args.preview, preview)
    print(f"Saved {cfg.vision.calibration_file}, {cfg.vision.empty_reference}; "
          f"check square labels in {args.preview} (a1 must be bottom-left; "
          f"otherwise re-run with --rotate 1..3)")
    return 0


def cmd_play(args: argparse.Namespace, cfg: Config) -> int:
    from .metrics import MetricsLogger

    if args.sim:
        from .simulation import build_sim_game

        g = build_sim_game(cfg, log_dir=cfg.game.log_dir)
        g.orchestrator.human = TypedSimHuman(g.orchestrator.human)
        g.orchestrator.notify = print
        s = g.orchestrator.play()
        _print_summary(s.to_dict())
        return 0

    from .engine import ChessEngine
    from .game import GameOrchestrator
    from .robot.groot_client import GrootRobot, ZmqPolicyTransport
    from .robot.planner import MovePlanner
    from .vision.calibration import BoardCalibration
    from .vision.camera import OpenCVCamera
    from .vision.move_detector import MoveDetector
    from .vision.occupancy import OccupancyClassifier

    gc = cfg.robot.groot
    if not gc.robot_io:
        print("Set robot.groot.robot_io to your RobotIO driver class (module:Class)", file=sys.stderr)
        return 1
    cal = BoardCalibration.load(cfg.vision.calibration_file)
    ref = cv2.imread(cfg.vision.empty_reference)
    clf = OccupancyClassifier(cal, ref, cfg.vision.roi_fraction, cfg.vision.occupancy_threshold)
    cam = OpenCVCamera(cfg.vision.camera_index)
    robot = GrootRobot(
        _load_class(gc.robot_io)(**gc.robot_io_kwargs),
        ZmqPolicyTransport(gc.host, gc.port, gc.timeout_ms),
        gc,
    )
    orch = GameOrchestrator(
        engine=ChessEngine(cfg.engine),
        planner=MovePlanner(cfg.robot),
        robot=robot,
        detector=MoveDetector(clf.classify, cfg.vision.stable_frames),
        read_frame=cam.read,
        human=ConsoleHuman(),
        assist=ConsoleAssist(),
        robot_color=chess.WHITE if cfg.game.robot_color == "white" else chess.BLACK,
        metrics=MetricsLogger(cfg.game.log_dir),
        max_retries=cfg.robot.max_retries,
    )
    try:
        s = orch.play()
        _print_summary(s.to_dict())
    finally:
        robot.close()
        cam.close()
        orch.engine.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="robot-chess", description=__doc__)
    ap.add_argument("--config", "-c", help="YAML config file")
    ap.add_argument("--verbose", "-v", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="play a game")
    p.add_argument("--sim", action="store_true", help="simulated board/robot; type your moves")

    c = sub.add_parser("calibrate", help="locate the empty board and save a reference image")
    c.add_argument("--image", help="calibrate from an image file instead of the camera")
    c.add_argument("--rotate", type=int, default=0, choices=range(4), help="quarter turns to fix orientation")
    c.add_argument("--corners", help="manual outer corners a1,h1,h8,a8 as 'x1,y1,...,x4,y4'")
    c.add_argument("--preview", default="calibration_preview.png")

    b = sub.add_parser("benchmark", help="run simulated games and report metrics")
    b.add_argument("--games", type=int, default=5)
    b.add_argument("--failure-rate", type=float, default=0.1, help="per-primitive robot failure probability")
    b.add_argument("--human", choices=["random", "engine"], default="random")
    b.add_argument("--max-plies", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--alternate", action="store_true", help="alternate robot colour between games")
    b.add_argument("--name", default="benchmark")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    cfg = load_config(args.config)
    return {"play": cmd_play, "calibrate": cmd_calibrate, "benchmark": cmd_benchmark}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
