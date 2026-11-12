"""Simulation dashboard using the same vision and execution loop as the CLI."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from threading import Lock

import chess
import cv2
from flask import Flask, Response, jsonify, render_template, request

from ..config import Config
from ..robot.planner import MovePlanner
from ..physical import occupancy_from_board
from ..simulation import build_sim_game


class DashboardHuman:
    """Apply the submitted move to the simulated physical board for vision to read."""

    def __init__(self, game):
        self.game = game
        self.pending: chess.Move | None = None

    def take_turn(self, board):
        move = self.pending
        if move is None or move not in board.legal_moves:
            raise ValueError('No legal human move submitted')
        self.pending = None
        for primitive in self.game.orchestrator.planner.plan(board, move):
            piece = self.game.physical.take(primitive.src, primitive.piece)
            if piece is None:
                raise RuntimeError('A physical piece is missing; reset the simulation')
            self.game.physical.put(primitive.dst, piece)
        return move


class DashboardPlanner(MovePlanner):
    """Expose the existing planner's most recent primitives to the dashboard."""

    def plan(self, board, move):
        primitives = super().plan(board, move)
        self.last_plan = [{'action': 'transfer', 'from': p.src, 'to': p.dst,
                           'piece': p.piece.symbol(), 'reason': p.reason} for p in primitives]
        return primitives


def create_app(config: Config | None = None):
    app = Flask(__name__)
    cfg = deepcopy(config or Config())
    # The dashboard controls only a simulation with the human playing white.
    cfg.robot.backend = 'sim'
    cfg.game.robot_color = 'black'
    lock = Lock()
    game = None
    plan = []
    fault = None

    def new_game():
        nonlocal game, plan, fault
        replacement = build_sim_game(cfg, seed=0)
        replacement.orchestrator.planner = DashboardPlanner(cfg.robot)
        replacement.orchestrator.human = DashboardHuman(replacement)
        replacement.orchestrator.human_move_timeout_s = 2.0
        if game is not None:
            game.orchestrator.engine.close()
        game = replacement
        plan = []
        fault = None
        app.extensions['chess_game'] = game

    def snapshot():
        orch = game.orchestrator
        board = orch.board
        outcome = board.outcome(claim_draw=True)
        records = orch.metrics.records
        return {
            'fen': board.fen(),
            'pieces': {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
            'turn': 'white' if board.turn else 'black',
            'legal_moves': [] if outcome or fault else [m.uci() for m in board.legal_moves],
            'history': [{'side': 'white' if r.side == 'human' else 'black', 'san': r.san,
                         'uci': r.uci} for r in records],
            'plan': plan, 'check': board.is_check(),
            'outcome': outcome.result() if outcome else None,
            'engine': 'Stockfish / UCI' if orch.engine.source == 'uci' else f'Built-in engine · depth {cfg.engine.fallback_depth}',
            'mode': 'simulation', 'robot': 'SimRobot · vision-verified', 'fault': fault,
            'metrics': orch.metrics.summary(outcome.result() if outcome else '*',
                                           outcome.termination.name.lower() if outcome else 'in_progress').to_dict(),
            'last_record': asdict(records[-1]) if records else None,
        }

    def payload():
        value = request.get_json(silent=True)
        return value if isinstance(value, dict) else {}

    def execute_turn(turn):
        nonlocal fault, plan
        try:
            record = turn()
            game.orchestrator.metrics.log(record)
            if record.side == 'robot' and not record.verified:
                raise RuntimeError('Final vision verification failed')
            if record.side == 'robot':
                plan = game.orchestrator.planner.last_plan
            return jsonify(snapshot())
        except Exception:
            # A failed step may have moved a piece. Never retry blindly.
            app.logger.exception('Dashboard simulation turn failed')
            fault = 'Simulation verification failed. Reset the game before continuing.'
            return jsonify(error=fault, state=snapshot()), 409

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/state')
    def state():
        with lock:
            return jsonify(snapshot())

    @app.get('/api/camera')
    def camera():
        with lock:
            ok, encoded = cv2.imencode('.png', game.camera.read())
            if not ok:
                return jsonify(error='Could not render simulated camera'), 500
            return Response(encoded.tobytes(), mimetype='image/png', headers={'Cache-Control': 'no-store'})

    @app.post('/api/reset')
    def reset():
        with lock:
            new_game()
            return jsonify(snapshot())

    @app.post('/api/move')
    def move():
        nonlocal plan
        with lock:
            board = game.orchestrator.board
            if fault or board.turn != chess.WHITE or board.is_game_over(claim_draw=True):
                return jsonify(error=fault or 'Wait for the robot or start a new game.'), 409
            try:
                candidate = chess.Move.from_uci(payload().get('move', ''))
                if candidate not in board.legal_moves:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError):
                return jsonify(error='Choose a legal UCI move.'), 400
            if candidate.promotion and candidate.promotion != chess.QUEEN:
                return jsonify(error='The dashboard vision path supports queen promotion only.'), 400
            game.orchestrator.human.pending = candidate
            plan = []
            return execute_turn(game.orchestrator.human_turn)

    @app.post('/api/robot')
    def robot():
        with lock:
            orch = game.orchestrator
            if fault or orch.board.turn != chess.BLACK or orch.board.is_game_over(claim_draw=True):
                return jsonify(error=fault or 'The robot has no move to make.'), 409
            return execute_turn(orch.robot_turn)

    @app.post('/api/observe')
    def observe():
        try:
            observed = chess.Board(payload().get('fen', ''))
            if not observed.is_valid():
                raise ValueError()
        except (ValueError, TypeError):
            return jsonify(error='Provide a valid full board FEN.'), 400
        with lock:
            board = game.orchestrator.board
            occupancy = occupancy_from_board(observed)
            detection = game.orchestrator.detector.match(board, occupancy)
            return jsonify(matches=[m.uci() for m in detection.candidates],
                           synchronized=occupancy == occupancy_from_board(board),
                           ambiguous=detection.ambiguous,
                           message='Color occupancy comparison only; observation does not advance the game.')

    new_game()
    return app


def main():
    import argparse
    from ..config import load_config
    parser = argparse.ArgumentParser(description='Robot Chess simulation dashboard')
    parser.add_argument('--config', help='Existing YAML configuration (engine and simulation settings)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app = create_app(load_config(args.config))
    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        app.extensions['chess_game'].orchestrator.engine.close()
