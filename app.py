"""Physical chess coordination prototype. Hardware execution is intentionally separate."""
import os
import threading
import chess
import chess.engine
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
board = chess.Board()
lock = threading.Lock()
history = []
last_plan = []
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def evaluate(position):
    if position.is_checkmate():
        return -100000 if position.turn else 100000
    if position.is_game_over():
        return 0
    return sum((1 if p.color else -1) * VALUES[p.piece_type]
               for p in position.piece_map().values())


def search(position, depth, alpha=-float('inf'), beta=float('inf')):
    if depth == 0 or position.is_game_over():
        return evaluate(position)
    maximizing = position.turn == chess.WHITE
    score = -float('inf') if maximizing else float('inf')
    for move in sorted(position.legal_moves, key=position.is_capture, reverse=True):
        position.push(move)
        value = search(position, depth - 1, alpha, beta)
        position.pop()
        score = max(score, value) if maximizing else min(score, value)
        if maximizing:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        if beta <= alpha:
            break
    return score


def choose_move(position):
    executable = os.getenv('STOCKFISH_PATH')
    if executable:
        with chess.engine.SimpleEngine.popen_uci(executable) as engine:
            return engine.play(position, chess.engine.Limit(time=0.3)).move
    scored = []
    for move in position.legal_moves:
        position.push(move)
        scored.append((search(position, 2), move))
        position.pop()
    return (max if position.turn else min)(scored, key=lambda item: item[0])[1]


def plan_move(position, move):
    """Symbolic tasks for a calibrated embodied policy; never raw joint commands."""
    tasks = []
    if position.is_capture(move):
        captured = move.to_square
        if position.is_en_passant(move):
            captured += -8 if position.turn else 8
        tasks.append({'action': 'remove', 'from': chess.square_name(captured), 'to': 'capture_tray'})
    tasks.append({'action': 'transfer', 'from': chess.square_name(move.from_square),
                  'to': chess.square_name(move.to_square)})
    if position.is_castling(move):
        rank = chess.square_rank(move.from_square)
        king_side = chess.square_file(move.to_square) > 4
        tasks.append({'action': 'transfer', 'from': chess.square_name(chess.square(7 if king_side else 0, rank)),
                      'to': chess.square_name(chess.square(5 if king_side else 3, rank))})
    if move.promotion:
        tasks.append({'action': 'replace', 'square': chess.square_name(move.to_square),
                      'piece': chess.piece_name(move.promotion)})
    tasks.append({'action': 'verify', 'expected_fen': after_move(position, move)})
    return tasks


def after_move(position, move):
    result = position.copy()
    result.push(move)
    return result.fen()


def state():
    return {'fen': board.fen(), 'pieces': {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
            'turn': 'white' if board.turn else 'black', 'legal_moves': [m.uci() for m in board.legal_moves],
            'history': history, 'plan': last_plan, 'check': board.is_check(),
            'outcome': board.outcome().result() if board.is_game_over() else None,
            'engine': 'Stockfish' if os.getenv('STOCKFISH_PATH') else 'Local alpha-beta · depth 3',
            'mode': 'simulation', 'robot': 'GR00T 2 adapter contract · disconnected'}


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/state')
def get_state():
    with lock:
        return jsonify(state())


@app.post('/api/reset')
def reset():
    global board, last_plan
    with lock:
        board = chess.Board()
        history.clear()
        last_plan = []
        return jsonify(state())


@app.post('/api/move')
def move():
    global last_plan
    data = request.get_json(silent=True) or {}
    with lock:
        if board.turn != chess.WHITE or board.is_game_over():
            return jsonify(error='Wait for the robot or start a new game.'), 409
        try:
            candidate = chess.Move.from_uci(data.get('move', ''))
            if candidate not in board.legal_moves:
                raise ValueError()
        except (ValueError, TypeError):
            return jsonify(error='Choose a legal move.'), 400
        history.append({'side': 'white', 'san': board.san(candidate), 'uci': candidate.uci()})
        board.push(candidate)
        last_plan = []
        return jsonify(state())


@app.post('/api/robot')
def robot():
    global last_plan
    with lock:
        if board.turn != chess.BLACK or board.is_game_over():
            return jsonify(error='The robot has no move to make.'), 409
        candidate = choose_move(board)
        last_plan = plan_move(board, candidate)
        history.append({'side': 'black', 'san': board.san(candidate), 'uci': candidate.uci()})
        board.push(candidate)
        return jsonify(state())


@app.post('/api/observe')
def observe():
    """Validate board occupancy from an external vision system without trusting moves."""
    data = request.get_json(silent=True) or {}
    try:
        observed = chess.Board(data.get('fen', ''))
        if not observed.is_valid():
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify(error='Provide a valid full board FEN.'), 400
    with lock:
        matches = [m.uci() for m in board.legal_moves
                   if chess.Board(after_move(board, m)).board_fen() == observed.board_fen()]
        return jsonify(matches=matches, synchronized=observed.board_fen() == board.board_fen(),
                       message='Observation only. Submit a unique legal move to advance the game.')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False)
