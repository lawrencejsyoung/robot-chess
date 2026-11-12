"""Dashboard integration tests exercise the real simulation and vision stack."""
import chess
import pytest

pytest.importorskip('flask')  # Dashboard is an optional extra.
from robot_chess.web import create_app


@pytest.fixture
def app(sim_cfg):
    application = create_app(sim_cfg)
    application.testing = True
    yield application
    application.extensions['chess_game'].orchestrator.engine.close()


@pytest.fixture
def client(app):
    return app.test_client()


def test_roundtrip_uses_vision_and_verified_execution(app, client):
    assert client.get('/').status_code == 200
    start = client.get('/api/state').json
    assert len(start['legal_moves']) == 20
    assert client.post('/api/robot').status_code == 409
    assert client.post('/api/move', json={'move': 'e2e5'}).status_code == 400
    human = client.post('/api/move', json={'move': 'e2e4'})
    assert human.status_code == 200
    assert human.json['last_record']['detection_correct'] is True
    assert human.json['last_record']['side'] == 'human'
    assert client.post('/api/move', json={'move': 'd2d4'}).status_code == 409
    robot = client.post('/api/robot')
    assert robot.status_code == 200
    data = robot.json
    assert data['turn'] == 'white'
    assert len(data['history']) == 2
    assert data['last_record']['verified'] is True
    assert data['metrics']['robot_autonomous_rate'] == 1.0
    assert len(data['plan']) == data['last_record']['primitives']
    assert app.extensions['chess_game'].physical.matches(chess.Board(data['fen']))


def test_observation_is_read_only(client):
    b = chess.Board()
    assert client.post('/api/observe', json={'fen': b.fen()}).json['synchronized']
    b.push_uci('e2e4')
    assert client.post('/api/observe', json={'fen': b.fen()}).json['matches'] == ['e2e4']
    assert client.get('/api/state').json['history'] == []
    assert client.post('/api/observe', json={'fen': 'bad'}).status_code == 400


@pytest.mark.parametrize('value', [None, [], {'move': 1}, {'move': None}, {}])
def test_bad_move_payload(client, value):
    assert client.post('/api/move', json=value).status_code == 400
    assert client.get('/api/state').json['history'] == []


def test_camera_and_packaged_assets(client):
    camera = client.get('/api/camera')
    assert camera.mimetype == 'image/png'
    assert camera.data.startswith(b'\x89PNG\r\n\x1a\n')
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/static/style.css').status_code == 200


def test_reset_clears_game_metrics_and_closes_old_engine(app, client, monkeypatch):
    client.post('/api/move', json={'move': 'e2e4'})
    old = app.extensions['chess_game']
    closed = []
    monkeypatch.setattr(old.orchestrator.engine, 'close', lambda: closed.append(True))
    data = client.post('/api/reset').json
    assert closed == [True]
    assert data['fen'] == chess.Board().fen()
    assert data['history'] == [] and data['plan'] == []
    assert data['metrics']['plies'] == 0
    assert app.extensions['chess_game'] is not old


def test_execution_failure_pauses_until_reset(app, client, monkeypatch):
    client.post('/api/move', json={'move': 'e2e4'})
    def fail(*args):
        raise RuntimeError('simulated driver failure')
    monkeypatch.setattr(app.extensions['chess_game'].robot, 'execute', fail)
    failed = client.post('/api/robot')
    assert failed.status_code == 409
    assert failed.json['state']['fault']
    assert failed.json['state']['legal_moves'] == []
    assert client.post('/api/robot').status_code == 409
    assert client.post('/api/reset').json['fault'] is None


def test_failure_injection_uses_existing_assist_and_metrics(sim_cfg):
    sim_cfg.robot.sim_failure_rate = 1.0
    sim_cfg.robot.max_retries = 1
    app = create_app(sim_cfg)
    try:
        client = app.test_client()
        assert client.post('/api/move', json={'move': 'e2e4'}).status_code == 200
        data = client.post('/api/robot').json
        assert data['last_record']['verified'] is True
        assert data['metrics']['human_assists'] > 0
        assert data['metrics']['robot_autonomous_rate'] == 0
        assert app.extensions['chess_game'].physical.matches(chess.Board(data['fen']))
    finally:
        app.extensions['chess_game'].orchestrator.engine.close()


def test_dashboard_forces_simulation_without_mutating_config(sim_cfg):
    sim_cfg.robot.backend = 'groot'
    sim_cfg.game.robot_color = 'white'
    app = create_app(sim_cfg)
    try:
        assert app.test_client().get('/api/state').json['mode'] == 'simulation'
        assert app.extensions['chess_game'].orchestrator.robot_color == chess.BLACK
        assert sim_cfg.robot.backend == 'groot'
        assert sim_cfg.game.robot_color == 'white'
    finally:
        app.extensions['chess_game'].orchestrator.engine.close()


def test_unverified_robot_result_pauses_game(app, client, monkeypatch):
    from robot_chess.metrics import MoveRecord
    client.post('/api/move', json={'move': 'e2e4'})
    orch = app.extensions['chess_game'].orchestrator
    monkeypatch.setattr(orch, 'robot_turn', lambda: MoveRecord(
        ply=1, side='robot', uci='e7e5', san='e5', verified=False))
    result = client.post('/api/robot')
    assert result.status_code == 409
    assert result.json['state']['fault']


def test_underpromotion_rejected_before_physical_mutation(app, client):
    game = app.extensions['chess_game']
    game.orchestrator.board = chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    before = game.physical.copy()
    result = client.post('/api/move', json={'move': 'a7a8n'})
    assert result.status_code == 400
    assert 'queen promotion' in result.json['error']
    assert game.physical.squares == before.squares
    assert game.orchestrator.board.piece_at(chess.A7) == chess.Piece(chess.PAWN, chess.WHITE)
