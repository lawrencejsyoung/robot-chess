import chess
import pytest
from app import app, choose_move, plan_move

@pytest.fixture
def client():
    with app.test_client() as c:
        c.post('/api/reset')
        yield c


def test_playable_roundtrip(client):
    assert client.get('/').status_code == 200
    assert client.post('/api/move', json={'move': 'e2e5'}).status_code == 400
    r = client.post('/api/move', json={'move': 'e2e4'})
    assert r.json['turn'] == 'black'
    assert client.post('/api/move', json={'move': 'd2d4'}).status_code == 409
    r = client.post('/api/robot')
    assert r.status_code == 200
    assert r.json['turn'] == 'white'
    assert len(r.json['history']) == 2
    assert r.json['plan'][-1]['expected_fen'] == r.json['fen']
    assert client.post('/api/robot').status_code == 409
    assert client.post('/api/reset').json['history'] == []


def test_vision_observation(client):
    b = chess.Board()
    assert client.post('/api/observe', json={'fen': b.fen()}).json['synchronized']
    b.push_uci('e2e4')
    assert client.post('/api/observe', json={'fen': b.fen()}).json['matches'] == ['e2e4']
    assert client.post('/api/observe', json={'fen': 'bad'}).status_code == 400
    assert client.get('/api/state').json['turn'] == 'white'


def test_capture_and_en_passant():
    b = chess.Board()
    for uci in ['e2e4', 'a7a6', 'e4e5', 'd7d5']:
        b.push_uci(uci)
    plan = plan_move(b, chess.Move.from_uci('e5d6'))
    assert plan[0] == {'action': 'remove', 'from': 'd5', 'to': 'capture_tray'}
    b = chess.Board('4k3/8/8/8/8/8/4r3/4K3 w - - 0 1')
    assert plan_move(b, chess.Move.from_uci('e1e2'))[0]['from'] == 'e2'


def test_castling_and_promotion():
    b = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert plan_move(b, chess.Move.from_uci('e1g1'))[1] == {'action': 'transfer', 'from': 'h1', 'to': 'f1'}
    b = chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    assert plan_move(b, chess.Move.from_uci('a7a8q'))[1]['piece'] == 'queen'


def test_engine_chooses_legal_move_without_mutation():
    b = chess.Board()
    b.push_uci('e2e4')
    before = b.fen()
    assert choose_move(b) in b.legal_moves
    assert b.fen() == before
