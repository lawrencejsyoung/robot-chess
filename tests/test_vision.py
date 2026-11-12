import chess
import numpy as np
import pytest

from robot_chess.physical import occupancy_from_board
from robot_chess.vision import BoardCalibration, MoveDetector, OccupancyClassifier, SimCamera


def make(board, seed=0, corners=None):
    cam = SimCamera(lambda: board.piece_map(), seed=seed, corners=corners)
    empty = cam.render({})
    cal = BoardCalibration.from_empty_board(empty)
    clf = OccupancyClassifier.from_camera_frame(cal, empty)
    return cam, cal, clf


def test_auto_calibration_recovers_corners():
    cam, cal, _ = make(chess.Board())
    assert np.abs(cal.corners - cam.corners).max() < 2.0


def test_calibration_with_other_viewpoint():
    corners = np.array([[200, 640], [760, 600], [700, 120], [240, 80]], dtype=np.float32)
    cam, cal, _ = make(chess.Board(), corners=corners)
    assert np.abs(cal.corners - corners).max() < 2.0


def test_calibration_roundtrip(tmp_path):
    cal = BoardCalibration(np.arange(8, dtype=np.float32).reshape(4, 2), 320)
    cal.save(tmp_path / "c.json")
    loaded = BoardCalibration.load(tmp_path / "c.json")
    assert loaded.size == 320 and np.allclose(loaded.corners, cal.corners)


@pytest.mark.parametrize("seed", range(3))
def test_occupancy_matches_position(seed):
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    cam, _, clf = make(board, seed)
    assert clf.classify(cam.read()) == occupancy_from_board(board)


@pytest.mark.parametrize(
    "fen,uci",
    [
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5"),
        ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
    ],
)
def test_detects_moves_including_special(fen, uci):
    board = chess.Board(fen)
    shown = board.copy()
    cam, _, clf = make(shown)
    move = chess.Move.from_uci(uci)
    shown.push(move)
    det = MoveDetector(clf.classify, stable_frames=2).wait_for_move(board, cam.read, timeout_s=2, poll_s=0)
    assert det.move == move


def test_unmatched_board_reported_then_timeout():
    board = chess.Board()
    shown = board.copy()
    shown.remove_piece_at(chess.E2)  # a pawn vanished: not a legal move
    cam, _, clf = make(shown)
    seen = []
    with pytest.raises(TimeoutError):
        MoveDetector(clf.classify, 1).wait_for_move(board, cam.read, timeout_s=0.05, poll_s=0.01,
                                                   on_unmatched=seen.append)
    assert seen and seen[0].changed_squares == [chess.E2]
