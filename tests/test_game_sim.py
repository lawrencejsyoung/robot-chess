import chess

from robot_chess.game import plan_correction
from robot_chess.metrics import aggregate
from robot_chess.physical import BLACK, EMPTY, WHITE
from robot_chess.robot.planner import PickPlace
from robot_chess.simulation import build_sim_game

KNIGHT = chess.Piece(chess.KNIGHT, chess.WHITE)


def test_perfect_robot_game(sim_cfg, tmp_path):
    g = build_sim_game(sim_cfg, seed=0, log_dir=str(tmp_path), game_name="g")
    s = g.orchestrator.play(max_plies=16)
    assert 0 < s.plies <= 16 and s.robot_moves == s.plies // 2
    assert s.robot_first_try_rate == 1.0 and s.detection_accuracy == 1.0
    assert g.physical.matches(g.orchestrator.board)
    assert len((tmp_path / "g.jsonl").read_text().splitlines()) == s.plies
    assert (tmp_path / "g.summary.json").exists()


def test_failures_are_caught_and_corrected(sim_cfg):
    sim_cfg.robot.sim_failure_rate = 0.4
    g = build_sim_game(sim_cfg, seed=3)
    s = g.orchestrator.play(max_plies=20)
    assert g.robot.injected_failures > 0
    assert s.robot_first_try_rate < 1.0
    assert s.robot_autonomous_rate is not None
    # Whatever happened, vision-verified state must match the game.
    assert g.physical.matches(g.orchestrator.board)


def test_human_assist_when_retries_exhausted(sim_cfg):
    sim_cfg.robot.sim_failure_rate = 1.0
    sim_cfg.robot.max_retries = 1
    sim_cfg.game.robot_color = "white"
    g = build_sim_game(sim_cfg, seed=5)
    s = g.orchestrator.play(max_plies=1)
    assert s.human_assists >= 1
    assert g.physical.matches(g.orchestrator.board)


def test_robot_as_white_plays_first(sim_cfg):
    sim_cfg.game.robot_color = "white"
    g = build_sim_game(sim_cfg, seed=1)
    g.orchestrator.play(max_plies=2)
    assert [r.side for r in g.orchestrator.metrics.records] == ["robot", "human"]


def occ(**squares):
    o = [EMPTY] * 64
    for name, v in squares.items():
        o[chess.parse_square(name)] = v
    return o


def test_correction_retry_on_grasp_miss():
    p = PickPlace(KNIGHT, "g1", "f3")
    assert plan_correction(p, occ(g1=WHITE), occ(f3=WHITE)) == p


def test_correction_from_misplaced_square():
    p = PickPlace(KNIGHT, "g1", "f3")
    c = plan_correction(p, occ(f4=WHITE), occ(f3=WHITE))
    assert (c.src, c.dst, c.reason) == ("f4", "f3", "correction")


def test_correction_gives_up_when_target_blocked():
    p = PickPlace(KNIGHT, "g1", "f3")
    assert plan_correction(p, occ(g1=WHITE, f3=BLACK), occ(f3=WHITE)) is None


def test_aggregate(sim_cfg):
    sums = [build_sim_game(sim_cfg, seed=i).orchestrator.play(max_plies=4) for i in range(2)]
    rep = aggregate(sums)
    assert rep["games"] == 2 and rep["robot_moves"] == 4
