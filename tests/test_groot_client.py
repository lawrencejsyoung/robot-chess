import chess
import numpy as np

from robot_chess.config import GrootConfig
from robot_chess.robot.groot_client import TASK_KEY, GrootRobot, _decode, _encode, build_instruction
from robot_chess.robot.planner import PickPlace

P = PickPlace(chess.Piece(chess.KNIGHT, chess.BLACK), "g8", "f6")


class FakeTransport:
    def __init__(self, chunk_len=16):
        self.observations = []
        self.chunk_len = chunk_len

    def get_action(self, obs):
        self.observations.append(obs)
        return {
            "action.single_arm": np.zeros((self.chunk_len, 5), np.float32),
            "action.gripper": np.ones((self.chunk_len, 1), np.float32),
        }

    def close(self):
        pass


class FakeIO:
    def __init__(self, done_after):
        self.applied = []
        self.done_after = done_after
        self.homed = 0

    def get_images(self):
        return {"front": np.zeros((4, 4, 3), np.uint8)}

    def get_state(self):
        return {"single_arm": np.zeros(5), "gripper": np.zeros(1)}

    def apply_action(self, a):
        self.applied.append(a)

    def primitive_done(self):
        return len(self.applied) >= self.done_after

    def home(self):
        self.homed += 1


def test_instruction_text():
    assert build_instruction(P, GrootConfig().instruction_template) == \
        "pick the black knight on g8 and place it on f6"
    q = PickPlace(chess.Piece(chess.QUEEN, chess.WHITE), "reserve", "b8")
    assert "on reserve and place it on b8" in build_instruction(q, GrootConfig().instruction_template)


def test_control_loop_requeries_until_done():
    io, tr = FakeIO(done_after=20), FakeTransport()
    res = GrootRobot(io, tr, GrootConfig(action_horizon=8)).execute(P)
    assert res.completed and res.steps == 20
    assert res.info["policy_queries"] == 3
    obs = tr.observations[0]
    assert obs["video.front"].shape == (1, 4, 4, 3)
    assert obs["state.single_arm"].shape == (1, 5)
    assert obs[TASK_KEY] == ["pick the black knight on g8 and place it on f6"]
    assert set(io.applied[0]) == {"single_arm", "gripper"}
    assert io.homed == 1


def test_step_cap():
    io = FakeIO(done_after=10**9)
    res = GrootRobot(io, FakeTransport(), GrootConfig(max_steps_per_primitive=30)).execute(P)
    assert not res.completed and res.steps == 30


def test_ndarray_codec_roundtrip():
    a = np.random.rand(3, 2).astype(np.float32)
    assert np.array_equal(_decode(_encode(a)), a)
