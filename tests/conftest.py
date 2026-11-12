import pytest

from robot_chess.config import Config


@pytest.fixture
def sim_cfg() -> Config:
    cfg = Config()
    cfg.engine.uci_path = None  # deterministic, no external binary
    cfg.engine.fallback_depth = 1
    cfg.vision.stable_frames = 1
    return cfg
