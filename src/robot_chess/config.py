"""Configuration loading (YAML -> dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EngineConfig:
    # Path to a UCI engine binary (e.g. "stockfish"). None -> built-in searcher.
    uci_path: str | None = "stockfish"
    think_time_s: float = 0.5
    # Stockfish "Skill Level" 0-20; lower makes a friendlier opponent.
    skill_level: int | None = 10
    fallback_depth: int = 2


@dataclass
class VisionConfig:
    camera_index: int = 0
    calibration_file: str = "calibration.json"
    # Warped top-down image of the empty board, written by `robot-chess calibrate`.
    empty_reference: str = "empty_reference.png"
    warp_size_px: int = 640
    # Fraction of each square (centre crop) used for occupancy statistics.
    roi_fraction: float = 0.6
    # Std-dev / mean-diff thresholds used by the occupancy classifier.
    occupancy_threshold: float = 18.0
    # Number of consecutive identical readings required before trusting a frame.
    stable_frames: int = 3


@dataclass
class GrootConfig:
    host: str = "localhost"
    port: int = 5555
    # "package.module:ClassName" of your RobotIO driver (see robot/groot_client.py).
    robot_io: str | None = None
    robot_io_kwargs: dict = field(default_factory=dict)
    timeout_ms: int = 15000
    # Steps of the returned action chunk to execute before re-querying the policy.
    action_horizon: int = 16
    max_steps_per_primitive: int = 400
    # Instruction template. Fields: piece, color, from_sq, to_sq, action.
    instruction_template: str = (
        "{action} the {color} {piece} on {from_sq} and place it on {to_sq}"
    )


@dataclass
class RobotConfig:
    backend: str = "sim"  # "sim" | "groot"
    # Off-board locations used for captured pieces and promotion reserves.
    graveyard_white: str = "graveyard_white"
    graveyard_black: str = "graveyard_black"
    reserve: str = "reserve"
    max_retries: int = 2
    # Simulator-only knobs used to stress the verification loop.
    sim_failure_rate: float = 0.0
    sim_seed: int | None = None
    groot: GrootConfig = field(default_factory=GrootConfig)


@dataclass
class GameConfig:
    robot_color: str = "black"  # "white" | "black"
    log_dir: str = "runs"


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    game: GameConfig = field(default_factory=GameConfig)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(dc)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{key}' for {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return _merge(cfg, data)
