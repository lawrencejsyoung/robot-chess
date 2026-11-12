"""Measurable results: per-move records and per-game / per-run summaries."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MoveRecord:
    ply: int
    side: str  # "robot" | "human"
    uci: str
    san: str
    # robot moves
    engine_time_s: float | None = None
    engine_source: str | None = None
    score_cp: int | None = None
    primitives: int = 0
    attempts: int = 0  # primitive executions incl. retries/corrections
    corrections: int = 0
    human_assists: int = 0
    execution_time_s: float | None = None
    verified: bool | None = None
    failures: list[str] = field(default_factory=list)
    # human moves
    detection_latency_s: float | None = None
    detection_correct: bool | None = None  # only known in simulation
    detection_ambiguous: bool | None = None
    unmatched_readings: int = 0

    @property
    def first_try(self) -> bool:
        return self.side == "robot" and self.corrections == 0 and self.human_assists == 0


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 4) if xs else None


@dataclass
class GameSummary:
    result: str
    termination: str
    plies: int
    robot_moves: int
    robot_primitives: int
    robot_first_try_rate: float | None
    robot_autonomous_rate: float | None  # verified without human help
    corrections_per_robot_move: float | None
    human_assists: int
    mean_engine_time_s: float | None
    mean_execution_time_s: float | None
    human_moves: int
    detection_accuracy: float | None
    mean_detection_latency_s: float | None
    wall_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsLogger:
    def __init__(self, log_dir: str | Path | None = "runs", game_name: str | None = None):
        self.records: list[MoveRecord] = []
        self.start = time.perf_counter()
        self.path: Path | None = None
        if log_dir is not None:
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{game_name or time.strftime('game-%Y%m%d-%H%M%S')}.jsonl"
            self.path.write_text("")

    def log(self, rec: MoveRecord) -> None:
        self.records.append(rec)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")

    def summary(self, result: str, termination: str) -> GameSummary:
        robot = [r for r in self.records if r.side == "robot"]
        human = [r for r in self.records if r.side == "human"]
        known = [r for r in human if r.detection_correct is not None]
        s = GameSummary(
            result=result,
            termination=termination,
            plies=len(self.records),
            robot_moves=len(robot),
            robot_primitives=sum(r.primitives for r in robot),
            robot_first_try_rate=_mean([float(r.first_try) for r in robot]),
            robot_autonomous_rate=_mean([float(bool(r.verified) and r.human_assists == 0) for r in robot]),
            corrections_per_robot_move=_mean([float(r.corrections) for r in robot]),
            human_assists=sum(r.human_assists for r in robot),
            mean_engine_time_s=_mean([r.engine_time_s for r in robot if r.engine_time_s is not None]),
            mean_execution_time_s=_mean([r.execution_time_s for r in robot if r.execution_time_s is not None]),
            human_moves=len(human),
            detection_accuracy=_mean([float(bool(r.detection_correct)) for r in known]),
            mean_detection_latency_s=_mean([r.detection_latency_s for r in human if r.detection_latency_s is not None]),
            wall_time_s=round(time.perf_counter() - self.start, 3),
        )
        if self.path:
            self.path.with_suffix(".summary.json").write_text(json.dumps(s.to_dict(), indent=2))
        return s


def aggregate(summaries: list[GameSummary]) -> dict:
    """Combine several games into one benchmark report."""
    def m(attr: str) -> float | None:
        vals = [getattr(s, attr) for s in summaries if getattr(s, attr) is not None]
        return _mean(vals)

    results = [s.result for s in summaries]
    return {
        "games": len(summaries),
        "results": {r: results.count(r) for r in sorted(set(results))},
        "robot_moves": sum(s.robot_moves for s in summaries),
        "robot_primitives": sum(s.robot_primitives for s in summaries),
        "robot_first_try_rate": m("robot_first_try_rate"),
        "robot_autonomous_rate": m("robot_autonomous_rate"),
        "corrections_per_robot_move": m("corrections_per_robot_move"),
        "human_assists": sum(s.human_assists for s in summaries),
        "mean_engine_time_s": m("mean_engine_time_s"),
        "mean_execution_time_s": m("mean_execution_time_s"),
        "detection_accuracy": m("detection_accuracy"),
        "mean_detection_latency_s": m("mean_detection_latency_s"),
    }
