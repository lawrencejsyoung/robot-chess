"""A kinematics-free simulated robot with injectable failures.

It mutates a ``PhysicalBoard`` that the ``SimCamera`` renders, so failures
are only discovered the same way they would be on hardware: through vision.
"""

from __future__ import annotations

import random
import time

import chess

from ..physical import PhysicalBoard, is_square
from .base import PrimitiveResult, RobotController
from .planner import PickPlace


class SimRobot(RobotController):
    def __init__(
        self,
        physical: PhysicalBoard,
        failure_rate: float = 0.0,
        seed: int | None = None,
        seconds_per_primitive: float = 0.0,
    ):
        self.physical = physical
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)
        self.seconds_per_primitive = seconds_per_primitive
        self.injected_failures = 0

    def execute(self, p: PickPlace) -> PrimitiveResult:
        start = time.perf_counter()
        if self.seconds_per_primitive:
            time.sleep(self.seconds_per_primitive)
        info: dict = {}
        if is_square(p.src) and self.physical.piece_at(p.src) is None:
            info["error"] = f"nothing to pick at {p.src}"
            return PrimitiveResult(p, False, time.perf_counter() - start, info=info)

        if self.rng.random() < self.failure_rate:
            self.injected_failures += 1
            mode = self.rng.choice(["grasp_miss", "misplace"])
            neighbour = self._free_neighbour(p.dst) if is_square(p.dst) else None
            if mode == "misplace" and neighbour:
                piece = self.physical.take(p.src, p.piece) or p.piece
                self.physical.put(neighbour, piece)
                info["failure"] = f"misplaced on {neighbour}"
            else:
                info["failure"] = "grasp_miss"
            # The controller doesn't know it failed - vision must catch it.
            return PrimitiveResult(p, True, time.perf_counter() - start, info=info)

        if is_square(p.dst) and self.physical.piece_at(p.dst) is not None:
            info["error"] = f"destination {p.dst} occupied"
            return PrimitiveResult(p, False, time.perf_counter() - start, info=info)
        # Off-board zones are treated as having spares available.
        piece = self.physical.take(p.src, p.piece) or p.piece
        self.physical.put(p.dst, piece)
        return PrimitiveResult(p, True, time.perf_counter() - start, steps=1, info=info)

    def _free_neighbour(self, square_name: str) -> str | None:
        sq = chess.parse_square(square_name)
        f, r = chess.square_file(sq), chess.square_rank(sq)
        options = []
        for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                name = chess.square_name(chess.square(nf, nr))
                if self.physical.piece_at(name) is None:
                    options.append(name)
        return self.rng.choice(options) if options else None
