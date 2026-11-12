"""Controller interface shared by the simulator and the GR00T client."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .planner import PickPlace


@dataclass
class PrimitiveResult:
    primitive: PickPlace
    completed: bool  # the controller believes it finished (vision still verifies)
    duration_s: float
    steps: int = 0
    info: dict = field(default_factory=dict)


class RobotController(ABC):
    @abstractmethod
    def execute(self, primitive: PickPlace) -> PrimitiveResult:
        """Physically perform one pick-and-place."""

    def home(self) -> None:
        """Move the arm out of the camera's view."""

    def close(self) -> None:
        pass
