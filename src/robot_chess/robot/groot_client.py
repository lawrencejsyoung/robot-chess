"""Closed-loop client for an NVIDIA Isaac GR00T policy server.

Each ``PickPlace`` primitive becomes a natural-language instruction
("pick the black knight on g8 and place it on f6"). The client then runs the
standard VLA control loop:

    observe (camera frames + proprioception)
      -> policy server returns an action chunk
      -> execute the first ``action_horizon`` steps on the arm
      -> repeat until the hardware reports the primitive is done or a step cap

Two seams are injectable so this module has no hard dependency on a specific
robot or GR00T release:

* ``PolicyTransport`` - how observations reach the model. ``ZmqPolicyTransport``
  speaks the request/response pattern used by the Isaac-GR00T inference
  service (``{"endpoint": "get_action", "data": obs}``). Field names and
  serialisation have changed between GR00T releases, so check them against
  the server you deploy and subclass if needed.
* ``RobotIO`` - your arm/gripper/camera driver (e.g. SO-101 via LeRobot,
  Franka via ROS 2).

The policy is expected to be fine-tuned on chess pick-and-place demos whose
task strings follow ``GrootConfig.instruction_template``; see the README.
"""

from __future__ import annotations

import io
import time
from typing import Any, Protocol

import chess
import numpy as np

from ..config import GrootConfig
from .base import PrimitiveResult, RobotController
from .planner import PickPlace

TASK_KEY = "annotation.human.task_description"


class PolicyTransport(Protocol):
    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...


class RobotIO(Protocol):
    def get_images(self) -> dict[str, np.ndarray]:
        """Camera name -> HxWx3 uint8 RGB frame (e.g. {"front": ..., "wrist": ...})."""
        ...

    def get_state(self) -> dict[str, np.ndarray]:
        """State key -> 1D float array (e.g. {"single_arm": q[5], "gripper": g[1]})."""
        ...

    def apply_action(self, action: dict[str, np.ndarray]) -> None:
        """Apply one timestep: action key -> 1D array (same keys as the policy output)."""
        ...

    def primitive_done(self) -> bool:
        """Hardware-side completion signal (e.g. gripper re-opened after a grasp
        and the arm has lifted clear). Vision verification happens afterwards."""
        ...

    def home(self) -> None: ...


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray__": buf.getvalue()}
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and "__ndarray__" in obj:
        return np.load(io.BytesIO(obj["__ndarray__"]), allow_pickle=False)
    return obj


class ZmqPolicyTransport:
    """REQ socket client for a GR00T inference server (msgpack + .npy arrays)."""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        import msgpack  # optional dependency: pip install robot-chess[groot]
        import zmq

        self._msgpack = msgpack
        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self) -> None:
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def _call(self, endpoint: str, data: Any = None) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(self._msgpack.packb(req, default=_encode, use_bin_type=True))
        try:
            raw = self.sock.recv()
        except self._zmq.Again:
            # REQ sockets are stuck after a timeout; recreate before raising.
            self.sock.close()
            self._connect()
            raise TimeoutError(f"GR00T server at {self.addr} did not respond")
        resp = self._msgpack.unpackb(raw, object_hook=_decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except Exception:
            return False

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._call("get_action", observation)

    def close(self) -> None:
        self.sock.close()


def build_instruction(p: PickPlace, template: str) -> str:
    return template.format(
        action="pick",
        color="white" if p.piece.color == chess.WHITE else "black",
        piece=chess.piece_name(p.piece.piece_type),
        from_sq=p.location_phrase(p.src),
        to_sq=p.location_phrase(p.dst),
    )


def build_observation(io_: RobotIO, instruction: str) -> dict[str, Any]:
    """GR00T-style observation: video.* as (T=1,H,W,C), state.* as (T=1,D)."""
    obs: dict[str, Any] = {}
    for name, img in io_.get_images().items():
        obs[f"video.{name}"] = np.asarray(img, dtype=np.uint8)[None]
    for name, vec in io_.get_state().items():
        obs[f"state.{name}"] = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    obs[TASK_KEY] = [instruction]
    return obs


class GrootRobot(RobotController):
    def __init__(self, io_: RobotIO, transport: PolicyTransport, cfg: GrootConfig | None = None):
        self.io = io_
        self.transport = transport
        self.cfg = cfg or GrootConfig()

    def execute(self, p: PickPlace) -> PrimitiveResult:
        instruction = build_instruction(p, self.cfg.instruction_template)
        start = time.perf_counter()
        steps = queries = 0
        done = False
        while steps < self.cfg.max_steps_per_primitive and not done:
            chunk = self.transport.get_action(build_observation(self.io, instruction))
            queries += 1
            actions = {k.removeprefix("action."): np.asarray(v) for k, v in chunk.items()}
            if not actions:
                raise RuntimeError("Policy returned an empty action chunk")
            chunk_len = min(len(v) for v in actions.values())
            for t in range(min(self.cfg.action_horizon, chunk_len)):
                self.io.apply_action({k: v[t] for k, v in actions.items()})
                steps += 1
                if self.io.primitive_done() or steps >= self.cfg.max_steps_per_primitive:
                    done = self.io.primitive_done()
                    break
        self.io.home()
        return PrimitiveResult(
            p, done, time.perf_counter() - start, steps,
            info={"instruction": instruction, "policy_queries": queries},
        )

    def home(self) -> None:
        self.io.home()

    def close(self) -> None:
        self.transport.close()
