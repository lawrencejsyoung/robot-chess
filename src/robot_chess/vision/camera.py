"""Camera sources: a real OpenCV device and a renderer for the simulator."""

from __future__ import annotations

from typing import Callable, Protocol

import chess
import cv2
import numpy as np

from .calibration import BoardCalibration


class Camera(Protocol):
    def read(self) -> np.ndarray:
        """Return a BGR uint8 frame."""
        ...

    def close(self) -> None: ...


class OpenCVCamera:
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None):
        self.cap = cv2.VideoCapture(index)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame

    def close(self) -> None:
        self.cap.release()


LIGHT_SQ = (150, 190, 210)  # BGR
DARK_SQ = (60, 100, 140)
WHITE_PIECE = (245, 248, 250)
BLACK_PIECE = (30, 30, 35)


def render_topdown(pieces: dict[int, chess.Piece], size: int = 640) -> np.ndarray:
    """Draw a top-down board (a8 top-left) with pieces as shaded discs."""
    img = np.zeros((size, size, 3), np.uint8)
    s = size / 8
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = int(f * s), int((7 - r) * s)
        x1, y1 = int((f + 1) * s), int((8 - r) * s)
        colour = LIGHT_SQ if (f + r) % 2 else DARK_SQ
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), colour, -1)
        piece = pieces.get(sq)
        if piece:
            cx, cy = int((f + 0.5) * s), int((7 - r + 0.5) * s)
            body = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
            rim = BLACK_PIECE if piece.color == chess.WHITE else WHITE_PIECE
            radius = int(s * (0.30 + 0.02 * piece.piece_type))
            cv2.circle(img, (cx, cy), radius, body, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), radius, rim, 2, cv2.LINE_AA)
            cv2.putText(img, piece.symbol(), (cx - int(s * .12), cy + int(s * .12)),
                        cv2.FONT_HERSHEY_SIMPLEX, s / 90, rim, 2, cv2.LINE_AA)
    return img


class SimCamera:
    """Renders the simulator's physical board as seen by a tilted camera.

    ``pieces_fn`` returns the current {square: piece} map, so the camera always
    reflects the latest physical state (including robot mistakes).
    """

    def __init__(
        self,
        pieces_fn: Callable[[], dict[int, chess.Piece]],
        width: int = 960,
        height: int = 720,
        corners: np.ndarray | None = None,
        noise_std: float = 4.0,
        seed: int | None = None,
        board_px: int = 640,
    ):
        self.pieces_fn = pieces_fn
        self.width, self.height = width, height
        self.board_px = board_px
        # Default: camera behind White, looking down at an angle (trapezoid).
        self.corners = (
            corners
            if corners is not None
            else np.array([[150, 660], [810, 660], [690, 90], [270, 90]], dtype=np.float32)
        )
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    @property
    def true_calibration(self) -> BoardCalibration:
        return BoardCalibration(self.corners.copy(), self.board_px)

    def render(self, pieces: dict[int, chess.Piece]) -> np.ndarray:
        top = render_topdown(pieces, self.board_px)
        cal = BoardCalibration(self.corners, self.board_px)
        H = np.linalg.inv(cal.homography())
        frame = np.full((self.height, self.width, 3), 70, np.uint8)
        warped = cv2.warpPerspective(top, H, (self.width, self.height))
        mask = cv2.warpPerspective(
            np.full((self.board_px, self.board_px), 255, np.uint8), H, (self.width, self.height)
        )
        frame[mask > 0] = warped[mask > 0]
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return frame

    def read(self) -> np.ndarray:
        return self.render(self.pieces_fn())

    def close(self) -> None:
        pass
