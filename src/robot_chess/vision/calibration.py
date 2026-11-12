"""Board localisation: map the camera image to a canonical top-down view.

The canonical view is a square image of side ``size`` with a8 at the top-left
and h1 at the bottom-right (White at the bottom). Calibration stores the four
*outer* board corners in camera pixels, in the order a1, h1, h8, a8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BoardCalibration:
    corners: np.ndarray  # (4, 2) float32: a1, h1, h8, a8 outer corners in camera px
    size: int = 640

    @property
    def square_px(self) -> float:
        return self.size / 8

    def dst_corners(self) -> np.ndarray:
        s = float(self.size)
        return np.array([[0, s], [s, s], [s, 0], [0, 0]], dtype=np.float32)

    def homography(self) -> np.ndarray:
        return cv2.getPerspectiveTransform(self.corners.astype(np.float32), self.dst_corners())

    def warp(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.homography(), (self.size, self.size))

    def square_roi(self, square: int, fraction: float = 0.6) -> tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of the centre crop of ``square`` in the warped image."""
        f, r = square % 8, square // 8
        s = self.square_px
        margin = s * (1 - fraction) / 2
        x0, y0 = f * s + margin, (7 - r) * s + margin
        return int(y0), int(y0 + s * fraction), int(x0), int(x0 + s * fraction)

    # --- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"corners": self.corners.tolist(), "size": self.size}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "BoardCalibration":
        data = json.loads(Path(path).read_text())
        return cls(np.array(data["corners"], dtype=np.float32), int(data["size"]))

    # --- automatic calibration ----------------------------------------------

    @classmethod
    def from_empty_board(cls, image: np.ndarray, size: int = 640, rotate: int = 0) -> "BoardCalibration":
        """Locate an *empty* board via its 7x7 inner corners.

        The checkerboard is symmetric, so the orientation cannot be recovered
        from the pattern alone: ``rotate`` (0-3 quarter turns) picks which
        detected corner is a1. Check the result with ``robot-chess calibrate
        --preview`` and adjust if needed.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        found, pts = cv2.findChessboardCornersSB(gray, (7, 7))
        if not found:
            found, pts = cv2.findChessboardCorners(
                gray, (7, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if found:
                pts = cv2.cornerSubPix(
                    gray, pts, (5, 5), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )
        if not found:
            raise RuntimeError("Could not find the 7x7 inner corners of an empty chessboard")
        pts = pts.reshape(-1, 2)
        ideal = np.array([[i, j] for j in range(1, 8) for i in range(1, 8)], dtype=np.float32)
        H, _ = cv2.findHomography(ideal, pts)
        outer = np.array([[[0, 0]], [[8, 0]], [[8, 8]], [[0, 8]]], dtype=np.float32)
        quad = cv2.perspectiveTransform(outer, H).reshape(4, 2)
        # Order the quad as a1, h1, h8, a8 assuming a1 is the corner closest to
        # the bottom-left of the image (camera behind White), then apply rotate.
        quad = _order_bottom_left_ccw(quad)
        quad = np.roll(quad, -rotate, axis=0)
        return cls(quad.astype(np.float32), size)


def _order_bottom_left_ccw(quad: np.ndarray) -> np.ndarray:
    """Return corners ordered bottom-left, bottom-right, top-right, top-left."""
    c = quad.mean(axis=0)
    # In image coords y grows downward; sort by angle clockwise-on-screen
    # starting at bottom-left.
    ang = np.arctan2(-(quad[:, 1] - c[1]), quad[:, 0] - c[0])  # math angle, y up
    order = np.argsort(ang)  # from -pi (left) increasing: BL, BR, TR, TL
    q = quad[order]
    # Rotate so the start is the corner with max (y - x) i.e. bottom-left.
    start = int(np.argmax(q[:, 1] - q[:, 0]))
    return np.roll(q, -start, axis=0)
