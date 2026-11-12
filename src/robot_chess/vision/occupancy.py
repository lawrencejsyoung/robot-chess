"""Per-square occupancy and colour classification.

Compares each square's centre crop against a reference image of the *empty*
board captured at calibration time. Pixels that differ from the reference
belong to a piece; their brightness tells white from black.

This is intentionally simple and transparent. For difficult lighting or
low-contrast piece sets, swap in a small CNN with the same ``classify``
signature (the move detector only needs the 64-entry grid).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..physical import BLACK, EMPTY, WHITE
from .calibration import BoardCalibration


class OccupancyClassifier:
    def __init__(
        self,
        calibration: BoardCalibration,
        empty_reference: np.ndarray,
        roi_fraction: float = 0.6,
        diff_threshold: float = 18.0,
        min_piece_fraction: float = 0.15,
    ):
        """``empty_reference`` is a *warped* (top-down) BGR image of the empty board."""
        self.cal = calibration
        self.ref = cv2.cvtColor(empty_reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.ref = cv2.GaussianBlur(self.ref, (5, 5), 0)
        self.roi_fraction = roi_fraction
        self.diff_threshold = diff_threshold
        self.min_piece_fraction = min_piece_fraction

    @classmethod
    def from_camera_frame(cls, calibration: BoardCalibration, empty_frame: np.ndarray, **kw) -> "OccupancyClassifier":
        return cls(calibration, calibration.warp(empty_frame), **kw)

    def classify_warped(self, warped: np.ndarray) -> list[int]:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = np.abs(gray - self.ref)
        occ = [EMPTY] * 64
        for sq in range(64):
            y0, y1, x0, x1 = self.cal.square_roi(sq, self.roi_fraction)
            d = diff[y0:y1, x0:x1]
            mask = d > self.diff_threshold
            if mask.mean() < self.min_piece_fraction:
                continue
            piece_px = gray[y0:y1, x0:x1][mask]
            ref_px = self.ref[y0:y1, x0:x1][mask]
            # A piece is "white" if it is brighter than the square beneath it
            # would be AND bright in absolute terms; otherwise "black".
            occ[sq] = WHITE if np.median(piece_px) > max(128.0, np.median(ref_px) - 10) else BLACK
        return occ

    def classify(self, frame: np.ndarray) -> list[int]:
        return self.classify_warped(self.cal.warp(frame))
