"""Optical-flow motion analyzer for tracked construction equipment.

Computes per-object motion magnitude inside bounding boxes using
Farneback dense optical flow, then classifies status and activity:
  - Excavator:  ACTIVE → DIGGING  (+ motion_source: Arm only / Full Body / static)
  - Dump Truck: near active excavator → LOADING, else → WAITING
                custom status: MOVE / IDLE
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from cv_service.tracker import TrackedEquipment

logger = logging.getLogger("eagle_vision.motion")

# Per-class flow thresholds
DEFAULT_FLOW_THRESHOLDS: Dict[str, float] = {
    "excavator": 0.1,
    "dump_truck": 0.5,
}


@dataclass
class MotionResult:
    """Motion analysis result for a single tracked object."""

    equipment_id: str
    equipment_class: str
    magnitude: float              # mean optical flow magnitude inside ROI
    status: str                   # excavator: ACTIVE/INACTIVE, truck: MOVE/IDLE
    activity: str                 # DIGGING / LOADING / WAITING
    motion_source: Optional[str]  # excavator only: "Arm only" / "Full Body" / "static"
    bbox: np.ndarray


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Return (cx, cy) center of a bounding box."""
    return np.array([
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    ], dtype=np.float32)


class MotionAnalyzer:
    """Analyze per-object motion via dense optical flow inside bounding boxes.

    Converts frames to grayscale, extracts each object's ROI, computes
    Farneback optical flow, and classifies status/activity with per-class
    thresholds and motion source detection for excavators.
    """

    def __init__(
        self,
        flow_thresholds: Optional[Dict[str, float]] = None,
        loading_distance: float = 600.0,
        min_roi_size: int = 10,
    ) -> None:
        """
        Args:
            flow_thresholds: Per-class magnitude thresholds. Falls back to defaults.
            loading_distance: Max center distance (px) to excavator for LOADING.
            min_roi_size: Minimum ROI width/height in pixels to analyze.
        """
        self.flow_thresholds = flow_thresholds or DEFAULT_FLOW_THRESHOLDS
        self.loading_distance = loading_distance
        self.min_roi_size = min_roi_size

    def _get_threshold(self, equipment_class: str) -> float:
        """Return flow threshold for a given class."""
        return self.flow_thresholds.get(equipment_class, 0.5)

    def analyze(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        tracked: List[TrackedEquipment],
    ) -> List[MotionResult]:
        """Compute motion and classify activity for each tracked object."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Pre-compute magnitudes
        magnitudes = {
            t.equipment_id: self._compute_roi_flow(prev_gray, gray, t.bbox)
            for t in tracked
        }

        # Excavator info: (center, is_active) for truck proximity check
        excavator_info: List[tuple] = []
        for t in tracked:
            if t.equipment_class == "excavator":
                thresh = self._get_threshold("excavator")
                is_active = magnitudes[t.equipment_id] > thresh
                excavator_info.append((_bbox_center(t.bbox), is_active))

        results: List[MotionResult] = []

        for t in tracked:
            mag = magnitudes[t.equipment_id]
            thresh = self._get_threshold(t.equipment_class)
            is_active = mag > thresh

            status = self._get_status(t.equipment_class, is_active)
            activity = self._classify_activity(
                t.equipment_class, is_active, t.bbox, excavator_info, mag,
            )
            motion_source = self._get_motion_source(t.equipment_class, mag)

            results.append(MotionResult(
                equipment_id=t.equipment_id,
                equipment_class=t.equipment_class,
                magnitude=round(mag, 2),
                status=status,
                activity=activity,
                motion_source=motion_source,
                bbox=t.bbox,
            ))

            logger.debug(
                "%s: mag=%.2f -> %s / %s / %s",
                t.equipment_id, mag, status, activity,
                motion_source or "N/A",
            )

        return results

    # ------------------------------------------------------------------
    # Status naming
    # ------------------------------------------------------------------

    @staticmethod
    def _get_status(equipment_class: str, is_active: bool) -> str:
        """Return display status — custom naming per class."""
        if equipment_class == "dump_truck":
            return "MOVE" if is_active else "IDLE"
        return "ACTIVE" if is_active else "INACTIVE"

    # ------------------------------------------------------------------
    # Motion source (excavator only)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_motion_source(equipment_class: str, mag: float) -> Optional[str]:
        """Classify motion source for excavators only."""
        if equipment_class != "excavator":
            return None
        if mag == 0.00:
            return "static"
        elif 0 < mag < 0.5:
            return "Arm only"
        else:
            return "Full Body"

    # ------------------------------------------------------------------
    # Activity classification
    # ------------------------------------------------------------------

    def _classify_activity(
        self,
        equipment_class: str,
        is_active: bool,
        bbox: np.ndarray,
        excavator_info: List[tuple],
        mag: float = 0.0,
    ) -> str:
        """Assign activity label based on class, status, and proximity."""
        if equipment_class == "excavator":
            return "DIGGING" if is_active else "WAITING"

        if equipment_class == "dump_truck":
            if not excavator_info:
                return "WAITING"

            truck_center = _bbox_center(bbox)
            nearest_dist = float("inf")
            nearest_active = False
            for ec, active in excavator_info:
                dist = float(np.linalg.norm(truck_center - ec))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_active = active

            logger.debug(
                f"[TRUCK DEBUG] dist={nearest_dist:.1f}, threshold={self.loading_distance}, "
                f"excavator_active={nearest_active}, truck_mag={mag:.2f}"
            )
            if nearest_active and nearest_dist < self.loading_distance:
                return "LOADING"
            return "WAITING"

        return "WAITING"

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    def _compute_roi_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        bbox: np.ndarray,
    ) -> float:
        """Compute mean optical flow magnitude inside a bounding box ROI."""
        h, w = prev_gray.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < self.min_roi_size or (y2 - y1) < self.min_roi_size:
            return 0.0

        roi_prev = prev_gray[y1:y2, x1:x2]
        roi_curr = curr_gray[y1:y2, x1:x2]

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(mag))
