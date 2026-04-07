"""Unified CV pipeline: detection → tracking → motion → activity → analytics.

Chains all CV modules into a single ``process_frame()`` call that returns
a structured ``FrameResult`` ready for downstream consumers (Kafka, API, UI).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from cv_service.tracker import EquipmentTracker, TrackedEquipment
from cv_service.motion_analyzer import MotionAnalyzer, MotionResult
from cv_service.time_analyzer import TimeAnalyzer


BOTSORT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "config", "botsort.yaml",
)


@dataclass
class ObjectState:
    """Full state of one tracked object for a single frame."""

    equipment_id: str
    equipment_class: str
    bbox: np.ndarray
    confidence: float
    status: str                   # ACTIVE/INACTIVE or MOVE/IDLE
    activity: str                 # DIGGING / LOADING / WAITING
    motion_source: Optional[str]  # excavator only
    magnitude: float

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "equipment_id": self.equipment_id,
            "equipment_class": self.equipment_class,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "activity": self.activity,
            "motion_source": self.motion_source,
            "magnitude": self.magnitude,
        }


@dataclass
class FrameResult:
    """Complete result of processing one frame."""

    frame_idx: int
    objects: List[ObjectState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_idx,
            "objects": [o.to_dict() for o in self.objects],
        }


class Pipeline:
    """End-to-end CV pipeline.

    Usage::

        pipe = Pipeline(model_path="models/best.pt")
        while has_frames:
            result = pipe.process_frame(frame)
            for obj in result.objects:
                print(obj.equipment_id, obj.status, obj.activity)
        stats = pipe.get_stats()
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        conf: float = 0.6,
        iou: float = 0.5,
        loading_distance: float = 600.0,
        fps: float = 30.0,
    ) -> None:
        self.model = YOLO(model_path)
        self.class_names: dict = self.model.names
        self.conf = conf
        self.iou = iou
        self.fps = fps
        self._dt = 1.0 / fps

        self._tracker = EquipmentTracker(class_lock_threshold=15)
        self._analyzer = MotionAnalyzer(loading_distance=loading_distance)
        self._time_analyzer = TimeAnalyzer()

        self._prev_frame: Optional[np.ndarray] = None
        self._frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Run full pipeline on one BGR frame.

        Returns:
            FrameResult with per-object state and analytics.
        """
        self._frame_idx += 1

        # 1. Detection + Tracking
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            tracker=BOTSORT_CONFIG,
            verbose=False,
        )
        tracked = self._tracker.update(results, self.class_names)

        # 2. Motion + Activity
        motions: Dict[str, MotionResult] = {}
        if self._prev_frame is not None and tracked:
            motion_results = self._analyzer.analyze(
                frame, self._prev_frame, tracked,
            )
            motions = {m.equipment_id: m for m in motion_results}
            self._time_analyzer.update(motion_results, self._dt)

        self._prev_frame = frame.copy()

        # 3. Build unified result
        objects: List[ObjectState] = []
        for t in tracked:
            m = motions.get(t.equipment_id)
            objects.append(ObjectState(
                equipment_id=t.equipment_id,
                equipment_class=t.equipment_class,
                bbox=t.bbox,
                confidence=t.confidence,
                status=m.status if m else "---",
                activity=m.activity if m else "---",
                motion_source=m.motion_source if m and m.status == "ACTIVE" else None,
                magnitude=m.magnitude if m else 0.0,
            ))

        return FrameResult(frame_idx=self._frame_idx, objects=objects)

    def get_stats(self) -> Dict[str, dict]:
        """Return cumulative time analytics per equipment."""
        return self._time_analyzer.get_stats()

    def reset(self) -> None:
        """Reset all pipeline state."""
        self._tracker.reset()
        self._time_analyzer.reset()
        self._prev_frame = None
        self._frame_idx = 0

    @property
    def frame_idx(self) -> int:
        return self._frame_idx
