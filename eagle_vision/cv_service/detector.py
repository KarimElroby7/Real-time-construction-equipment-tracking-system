"""YOLOv8 object detector wrapper for construction equipment detection."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from ultralytics import YOLO


# COCO class IDs mapped to construction equipment types.
# YOLOv8 pretrained on COCO lacks dedicated excavator/dump-truck classes,
# so we use the closest available proxies:
#   car(2), bus(5), truck(7) → dump_truck
#   train(6) → excavator (tracked heavy machinery resemblance)
# Swap these out when using a fine-tuned model.
EQUIPMENT_CLASS_MAP: Dict[int, str] = {
    2: "dump_truck",
    5: "dump_truck",
    6: "excavator",
    7: "dump_truck",
}


@dataclass
class Detection:
    """Single detected object in a frame."""

    bbox: np.ndarray        # [x1, y1, x2, y2] absolute pixel coords
    confidence: float       # detection confidence score
    class_id: int           # COCO class id
    equipment_class: str    # mapped equipment label


class EquipmentDetector:
    """Wraps YOLOv8 for construction equipment detection.

    Loads the model once and exposes a simple ``detect`` method that
    returns only equipment-relevant detections.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        confidence: float,
        iou_threshold: float,
    ) -> None:
        """Initialize the detector.

        Args:
            model_path: Path to YOLOv8 weights (e.g. ``yolov8s.pt``).
            device: Inference device (``cuda`` or ``cpu``).
            confidence: Minimum confidence threshold.
            iou_threshold: IoU threshold for NMS.
        """
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects for equipment classes only.
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            if class_id not in EQUIPMENT_CLASS_MAP:
                continue

            bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
            conf = float(boxes.conf[i].item())

            detections.append(Detection(
                bbox=bbox,
                confidence=conf,
                class_id=class_id,
                equipment_class=EQUIPMENT_CLASS_MAP[class_id],
            ))

        return detections
