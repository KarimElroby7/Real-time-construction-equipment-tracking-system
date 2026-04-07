"""Minimal script to visually verify YOLOv8 equipment detection on a video."""

import sys
import os

import cv2
import numpy as np
from dotenv import load_dotenv

# Allow imports from project root.
sys.path.insert(0, os.path.dirname(__file__))

from cv_service.detector import EquipmentDetector

load_dotenv()


def main() -> None:
    """Run detection on a video and display annotated frames."""
    video_path = os.getenv("VIDEO_SOURCE", "input/sample.mp4")
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    detector = EquipmentDetector(
        model_path=os.getenv("YOLO_MODEL", "yolov8s.pt"),
        device=os.getenv("YOLO_DEVICE", "cuda"),
        confidence=float(os.getenv("YOLO_CONFIDENCE", "0.35")),
        iou_threshold=float(os.getenv("YOLO_IOU_THRESHOLD", "0.45")),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    print("Press 'q' to quit, any other key to pause/resume.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            label = f"{det.equipment_class} {det.confidence:.2f}"
            color = (0, 255, 0) if det.equipment_class == "excavator" else (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Eagle Vision - Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
