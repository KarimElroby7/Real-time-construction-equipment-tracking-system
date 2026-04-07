"""Core inference engine for YOLOv8 equipment detection."""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Colors per class (BGR)
CLASS_COLORS = {
    "excavator": (0, 200, 0),
    "dump_truck": (0, 165, 255),
}
DEFAULT_COLOR = (255, 255, 255)

# Centralized defaults -- single source of truth
DEFAULT_CONF = 0.7
DEFAULT_IOU = 0.5
DEFAULT_VID_STRIDE = 10


class InferenceEngine:
    """Handles model loading, frame inference, and video processing."""

    def __init__(
        self,
        model_path: str,
        confidence: float = DEFAULT_CONF,
        iou_threshold: float = DEFAULT_IOU,
        vid_stride: int = DEFAULT_VID_STRIDE,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model: %s on %s", model_path, self.device)

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.vid_stride = vid_stride

        # Warm up the model with a dummy frame
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False)
        logger.info("Model loaded and warmed up")

    def predict_frame(self, frame: np.ndarray) -> list:
        """Run inference on a single frame and return results."""
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )
        return results

    def annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Draw bounding boxes and labels on a frame."""
        annotated = frame.copy()
        if not results or results[0].boxes is None:
            return annotated

        boxes = results[0].boxes
        names = results[0].names

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].item())
            cls_name = names[int(boxes.cls[i].item())]
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{cls_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated

    def process_video(
        self,
        input_path: str,
        output_path: str,
        show_fps: bool = True,
    ) -> dict:
        """Process a full video: detect, annotate, and write output.

        Returns a summary dict with frame count, avg FPS, and output path.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(
            "Processing: %s | %d frames | %dx%d @ %.1f fps",
            input_path, total_frames, width, height, fps,
        )

        frame_idx = 0
        last_results = []
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.vid_stride == 0:
                last_results = self.predict_frame(frame)
            annotated = self.annotate_frame(frame, last_results)

            # Overlay FPS counter
            if show_fps and frame_idx > 0:
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed
                cv2.putText(
                    annotated, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )

            writer.write(annotated)
            frame_idx += 1

            # Progress log every 100 frames
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                logger.info("Progress: %d/%d (%.0f%%)", frame_idx, total_frames, pct)

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0

        summary = {
            "input": input_path,
            "output": output_path,
            "frames_processed": frame_idx,
            "avg_fps": round(avg_fps, 1),
            "elapsed_seconds": round(elapsed, 1),
        }
        logger.info(
            "Done: %d frames in %.1fs (avg %.1f FPS) -> %s",
            frame_idx, elapsed, avg_fps, output_path,
        )
        return summary
