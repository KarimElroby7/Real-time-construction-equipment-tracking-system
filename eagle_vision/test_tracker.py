"""Test script: run YOLO detection + BoT-SORT tracking on a video.

Usage:
    python test_tracker.py
    python test_tracker.py --video path/to/video.mp4
    python test_tracker.py --model models/best.pt --conf 0.1
"""

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

from cv_service.tracker import EquipmentTracker

# Path to BoT-SORT config (relative to this script)
BOTSORT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "botsort.yaml")


# Base hues per class (OpenCV HSV: blue ~105, red ~0)
CLASS_HUE = {"excavator": 105, "dump_truck": 0}


def color_for_id(equipment_id: str, equipment_class: str = "") -> tuple:
    """BGR color based on class (blue=excavator, red=dump_truck) with per-ID variation."""
    base_hue = CLASS_HUE.get(equipment_class, 90)
    # Small deterministic offset (±10) so different IDs within same class are distinguishable
    offset = (hash(equipment_id) % 21) - 10  # range -10..+10
    hue = (base_hue + offset) % 180
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


def draw_tracked(frame: np.ndarray, tracked: list) -> np.ndarray:
    """Draw bounding boxes with friendly IDs on the frame."""
    annotated = frame.copy()

    for t in tracked:
        x1, y1, x2, y2 = t.bbox.astype(int)
        color = color_for_id(t.equipment_id, t.equipment_class)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label: friendly ID + confidence
        label = f"{t.equipment_id} ({t.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Test BoT-SORT tracker")
    parser.add_argument("--video", default="input/sample.mp4", help="Input video path")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (1 = no skipping)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback delay multiplier (e.g. 2.0 = shorter delay)")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)
    class_names = model.names  # {0: 'dump_truck', 1: 'excavator'}
    print(f"Model classes: {class_names}")

    # Create tracker wrapper (class voting + friendly IDs)
    tracker = EquipmentTracker(class_lock_threshold=15)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms = 1  # minimal wait — inference is the bottleneck, not playback
    print(f"Video: {args.video} | {total} frames @ {fps:.1f} FPS")
    print(f"Tracker: BoT-SORT + Re-ID ({BOTSORT_CONFIG})")
    print(f"Conf: {args.conf} | IoU: {args.iou} | Stride: {args.stride} | Speed: {args.speed}x")
    print("Press ESC to quit, SPACE to pause/resume\n")

    frame_idx = 0
    processed = 0
    paused = False

    while True:
        if not paused:
            # Skip (stride - 1) frames by grabbing without decoding
            for _ in range(args.stride - 1):
                if not cap.grab():
                    break
                frame_idx += 1

            # Read the frame we actually process
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run detection + BoT-SORT tracking in one call
            results = model.track(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                persist=True,
                tracker=BOTSORT_CONFIG,
                verbose=False,
            )

            tracked = tracker.update(results, class_names)

            # Draw and display
            annotated = draw_tracked(frame, tracked)

            # HUD
            info = f"Frame {frame_idx}/{total} | Tracked: {len(tracked)} | Stride: {args.stride}"
            cv2.putText(
                annotated, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )

            # Console output every 30 processed frames
            if processed % 30 == 0:
                print(f"[Frame {frame_idx}] active={len(tracked)}")
                for t in tracked:
                    print(f"  {t.equipment_id} ({t.equipment_class}) "
                          f"conf={t.confidence:.2f} "
                          f"bbox=[{t.bbox[0]:.0f},{t.bbox[1]:.0f},"
                          f"{t.bbox[2]:.0f},{t.bbox[3]:.0f}]")

            cv2.imshow("Eagle Vision - Tracker Test", annotated)
            processed += 1

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {processed} of {frame_idx} frames (stride={args.stride}).")


if __name__ == "__main__":
    main()
