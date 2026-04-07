"""Test script: run tracking + motion analysis on a video.

Usage:
    python test_motion.py
    python test_motion.py --video input/sample5.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

from cv_service.tracker import EquipmentTracker
from cv_service.motion_analyzer import MotionAnalyzer
from cv_service.time_analyzer import TimeAnalyzer
from cv_service.debug_printer import print_debug, _format_timestamp

BOTSORT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "botsort.yaml")

# Class colors (blue = excavator, red = dump truck)
CLASS_HUE = {"excavator": 105, "dump_truck": 0}


def color_for_id(equipment_id: str, equipment_class: str = "") -> tuple:
    base_hue = CLASS_HUE.get(equipment_class, 90)
    offset = (hash(equipment_id) % 21) - 10
    hue = (base_hue + offset) % 180
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


def main():
    parser = argparse.ArgumentParser(description="Test motion analyzer")
    parser.add_argument("--video", default="input/sample.mp4")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.model)
    class_names = model.names
    tracker = EquipmentTracker(class_lock_threshold=15)
    analyzer = MotionAnalyzer(loading_distance=400)
    time_analyzer = TimeAnalyzer()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} | {total} frames @ {fps:.1f} FPS")
    print("Press ESC to quit, SPACE to pause/resume\n")

    prev_frame = None
    frame_idx = 0
    paused = False
    stopped_early = False

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_data = {
        "video_info": {
            "source": args.video,
            "total_frames": total,
            "fps": fps,
        },
        "frames": [],
        "summary": {},
    }

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Track
            results = model.track(
                source=frame, conf=args.conf, iou=args.iou,
                persist=True, tracker=BOTSORT_CONFIG, verbose=False,
            )
            tracked = tracker.update(results, class_names)

            # Motion analysis (needs previous frame)
            motions = {}
            if prev_frame is not None and tracked:
                motion_results = analyzer.analyze(frame, prev_frame, tracked)
                motions = {m.equipment_id: m for m in motion_results}
                time_analyzer.update(motion_results, dt=1.0 / fps)

            prev_frame = frame.copy()

            # Collect frame data for JSON
            frame_data = {
                "frame_id": frame_idx,
                "timestamp": _format_timestamp(frame_idx, fps),
                "objects": [],
            }
            for m in motions.values():
                msrc = m.motion_source if m.status == "ACTIVE" else None
                frame_data["objects"].append({
                    "equipment_id": m.equipment_id,
                    "equipment_class": m.equipment_class,
                    "status": m.status,
                    "activity": m.activity,
                    "motion_source": msrc,
                    "magnitude": m.magnitude,
                })
            results_data["frames"].append(frame_data)

            # Draw
            annotated = frame.copy()
            for t in tracked:
                x1, y1, x2, y2 = t.bbox.astype(int)
                m = motions.get(t.equipment_id)
                status = m.status if m else "---"
                activity = m.activity if m else "---"
                mag = m.magnitude if m else 0.0
                msrc = m.motion_source if m else None

                color = color_for_id(t.equipment_id, t.equipment_class)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if msrc and status == "ACTIVE":
                    label = f"{t.equipment_id} | {status} | {activity} | {msrc} | mag={mag:.2f}"
                else:
                    label = f"{t.equipment_id} | {status} | {activity} | mag={mag:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(
                    annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
                )

            # HUD
            cv2.putText(
                annotated, f"Frame {frame_idx}/{total} | Tracked: {len(tracked)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )

            # Console output every 30 frames
            if frame_idx % 30 == 0:
                print(f"[Frame {frame_idx}]")
                for t in tracked:
                    m = motions.get(t.equipment_id)
                    st = m.status if m else "---"
                    act = m.activity if m else "---"
                    mag = m.magnitude if m else 0.0
                    msrc = m.motion_source if m else None
                    src_str = f" | {msrc}" if msrc else ""
                    print(f"  {t.equipment_id} ({t.equipment_class}): "
                          f"{st} | {act}{src_str} mag={mag:.2f}")

            # Debug + analytics every 100 frames
            if frame_idx % 100 == 0 and motions:
                print_debug(
                    frame_idx, fps,
                    list(motions.values()),
                    time_analyzer.get_stats(),
                    loading_distance=400.0,
                )

            cv2.imshow("Eagle Vision - Motion Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\nStopping early...")
            stopped_early = True
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_idx} frames.")

    # Save JSON analytics (always — even on early stop)
    print("Saving analytics...")
    results_data["summary"] = time_analyzer.get_stats()
    json_path = os.path.join(OUTPUT_DIR, "analytics.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    if stopped_early:
        print(f"Partial analytics saved (early stop). -> {json_path}")
    else:
        print(f"Full analytics saved. -> {json_path}")


if __name__ == "__main__":
    main()
