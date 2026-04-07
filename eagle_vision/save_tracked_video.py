"""Process a video through the tracking pipeline and save the annotated output.

Usage:
    python save_tracked_video.py
    python save_tracked_video.py --video input/sample5.mp4
    python save_tracked_video.py --model models/best.pt --conf 0.5
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from cv_service.tracker import EquipmentTracker
from cv_service.motion_analyzer import MotionAnalyzer
from cv_service.time_analyzer import TimeAnalyzer
from cv_service.debug_printer import print_debug, _format_timestamp

BOTSORT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "botsort.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Base hues per class (OpenCV HSV: blue ~105, red ~0)
CLASS_HUE = {"excavator": 105, "dump_truck": 0}


def color_for_id(equipment_id: str, equipment_class: str = "") -> tuple:
    """BGR color based on class with per-ID variation."""
    base_hue = CLASS_HUE.get(equipment_class, 90)
    offset = (hash(equipment_id) % 21) - 10
    hue = (base_hue + offset) % 180
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


def draw_tracked(frame: np.ndarray, tracked: list, motions: dict, info: str) -> np.ndarray:
    """Draw bounding boxes, labels with motion state, and HUD on the frame."""
    annotated = frame.copy()

    for t in tracked:
        x1, y1, x2, y2 = t.bbox.astype(int)
        color = color_for_id(t.equipment_id, t.equipment_class)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        m = motions.get(t.equipment_id)
        status = m.status if m else "---"
        activity = m.activity if m else "---"
        mag = m.magnitude if m else 0.0
        msrc = m.motion_source if m else None

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

    cv2.putText(
        annotated, info, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Save tracked video to file")
    parser.add_argument("--video", default="input/sample.mp4", help="Input video path")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--output", default=None, help="Output filename (default: tracked_<input>.mp4)")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)
    class_names = model.names
    print(f"Model classes: {class_names}")

    tracker = EquipmentTracker(class_lock_threshold=15)
    analyzer = MotionAnalyzer(loading_distance=400)
    time_analyzer = TimeAnalyzer()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.output:
        out_name = args.output
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_name = f"tracked_{base}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Video: {args.video} | {total} frames @ {fps:.1f} FPS | {width}x{height}")
    print(f"Output: {out_path}")
    print(f"Conf: {args.conf} | IoU: {args.iou}")
    print("Processing...\n")

    frame_idx = 0
    prev_frame = None
    start = time.time()

    # JSON analytics collection
    results_data = {
        "video_info": {
            "source": args.video,
            "total_frames": total,
            "fps": fps,
            "resolution": f"{width}x{height}",
        },
        "frames": [],
        "summary": {},
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            persist=True,
            tracker=BOTSORT_CONFIG,
            verbose=False,
        )
        tracked = tracker.update(results, class_names)

        # Motion analysis
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

        info = f"Frame {frame_idx}/{total} | Tracked: {len(tracked)}"
        annotated = draw_tracked(frame, tracked, motions, info)
        writer.write(annotated)

        # Progress every 100 frames
        if frame_idx % 100 == 0 or frame_idx == total:
            elapsed = time.time() - start
            pct = frame_idx / total * 100 if total else 0
            avg_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  [{frame_idx}/{total}] {pct:.0f}% | {avg_fps:.1f} FPS")

            # Debug + analytics every 500 frames
            if frame_idx % 500 == 0 and motions:
                print_debug(
                    frame_idx, fps,
                    list(motions.values()),
                    time_analyzer.get_stats(),
                    loading_distance=400.0,
                )

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print(f"\nDone. Video saved to: {out_path}")
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} FPS)")

    # Final debug summary
    final_motions = list(motions.values()) if motions else []
    if final_motions:
        print_debug(
            frame_idx, fps,
            final_motions,
            time_analyzer.get_stats(),
            loading_distance=400.0,
        )

    # Save JSON analytics
    results_data["summary"] = time_analyzer.get_stats()
    json_path = os.path.join(OUTPUT_DIR, "analytics.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Analytics saved to: {json_path}")


if __name__ == "__main__":
    main()
