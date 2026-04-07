"""Run the full Eagle Vision CV pipeline on a video.

Unified entry point: detection → tracking → motion → activity → analytics.
Displays annotated video in an OpenCV window.

Usage:
    python run_pipeline.py --video input/sample5.mp4
    python run_pipeline.py --video input/sample5.mp4 --kafka --save
"""

import argparse
import json
import os
import tempfile
import time

import cv2
import numpy as np

from cv_service.pipeline import Pipeline, FrameResult
from cv_service.debug_printer import print_debug, _format_timestamp
from cv_service.time_analyzer import TimeAnalyzer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

def atomic_json_write(path: str, data: dict):
    """Write JSON atomically: temp file → flush → fsync → rename."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Cleanup temp file on failure, then re-raise
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# Class colors (blue = excavator, red = dump truck)
CLASS_HUE = {"excavator": 105, "dump_truck": 0}


def color_for_id(equipment_id: str, equipment_class: str = "") -> tuple:
    base_hue = CLASS_HUE.get(equipment_class, 90)
    offset = (hash(equipment_id) % 21) - 10
    hue = (base_hue + offset) % 180
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


def draw_frame(frame: np.ndarray, result: FrameResult, total: int) -> np.ndarray:
    """Draw bounding boxes, labels, and HUD."""
    annotated = frame.copy()

    for obj in result.objects:
        x1, y1, x2, y2 = obj.bbox.astype(int)
        color = color_for_id(obj.equipment_id, obj.equipment_class)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if obj.motion_source and obj.status == "ACTIVE":
            label = f"{obj.equipment_id} | {obj.status} | {obj.activity} | {obj.motion_source} | mag={obj.magnitude:.2f}"
        else:
            label = f"{obj.equipment_id} | {obj.status} | {obj.activity} | mag={obj.magnitude:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )

    hud = f"Frame {result.frame_idx}/{total} | Tracked: {len(result.objects)}"
    cv2.putText(
        annotated, hud, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Eagle Vision — Full Pipeline")
    parser.add_argument("--video", default="input/sample.mp4", help="Input video path")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--save", action="store_true", help="Save output video to output/")
    parser.add_argument("--kafka", action="store_true", help="Enable Kafka streaming")
    args = parser.parse_args()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init pipeline
    pipe = Pipeline(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        fps=fps,
    )
    print(f"Model classes: {pipe.class_names}")
    print(f"Video: {args.video} | {total} frames @ {fps:.1f} FPS | {width}x{height}")
    print("Press ESC to quit, SPACE to pause/resume\n")

    # Kafka producer (optional)
    producer = None
    if args.kafka:
        from cv_service.kafka_producer import EventProducer
        producer = EventProducer()
        if producer.connected:
            print("Kafka streaming enabled.")
        else:
            print("Kafka unavailable — streaming disabled.")
            producer = None

    # Video writer (optional)
    writer = None
    out_path = None
    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(OUTPUT_DIR, f"pipeline_{base}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving video to: {out_path}")

    # JSON collection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "analytics.json")
    print(f"DEBUG PIPELINE PATH: {json_path}")
    results_data = {
        "status": "processing",
        "video_info": {"source": args.video, "total_frames": total, "fps": fps},
        "frames": [],
        "summary": {},
    }
    atomic_json_write(json_path, results_data)

    paused = False
    stopped_early = False

    try:
        while True:
            # Check stop flag from UI
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                result = pipe.process_frame(frame)

                # Collect JSON
                frame_data = {
                    "frame_id": result.frame_idx,
                    "timestamp": _format_timestamp(result.frame_idx, fps),
                    "objects": [o.to_dict() for o in result.objects],
                }
                results_data["frames"].append(frame_data)

                # Kafka: send every 10 frames
                if producer and result.frame_idx % 10 == 0:
                    producer.send(result.to_dict())

                # Draw and display
                annotated = draw_frame(frame, result, total)

                if writer:
                    writer.write(annotated)

                cv2.imshow("Eagle Vision", annotated)

                # Console every 30 frames
                if result.frame_idx % 30 == 0:
                    print(f"[Frame {result.frame_idx}/{total}]")
                    for o in result.objects:
                        src = f" | {o.motion_source}" if o.motion_source else ""
                        print(f"  {o.equipment_id} ({o.equipment_class}): "
                              f"{o.status} | {o.activity}{src} mag={o.magnitude:.2f}")

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\nStopping early...")
                stopped_early = True
                break
            elif key == ord(" "):
                paused = not paused

    except Exception as e:
        print(f"\nPipeline error: {e}")
        stopped_early = True

    finally:
        # ── Write analytics FIRST, before any cleanup that might hang ──
        print("\nSaving final analytics...")

        try:
            stats = pipe.get_stats()
            if not stats:
                print("WARNING: Empty stats after stop")
        except Exception as e:
            print(f"Stats error: {e}")
            stats = {}

        final_data = {
            "status": "completed",
            "video_info": results_data.get("video_info", {}),
            "summary": stats,
        }
        try:
            atomic_json_write(json_path, final_data)
            time.sleep(0.5)
            print(f"STATUS WRITTEN: completed -> {json_path}")
        except Exception as e:
            # Last-resort fallback: plain write
            print(f"Atomic write failed ({e}), trying plain write...")
            with open(json_path, "w") as f:
                json.dump(final_data, f, indent=2)
            print(f"STATUS WRITTEN (plain): completed -> {json_path}")

        # Full frame data (non-critical)
        results_data["summary"] = stats
        results_data["status"] = "completed"
        full_path = os.path.join(OUTPUT_DIR, "analytics_full.json")
        try:
            atomic_json_write(full_path, results_data)
            print(f"Full frame data saved -> {full_path}")
        except Exception as e:
            print(f"Full data write failed (non-critical): {e}")

        # ── Now cleanup resources (safe to hang/fail) ──
        print(f"Processed {pipe.frame_idx} frames.")
        if out_path:
            print(f"Video saved to: {out_path}")

        cap.release()
        if writer:
            writer.release()
        if producer:
            producer.flush()
            producer.close()
        cv2.destroyAllWindows()

        if stopped_early:
            print("Pipeline stopped early.")
        else:
            print("Pipeline completed normally.")


if __name__ == "__main__":
    main()
