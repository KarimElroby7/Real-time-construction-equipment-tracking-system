"""CLI entry point for running YOLOv8 equipment detection on videos or images."""

import argparse
import logging
import sys
from pathlib import Path

from inference.engine import InferenceEngine, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_VID_STRIDE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "models/best.pt"
OUTPUT_DIR = Path("output")
SUPPORTED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def build_output_path(input_path: Path, suffix: str = "_detected") -> Path:
    """Generate output path preserving the original filename."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / f"{input_path.stem}{suffix}{input_path.suffix}"


def process_image(engine: InferenceEngine, input_path: Path) -> None:
    """Run inference on a single image and save the result."""
    import cv2

    frame = cv2.imread(str(input_path))
    if frame is None:
        logger.error("Cannot read image: %s", input_path)
        return

    results = engine.predict_frame(frame)
    annotated = engine.annotate_frame(frame, results)

    output_path = build_output_path(input_path)
    cv2.imwrite(str(output_path), annotated)
    logger.info("Saved: %s", output_path)


def process_source(engine: InferenceEngine, source: Path, no_fps: bool) -> None:
    """Process a single file or all supported files in a directory."""
    if source.is_dir():
        files = sorted(
            f for f in source.iterdir()
            if f.suffix.lower() in SUPPORTED_VIDEO | SUPPORTED_IMAGE
        )
        if not files:
            logger.error("No supported files found in: %s", source)
            return
        logger.info("Found %d files in %s", len(files), source)
        for f in files:
            process_source(engine, f, no_fps)
        return

    ext = source.suffix.lower()
    if ext in SUPPORTED_VIDEO:
        output_path = build_output_path(source)
        engine.process_video(str(source), str(output_path), show_fps=not no_fps)
    elif ext in SUPPORTED_IMAGE:
        process_image(engine, source)
    else:
        logger.warning("Skipping unsupported file: %s", source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eagle Vision -- Run equipment detection on videos or images",
    )
    parser.add_argument(
        "--video", "--source",
        type=str,
        required=True,
        dest="source",
        help="Path to a video, image, or folder of files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to YOLOv8 weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IoU threshold for NMS (default: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_VID_STRIDE,
        help=f"Process every Nth frame, write all frames (default: {DEFAULT_VID_STRIDE})",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Disable FPS overlay on output video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)

    if not source.exists():
        logger.error("Source not found: %s", source)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Place your best.pt in the models/ directory")
        sys.exit(1)

    engine = InferenceEngine(
        model_path=str(model_path),
        confidence=args.conf,
        iou_threshold=args.iou,
        vid_stride=args.stride,
    )

    process_source(engine, source, args.no_fps)
    logger.info("All done.")


if __name__ == "__main__":
    main()
