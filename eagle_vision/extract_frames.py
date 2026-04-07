"""Extract frames from a video at a configurable interval."""

import os
import cv2

VIDEO_PATH = "input/sample.mp4"
OUTPUT_DIR = "dataset/raw_frames"
FRAME_INTERVAL = 5  # save 1 frame every N frames


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            saved += 1
            filename = os.path.join(OUTPUT_DIR, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
