"""Select visually diverse frames using uniform sampling + difference filtering."""

import os
import cv2
import numpy as np

INPUT_DIR = "dataset/raw_frames"
OUTPUT_DIR = "dataset/selected_frames"
MAX_FRAMES = 200


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg"))
    total = len(files)

    if not files:
        print(f"No .jpg files found in {INPUT_DIR}/")
        return

    # توزيع الفريمات على الفيديو كله
    step = max(1, total // MAX_FRAMES)

    selected_files = files[::step]

    prev_gray = None
    saved = 0

    for filename in selected_files:
        if saved >= MAX_FRAMES:
            break

        path = os.path.join(INPUT_DIR, filename)
        frame = cv2.imread(path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            keep = True
        else:
            diff = cv2.absdiff(gray, prev_gray)
            keep = np.mean(diff) > 5  # فلترة بسيطة للتكرار

        if keep:
            saved += 1
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
            prev_gray = gray

    print(f"Selected {saved} diverse frames out of {total} → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()