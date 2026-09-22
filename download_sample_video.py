"""
download_sample_video.py — Generate or download a test video for the pipeline.

Since CDN downloads are unreliable (auth, geo-blocking, expired tokens), we default
to generating a synthetic classroom video with OpenCV.  This gives us a reproducible
test input that works offline.

Run once:
    python download_sample_video.py
"""

import os
import sys
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAVE_PATH = os.path.join(SAVE_DIR, "sample_classroom.mp4")


def generate_synthetic_video():
    """Generate a 10-second 720p synthetic classroom video.

    Creates 3 moving "person" silhouettes (colored rectangles + head circles)
    against a dark background.  This is enough to validate:
      - Video ingest (Stage 1)
      - Person detection (Stage 2) — real YOLO may not detect rectangles,
        but we can swap in a real clip later
      - Frame sampling arithmetic
      - Pipeline plumbing end-to-end
    """
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    if os.path.exists(SAVE_PATH):
        size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"[OK] Video already exists: {SAVE_PATH} ({size_mb:.1f} MB)")
        return SAVE_PATH

    fps = 30
    duration = 10  # seconds
    width, height = 1280, 720
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[ERROR] Could not create video writer. Check OpenCV codecs.")
        sys.exit(1)

    print(f"Generating {duration}s synthetic classroom video...")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output     : {SAVE_PATH}")

    for i in range(total_frames):
        # Dark gray background (simulates a dimly lit classroom)
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # 3 "person" silhouettes that sway left-right (simulates motion)
        offset = int(20 * np.sin(i * 0.05))

        persons = [
            # (x1, y1, x2, y2) bounding boxes
            (200 + offset, 150, 380 + offset, 550),   # left person
            (520 - offset, 130, 700 - offset, 560),    # center person
            (850 + offset, 160, 1030 + offset, 540),   # right person
        ]
        colors = [
            (0, 120, 255),   # orange-ish (BGR)
            (0, 255, 120),   # green
            (255, 120, 0),   # blue
        ]

        for (x1, y1, x2, y2), color in zip(persons, colors):
            # Body rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            # Head circle
            cx = (x1 + x2) // 2
            cv2.circle(frame, (cx, y1 - 30), 35, color, -1)

        # Frame counter overlay (useful for debugging sampling)
        cv2.putText(
            frame, f"Frame {i}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        writer.write(frame)

        # Progress every 2 seconds
        if i % (fps * 2) == 0:
            pct = (i + 1) * 100 // total_frames
            print(f"  Progress: {pct}%")

    writer.release()

    size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"[OK] Synthetic video saved: {SAVE_PATH} ({size_mb:.1f} MB)")
    return SAVE_PATH


if __name__ == "__main__":
    generate_synthetic_video()
