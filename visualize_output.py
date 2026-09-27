"""
visualize_output.py

Draws the saved per-frame detection JSONs (bbox, keypoints, track_id) back
onto the source video frames, and writes an annotated .mp4 you can watch.

No GPU / inference needed — this only reads the JSON files your pipeline
already produced in output/ and paints them onto matching frames from the
source video. Safe to run on CPU, Colab free tier, or your own laptop.

Usage:
    python visualize_output.py                          # Uses config defaults
    python visualize_output.py --video data/Video1.mp4  # Custom video
    python visualize_output.py --video data/Video1.mp4 --json_dir output --out output/annotated.mp4
"""

import argparse
import json
import os
import re

import cv2
import config

# Skeleton connections for the 13-point contract (index pairs by name)
SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]

# Deterministic color per track_id so the same person keeps the same color
def color_for_track(track_id: int):
    palette = [
        (66, 135, 245), (66, 245, 111), (245, 66, 66), (245, 200, 66),
        (188, 66, 245), (66, 245, 212), (245, 66, 173), (150, 245, 66),
        (255, 140, 0), (0, 206, 209), (148, 0, 211), (255, 20, 147),
    ]
    return palette[track_id % len(palette)]


def load_frame_jsons(json_dir):
    """Returns a dict: frame_id -> parsed JSON."""
    pattern = re.compile(r"frame_(\d+)\.json$")
    frames = {}
    if not os.path.exists(json_dir):
        return frames

    for fname in os.listdir(json_dir):
        m = pattern.match(fname)
        if not m:
            continue
        frame_id = int(m.group(1))
        with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as f:
            frames[frame_id] = json.load(f)
    return frames


def draw_frame(img, frame_data):
    for track in frame_data.get("tracks", []):
        tid = track["track_id"]
        color = color_for_track(tid)

        # Bounding box
        x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Track label background for readability
        label = f"ID {tid} ({track.get('confidence', 0):.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - 20, 0)), (x1 + w + 4, max(y1, 20)), color, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Keypoints & Skeleton
        kp_by_name = {kp["name"]: kp for kp in track.get("keypoints", [])}
        for kp in kp_by_name.values():
            if kp.get("conf", 0) < 0.25:
                continue
            cv2.circle(img, (int(kp["x"]), int(kp["y"])), 4, color, -1)

        for a, b in SKELETON:
            ka, kb = kp_by_name.get(a), kp_by_name.get(b)
            if not ka or not kb:
                continue
            if ka.get("conf", 0) < 0.25 or kb.get("conf", 0) < 0.25:
                continue
            cv2.line(img, (int(ka["x"]), int(ka["y"])),
                     (int(kb["x"]), int(kb["y"])), color, 2, cv2.LINE_AA)

    return img


def main():
    ap = argparse.ArgumentParser(description="Render bounding boxes, tracks, and poses onto video")
    ap.add_argument("--video", default=config.VIDEO_PATH, help="Path to source video")
    ap.add_argument("--json_dir", default=config.OUTPUT_DIR, help="Directory containing frame_XXXXXX.json files")
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "annotated_output.mp4"),
                    help="Output annotated video path")
    ap.add_argument("--sample_fps", type=float, default=config.SAMPLE_FPS,
                    help="Must match sample_fps used when JSONs were generated (default from config)")
    args = ap.parse_args()

    frames_data = load_frame_jsons(args.json_dir)
    print(f"Loaded {len(frames_data)} annotated frames from {args.json_dir}")
    if not frames_data:
        print("[ERROR] No frame_XXXXXX.json files found — make sure to run pipeline first.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {args.video}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(int(round(src_fps / args.sample_fps)), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Use mp4v or fallback codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.sample_fps, (width, height))

    sample_idx = 0
    src_frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if src_frame_idx % frame_interval == 0:
            data = frames_data.get(sample_idx)
            if data is not None:
                frame = draw_frame(frame, data)
                cv2.putText(
                    frame,
                    f"Frame: {sample_idx} | Time: {data.get('frame_ts', 0):.2f}s | Persons: {len(data.get('tracks', []))}",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                written += 1
            sample_idx += 1
        src_frame_idx += 1

    cap.release()
    writer.release()
    print(f"[SUCCESS] Wrote {written} annotated frames to {args.out}")


if __name__ == "__main__":
    main()
