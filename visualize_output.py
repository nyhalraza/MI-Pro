"""
visualize_output.py

Draws the saved per-frame detection JSONs (bbox, keypoints, track_id) back
onto the source video frames, and writes an annotated .mp4 you can watch.

No GPU / inference needed — this only reads the JSON files your pipeline
already produced in output/ and paints them onto matching frames from the
source video. Safe to run on CPU, Colab free tier, or your own laptop.

Usage:
    python visualize_output.py \
        --video data/Video1.mp4 \
        --json_dir output \
        --out output/annotated_Video1.mp4 \
        --sample_fps 2.0
"""

import argparse
import json
import os
import re

import cv2

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
# across the whole clip
def color_for_track(track_id: int):
    palette = [
        (66, 135, 245), (66, 245, 111), (245, 66, 66), (245, 200, 66),
        (188, 66, 245), (66, 245, 212), (245, 66, 173), (150, 245, 66),
    ]
    return palette[track_id % len(palette)]


def load_frame_jsons(json_dir):
    """Returns a dict: frame_id -> parsed JSON."""
    pattern = re.compile(r"frame_(\d+)\.json$")
    frames = {}
    for fname in os.listdir(json_dir):
        m = pattern.match(fname)
        if not m:
            continue
        frame_id = int(m.group(1))
        with open(os.path.join(json_dir, fname)) as f:
            frames[frame_id] = json.load(f)
    return frames


def draw_frame(img, frame_data):
    for track in frame_data.get("tracks", []):
        tid = track["track_id"]
        color = color_for_track(tid)

        # bbox
        x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid} ({track.get('confidence', 0):.2f})"
        cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # keypoints
        kp_by_name = {kp["name"]: kp for kp in track.get("keypoints", [])}
        for kp in kp_by_name.values():
            if kp["conf"] < 0.3:
                continue
            cv2.circle(img, (int(kp["x"]), int(kp["y"])), 3, color, -1)

        # skeleton lines
        for a, b in SKELETON:
            ka, kb = kp_by_name.get(a), kp_by_name.get(b)
            if not ka or not kb:
                continue
            if ka["conf"] < 0.3 or kb["conf"] < 0.3:
                continue
            cv2.line(img, (int(ka["x"]), int(ka["y"])),
                      (int(kb["x"]), int(kb["y"])), color, 2)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to source video (e.g. data/Video1.mp4)")
    ap.add_argument("--json_dir", required=True, help="Directory containing frame_XXXXXX.json files")
    ap.add_argument("--out", required=True, help="Output annotated video path")
    ap.add_argument("--sample_fps", type=float, default=2.0,
                     help="Must match the sample_fps used when the JSONs were generated")
    args = ap.parse_args()

    frames_data = load_frame_jsons(args.json_dir)
    print(f"Loaded {len(frames_data)} annotated frames from {args.json_dir}")
    if not frames_data:
        print("No frame_XXXXXX.json files found — check --json_dir path.")
        return

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(round(src_fps / args.sample_fps)), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video plays at the SAMPLE fps (e.g. 2 fps), so each annotated
    # frame is clearly visible rather than flashing by at source fps
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.sample_fps, (width, height)
    )

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
                cv2.putText(frame, f"t={data['frame_ts']:.2f}s  frame_id={sample_idx}",
                            (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
            writer.write(frame)
            written += 1
            sample_idx += 1
        src_frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {written} annotated frames to {args.out}")


if __name__ == "__main__":
    main()
