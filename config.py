"""
config.py — Central configuration for the MI-EYE PRO perception pipeline.

WHY A SINGLE CONFIG FILE?
    Every stage (ingest, detection, tracking, pose, serialization) reads from here,
    so changing a path or model size is a one-line edit, not a grep-and-replace across
    five files.  For a 4-person team this also prevents "works on my machine" drift.

DESIGN NOTE (GPU CONSTRAINT):
    All model-size and batch-size defaults are tuned for a Google Colab T4 (16 GB VRAM).
    If you switch to a different GPU, revisit MODEL_DETECT_NAME, MODEL_POSE_NAME,
    and SAMPLE_FPS.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default test video — download_sample_video.py puts it here
VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "Video1.mp4")

# Where per-frame JSON and face crops are written
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FACE_CROPS_DIR = os.path.join(OUTPUT_DIR, "face_crops")

# ──────────────────────────────────────────────
# Session / Camera identifiers
# ──────────────────────────────────────────────
# These get embedded in every JSON frame so downstream modules (attendance,
# behavior analysis) can associate detections with the right classroom/session.
SESSION_ID = "session_001"
CAMERA_ID = "cam_01"

# ──────────────────────────────────────────────
# Video Ingest
# ──────────────────────────────────────────────
# SAMPLE_FPS controls how many frames per second we actually process.
#
# WHY 2 FPS?
#   On a T4 with 16 GB VRAM we *could* run higher (up to ~8-10 FPS with YOLOv8n),
#   but 2 FPS is a safe starting point that leaves headroom for detection + pose
#   on the same GPU.  We'll benchmark in Stage 6 and raise this if the numbers
#   support it.
#
# GPU-CONSTRAINT FLAG:
#   If running on a 4 GB card (e.g. free-tier K80), drop this to 1 FPS.
SAMPLE_FPS = 2.0

# ──────────────────────────────────────────────
# Detection (Stage 2)
# ──────────────────────────────────────────────
MODEL_DETECT_NAME = "yolov8s.pt"   # small — see detector.py for size rationale

# ──────────────────────────────────────────────
# Tracking (Stage 3)
# ──────────────────────────────────────────────
TRACKER_CONFIG = "bytetrack.yaml"  # ships with ultralytics

# ──────────────────────────────────────────────
# Pose Estimation (Stage 4)
# ──────────────────────────────────────────────
MODEL_POSE_NAME = "yolov8s-pose.pt"  # small-pose — matches detect model size

# ──────────────────────────────────────────────
# Optimization (Stage 6)
# ──────────────────────────────────────────────
USE_FP16 = True          # T4 Tensor Cores support FP16 natively
ONNX_EXPORT = False      # Flip to True after Stage 5 is validated
