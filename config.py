"""
config.py -- Central configuration for the MI-EYE PRO perception pipeline.

GPU AUTO-DETECTION:
    This module probes the available GPU at import time and selects model sizes,
    tracker type, and sample FPS accordingly.  Three tiers:

    Tier 1 (>= 12 GB VRAM):  e.g. T4, RTX 3060 12GB, A100
        - YOLOv8s (small) for detection and pose
        - BoT-SORT (appearance re-ID) for tracking
        - 6 FPS sampling
        - FP16 enabled (if Tensor Cores available)

    Tier 2 (4-11 GB VRAM):  e.g. GTX 1650, RTX 2060, GTX 1080
        - YOLOv8n (nano) for detection and pose
        - BoT-SORT (appearance re-ID) — re-ID model is only ~300 MB
        - 4 FPS sampling
        - FP16 only if compute capability >= 7.0 AND has Tensor Cores (RTX)

    Tier 3 (CPU only / < 4 GB):
        - YOLOv8n (nano) for detection and pose
        - ByteTrack (motion only, no VRAM cost)
        - 2 FPS sampling
        - FP16 disabled

    GPU-CONSTRAINT FLAGS are documented inline for FYP defense.
"""

import os

# ──────────────────────────────────────────────
# GPU Auto-Detection
# ──────────────────────────────────────────────

def _detect_gpu_tier():
    """Probe the GPU and return (tier, device_name, vram_gb, has_tensor_cores).

    Tier 1: >= 12 GB VRAM
    Tier 2: 4-11 GB VRAM
    Tier 3: CPU or < 4 GB
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 3, "CPU", 0, False

        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        vram_bytes = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        vram_gb = vram_bytes / (1024 ** 3)

        # Tensor Cores: available on compute capability >= 7.0 AND RTX cards
        # GTX 1650 is Turing (7.5) but has NO Tensor Cores (only RTX 20xx+ do)
        # We check for "RTX" or known Tensor Core GPUs (T4, A100, etc.)
        cc = (props.major, props.minor)
        has_tensor_cores = cc >= (7, 0) and (
            "RTX" in device_name.upper() or
            "T4" in device_name.upper() or
            "A100" in device_name.upper() or
            "A10" in device_name.upper() or
            "H100" in device_name.upper() or
            "L4" in device_name.upper() or
            "L40" in device_name.upper()
        )

        if vram_gb >= 12:
            tier = 1
        elif vram_gb >= 4:
            tier = 2
        else:
            tier = 3

        return tier, device_name, vram_gb, has_tensor_cores

    except ImportError:
        return 3, "CPU (no PyTorch)", 0, False


# Run detection at import time
GPU_TIER, GPU_NAME, GPU_VRAM_GB, HAS_TENSOR_CORES = _detect_gpu_tier()


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "Video1.mp4")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FACE_CROPS_DIR = os.path.join(OUTPUT_DIR, "face_crops")

# ──────────────────────────────────────────────
# Session / Camera identifiers
# ──────────────────────────────────────────────
SESSION_ID = "session_001"
CAMERA_ID = "cam_01"

# ──────────────────────────────────────────────
# Tiered Configuration
# ──────────────────────────────────────────────

if GPU_TIER == 1:
    # Tier 1: >= 12 GB VRAM (T4, A100, RTX 3060 12GB+)
    # Use small models for best accuracy, BoT-SORT for re-ID
    MODEL_DETECT_NAME = "yolov8s.pt"
    MODEL_POSE_NAME = "yolov8s-pose.pt"
    TRACKER_CONFIG = os.path.join(PROJECT_ROOT, "tracker_config", "botsort_custom.yaml")
    SAMPLE_FPS = 6.0
    USE_FP16 = HAS_TENSOR_CORES   # FP16 only useful with Tensor Cores

elif GPU_TIER == 2:
    # Tier 2: 4-11 GB VRAM (GTX 1650, RTX 2060, GTX 1080)
    # GPU-CONSTRAINT FLAG: Nano models (~1.5 GB each) fit within 4 GB VRAM
    # alongside BoT-SORT re-ID (~300 MB) and PyTorch overhead (~500 MB).
    # Accuracy tradeoff: mAP 37.3 (nano) vs 44.9 (small) — acceptable for
    # well-lit classrooms with clear sightlines.
    MODEL_DETECT_NAME = "yolov8n.pt"
    MODEL_POSE_NAME = "yolov8n-pose.pt"
    TRACKER_CONFIG = os.path.join(PROJECT_ROOT, "tracker_config", "botsort_custom.yaml")
    SAMPLE_FPS = 4.0
    USE_FP16 = HAS_TENSOR_CORES   # GTX 1650 → False; RTX 2060 → True

else:
    # Tier 3: CPU only or < 4 GB VRAM
    # GPU-CONSTRAINT FLAG: No VRAM for BoT-SORT's re-ID model, fall back to
    # ByteTrack (pure motion, runs on CPU).
    MODEL_DETECT_NAME = "yolov8n.pt"
    MODEL_POSE_NAME = "yolov8n-pose.pt"
    TRACKER_CONFIG = os.path.join(PROJECT_ROOT, "tracker_config", "bytetrack_custom.yaml")
    SAMPLE_FPS = 2.0
    USE_FP16 = False

# ──────────────────────────────────────────────
# ONNX Export (Stage 6)
# ──────────────────────────────────────────────
ONNX_EXPORT = False  # Flip to True after benchmarking


def print_config():
    """Print the active configuration for debugging / logging."""
    print(f"  GPU Tier       : {GPU_TIER} ({GPU_NAME}, {GPU_VRAM_GB:.1f} GB)")
    print(f"  Tensor Cores   : {HAS_TENSOR_CORES}")
    print(f"  Detect Model   : {MODEL_DETECT_NAME}")
    print(f"  Pose Model     : {MODEL_POSE_NAME}")
    print(f"  Tracker        : {os.path.basename(TRACKER_CONFIG)}")
    print(f"  Sample FPS     : {SAMPLE_FPS}")
    print(f"  FP16           : {USE_FP16}")
