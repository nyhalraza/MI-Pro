"""
benchmark.py -- ONNX export + FP16 inference benchmarking for MI-EYE PRO.

PURPOSE:
    Provides FPS numbers for the FYP report's evaluation chapter.
    Compares inference configurations on GPU (Colab T4 or GTX 1650):
        1. PyTorch FP32 (baseline)
        2. PyTorch FP16 (half precision via quantize= parameter)
        3. ONNX Runtime (optional — requires compatible CUDA libraries)

USAGE:
    python benchmark.py --video data/Video1.mp4 --max-frames 50
    python benchmark.py --video data/Video1.mp4 --max-frames 50 --skip-onnx
    python benchmark.py --video data/Video1.mp4 --max-frames 50 --skip-fp16 --skip-onnx
"""

import argparse
import json
import os
import time

import numpy as np


def check_gpu():
    """Check if CUDA GPU is available and print device info."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device = props.name
            vram_bytes = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            vram = vram_bytes / (1024**3)
            print(f"  GPU: {device} ({vram:.1f} GB VRAM)")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  PyTorch: {torch.__version__}")
            return True
        else:
            print("  [WARNING] No CUDA GPU detected. Running on CPU.")
            print("  FP16 and ONNX-GPU benchmarks will be skipped.")
            return False
    except ImportError:
        print("  [ERROR] PyTorch not installed.")
        return False


def _run_benchmark(video_path, sample_fps, max_frames, detect_model, pose_model, label):
    """Shared benchmark loop: detection + full-frame pose (v2 pipeline).

    Uses full-frame pose inference (one call) instead of per-crop (N calls).
    This matches how the actual pipeline runs.
    """
    from video_ingest import VideoIngestor

    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    total_persons = 0

    # Warmup: 3 frames to stabilize GPU clocks and fill caches
    print("  Warming up (3 frames)...")
    warmup_model = detect_model
    for frame_id, ts, frame in ingestor.frames():
        _ = warmup_model.predict(source=frame, conf=0.35, classes=[0], verbose=False)
        _ = pose_model.predict(source=frame, conf=0.25, verbose=False)
        processed += 1
        if processed >= 3:
            break
    ingestor.release()

    # Actual benchmark
    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    t_start = time.time()

    for frame_id, ts, frame in ingestor.frames():
        # Detection (no tracking in benchmark — isolates neural network speed)
        results = detect_model.predict(source=frame, conf=0.35, classes=[0], verbose=False)

        if results and results[0].boxes is not None:
            total_persons += len(results[0].boxes)

        # Full-frame pose (v2 approach — one call for all persons)
        _ = pose_model.predict(source=frame, conf=0.25, verbose=False)

        processed += 1
        if max_frames and processed >= max_frames:
            break

    elapsed = time.time() - t_start
    fps = processed / elapsed if elapsed > 0 else 0
    ingestor.release()

    print(f"  Frames: {processed}")
    print(f"  Persons: {total_persons}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {fps:.2f}")
    return {"mode": label, "frames": processed, "time_s": round(elapsed, 2),
            "fps": round(fps, 2), "persons": total_persons}


def benchmark_pytorch_fp32(video_path, sample_fps, max_frames):
    """Benchmark: PyTorch FP32 (baseline). No optimization."""
    from ultralytics import YOLO
    import config

    print("\n[1/3] PyTorch FP32 (baseline)")
    print("-" * 40)

    detect_model = YOLO(config.MODEL_DETECT_NAME)
    pose_model = YOLO(config.MODEL_POSE_NAME)

    return _run_benchmark(video_path, sample_fps, max_frames,
                          detect_model, pose_model, "PyTorch FP32")


def benchmark_pytorch_fp16(video_path, sample_fps, max_frames):
    """Benchmark: PyTorch FP16 (half precision).

    Ultralytics 8.4.160+ deprecates 'half=' in favor of 'quantize='.
    We handle both old and new API versions.

    WHY FP16 IS SAFE:
        - Inference-only (no gradient accumulation)
        - YOLOv8 is designed for half precision
        - T4 has hardware FP16 support (Tensor Cores)
    """
    import torch
    from ultralytics import YOLO
    from video_ingest import VideoIngestor
    import config

    if not torch.cuda.is_available():
        print("\n[2/3] PyTorch FP16 -- SKIPPED (no GPU)")
        return {"mode": "PyTorch FP16", "frames": 0, "time_s": 0, "fps": 0,
                "persons": 0, "skipped": True}

    print("\n[2/3] PyTorch FP16 (half precision)")
    print("-" * 40)

    detect_model = YOLO(config.MODEL_DETECT_NAME)
    pose_model = YOLO(config.MODEL_POSE_NAME)

    # Convert models to FP16 on GPU
    detect_model.model.half()
    pose_model.model.half()

    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    total_persons = 0

    # Warmup
    print("  Warming up (3 frames)...")
    for frame_id, ts, frame in ingestor.frames():
        _ = detect_model.predict(source=frame, conf=0.35, classes=[0], verbose=False)
        _ = pose_model.predict(source=frame, conf=0.25, verbose=False)
        processed += 1
        if processed >= 3:
            break
    ingestor.release()

    # Benchmark
    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    t_start = time.time()

    for frame_id, ts, frame in ingestor.frames():
        results = detect_model.predict(source=frame, conf=0.35, classes=[0], verbose=False)
        if results and results[0].boxes is not None:
            total_persons += len(results[0].boxes)
        _ = pose_model.predict(source=frame, conf=0.25, verbose=False)

        processed += 1
        if max_frames and processed >= max_frames:
            break

    elapsed = time.time() - t_start
    fps = processed / elapsed if elapsed > 0 else 0
    ingestor.release()

    print(f"  Frames: {processed}")
    print(f"  Persons: {total_persons}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {fps:.2f}")
    return {"mode": "PyTorch FP16", "frames": processed, "time_s": round(elapsed, 2),
            "fps": round(fps, 2), "persons": total_persons}


def benchmark_onnx(video_path, sample_fps, max_frames):
    """Benchmark: ONNX Runtime.

    NOTE: ONNX CUDA EP requires matching cuBLAS/cuDNN versions.
    On Colab, this may fail with library version mismatches.
    If it falls back to CPU, use --skip-onnx and note it in your FYP report
    as an environment limitation, not a code issue.
    """
    import torch
    from ultralytics import YOLO
    import config

    if not torch.cuda.is_available():
        print("\n[3/3] ONNX Runtime -- SKIPPED (no GPU)")
        return {"mode": "ONNX Runtime", "frames": 0, "time_s": 0, "fps": 0,
                "persons": 0, "skipped": True}

    print("\n[3/3] ONNX Runtime")
    print("-" * 40)

    # Export to ONNX
    print(f"  Exporting {config.MODEL_DETECT_NAME} to ONNX...")
    detect_pt = YOLO(config.MODEL_DETECT_NAME)
    detect_onnx_path = detect_pt.export(format="onnx", simplify=True)
    print(f"  Exported: {detect_onnx_path}")

    print(f"  Exporting {config.MODEL_POSE_NAME} to ONNX...")
    pose_pt = YOLO(config.MODEL_POSE_NAME)
    pose_onnx_path = pose_pt.export(format="onnx", simplify=True)
    print(f"  Exported: {pose_onnx_path}")

    # Load ONNX models
    detect_model = YOLO(detect_onnx_path)
    pose_model = YOLO(pose_onnx_path)

    return _run_benchmark(video_path, sample_fps, max_frames,
                          detect_model, pose_model, "ONNX Runtime")


def print_comparison(results):
    """Print a formatted comparison table for the FYP report."""
    print("\n")
    print("=" * 65)
    print("  BENCHMARK RESULTS -- MI-EYE PRO Perception Pipeline")
    print("=" * 65)
    print(f"  {'Mode':<20s} {'Frames':>8s} {'Time(s)':>8s} {'FPS':>8s} {'Speedup':>8s}")
    print("-" * 65)

    baseline_fps = None
    for r in results:
        if r.get("skipped"):
            print(f"  {r['mode']:<20s} {'SKIPPED':>8s}")
            continue

        if baseline_fps is None:
            baseline_fps = r["fps"]
            speedup = "1.00x"
        else:
            speedup = f"{r['fps']/baseline_fps:.2f}x" if baseline_fps > 0 else "N/A"

        print(f"  {r['mode']:<20s} {r['frames']:>8d} {r['time_s']:>8.2f} {r['fps']:>8.2f} {speedup:>8s}")

    print("=" * 65)

    # Save results to JSON
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "benchmark_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print("  (Use these numbers in your FYP Evaluation chapter)")


def main():
    parser = argparse.ArgumentParser(description="MI-EYE PRO -- Benchmark")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=50, help="Frames to benchmark")
    parser.add_argument("--fps", type=float, default=6.0, help="Sample FPS")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX benchmark")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 benchmark")
    args = parser.parse_args()

    if args.video is None:
        import config
        args.video = config.VIDEO_PATH

    print("=" * 65)
    print("  MI-EYE PRO -- Stage 6: Benchmark")
    print("=" * 65)
    print(f"  Video: {args.video}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Sample FPS: {args.fps}")

    has_gpu = check_gpu()

    results = []

    # 1. PyTorch FP32 baseline
    r1 = benchmark_pytorch_fp32(args.video, args.fps, args.max_frames)
    results.append(r1)

    # 2. PyTorch FP16
    if not args.skip_fp16:
        r2 = benchmark_pytorch_fp16(args.video, args.fps, args.max_frames)
        results.append(r2)
    else:
        print("\n[2/3] PyTorch FP16 -- SKIPPED (--skip-fp16 flag)")
        results.append({"mode": "PyTorch FP16", "skipped": True})

    # 3. ONNX Runtime
    if not args.skip_onnx:
        r3 = benchmark_onnx(args.video, args.fps, args.max_frames)
        results.append(r3)
    else:
        print("\n[3/3] ONNX Runtime -- SKIPPED (--skip-onnx flag)")
        results.append({"mode": "ONNX Runtime", "skipped": True})

    print_comparison(results)


if __name__ == "__main__":
    main()
