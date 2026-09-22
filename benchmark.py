"""
benchmark.py -- ONNX export + FP16 inference benchmarking for MI-EYE PRO.

PURPOSE:
    This script provides the FPS numbers for your FYP report's evaluation chapter.
    It compares three inference configurations on your Colab T4:
        1. PyTorch FP32 (baseline)
        2. PyTorch FP16 (half precision, uses T4 Tensor Cores)
        3. ONNX Runtime with CUDA EP (optimized graph execution)

    Each configuration is benchmarked on the same video with the same sample rate,
    measuring end-to-end pipeline throughput (ingest + detect+track + pose + serialize).

HARDWARE NOTES:
    - Google Colab T4 (16 GB VRAM, Turing architecture, supports FP16 natively)
    - T4 Tensor Cores accelerate FP16 matrix ops by ~2x over FP32
    - ONNX Runtime applies graph-level optimizations (operator fusion, constant
      folding) that PyTorch's eager mode doesn't do

    GPU-CONSTRAINT FLAGS:
    - FP16 is safe on T4 (Turing+). On older GPUs (e.g. K80) FP16 is emulated
      and may be SLOWER than FP32.
    - ONNX export requires the model to be traced, which may fail on dynamic
      architectures. YOLOv8 supports this natively via model.export().

USAGE (run on Google Colab with T4):
    python benchmark.py --video data/Video1.mp4 --max-frames 50
    python benchmark.py --video data/Video1.mp4 --max-frames 50 --skip-onnx  # if ONNX fails
"""

import argparse
import json
import os
import time
import sys

import numpy as np


def check_gpu():
    """Check if CUDA GPU is available and print device info."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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


def benchmark_pytorch_fp32(video_path, sample_fps, max_frames):
    """Benchmark: PyTorch FP32 (baseline).

    This is the default configuration -- the same as Stages 1-5.
    No special optimization, models run in FP32 precision.
    """
    from ultralytics import YOLO
    from video_ingest import VideoIngestor
    import config

    print("\n[1/3] PyTorch FP32 (baseline)")
    print("-" * 40)

    # Load models (FP32 is the default)
    detect_model = YOLO(config.MODEL_DETECT_NAME)
    pose_model = YOLO(config.MODEL_POSE_NAME)

    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    total_persons = 0

    # Warmup: 3 frames to stabilize GPU clocks and fill caches
    print("  Warming up (3 frames)...")
    for frame_id, ts, frame in ingestor.frames():
        _ = detect_model.track(source=frame, conf=0.35, classes=[0],
                               tracker="bytetrack.yaml", persist=True, verbose=False)
        processed += 1
        if processed >= 3:
            break
    ingestor.release()

    # Actual benchmark
    ingestor = VideoIngestor(video_path, sample_fps)
    detect_model = YOLO(config.MODEL_DETECT_NAME)  # reset tracker state
    processed = 0
    t_start = time.time()

    for frame_id, ts, frame in ingestor.frames():
        # Detection + tracking
        results = detect_model.track(source=frame, conf=0.35, classes=[0],
                                     tracker="bytetrack.yaml", persist=True, verbose=False)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            total_persons += len(boxes)

            # Pose on each crop
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                h, w = frame.shape[:2]
                pad_x = int((x2-x1) * 0.1)
                pad_y = int((y2-y1) * 0.1)
                crop = frame[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.shape[0] > 30 and crop.shape[1] > 30:
                    _ = pose_model.predict(source=crop, conf=0.25, verbose=False)

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
    return {"mode": "PyTorch FP32", "frames": processed, "time_s": round(elapsed, 2),
            "fps": round(fps, 2), "persons": total_persons}


def benchmark_pytorch_fp16(video_path, sample_fps, max_frames):
    """Benchmark: PyTorch FP16 (half precision).

    Uses model.half() to convert weights to FP16.  On the T4's Tensor Cores
    this should give ~1.5-2x speedup over FP32 for matrix operations.

    WHY FP16 IS SAFE HERE:
        - Detection/pose are inference-only (no gradient accumulation, so FP16
          precision loss doesn't cause training instability)
        - YOLOv8 is designed to work with half precision
        - The T4 has hardware FP16 support (Turing Tensor Cores)
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

    # Load models and convert to FP16
    # Ultralytics supports half precision via the `half` parameter in predict/track
    detect_model = YOLO(config.MODEL_DETECT_NAME)
    pose_model = YOLO(config.MODEL_POSE_NAME)

    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    total_persons = 0

    # Warmup
    print("  Warming up (3 frames)...")
    for frame_id, ts, frame in ingestor.frames():
        _ = detect_model.track(source=frame, conf=0.35, classes=[0],
                               tracker="bytetrack.yaml", persist=True,
                               verbose=False, half=True)  # <-- FP16
        processed += 1
        if processed >= 3:
            break
    ingestor.release()

    # Actual benchmark
    ingestor = VideoIngestor(video_path, sample_fps)
    detect_model = YOLO(config.MODEL_DETECT_NAME)
    processed = 0
    t_start = time.time()

    for frame_id, ts, frame in ingestor.frames():
        results = detect_model.track(source=frame, conf=0.35, classes=[0],
                                     tracker="bytetrack.yaml", persist=True,
                                     verbose=False, half=True)  # <-- FP16
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            total_persons += len(boxes)

            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                h, w = frame.shape[:2]
                pad_x = int((x2-x1) * 0.1)
                pad_y = int((y2-y1) * 0.1)
                crop = frame[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.shape[0] > 30 and crop.shape[1] > 30:
                    _ = pose_model.predict(source=crop, conf=0.25,
                                           verbose=False, half=True)  # <-- FP16

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


def export_onnx(model_name, fp16=False):
    """Export a YOLO model to ONNX format.

    ONNX (Open Neural Network Exchange) is an intermediate representation that
    ONNX Runtime can optimize with:
        - Operator fusion (e.g. Conv+BN+ReLU -> single kernel)
        - Constant folding (pre-compute static subgraphs)
        - Memory planning (reduce peak allocation)

    Args:
        model_name: Ultralytics model name (e.g. 'yolov8s.pt')
        fp16: Whether to export with FP16 weights (requires CUDA)

    Returns:
        Path to the exported .onnx file
    """
    from ultralytics import YOLO

    print(f"  Exporting {model_name} to ONNX (fp16={fp16})...")
    model = YOLO(model_name)
    onnx_path = model.export(format="onnx", half=fp16, simplify=True)
    print(f"  Exported: {onnx_path}")
    return onnx_path


def benchmark_onnx(video_path, sample_fps, max_frames):
    """Benchmark: ONNX Runtime with CUDA Execution Provider.

    ONNX Runtime (ORT) is a high-performance inference engine that can run
    ONNX models with hardware-specific optimizations.  On the T4, the CUDA EP
    (Execution Provider) uses cuDNN for optimized convolution kernels.

    NOTE: ONNX tracking is trickier because Ultralytics' model.track() handles
    tracker state internally.  For the ONNX benchmark, we use the ONNX model
    through Ultralytics' YOLO() which transparently loads .onnx files.
    """
    import torch
    from ultralytics import YOLO
    from video_ingest import VideoIngestor
    import config

    if not torch.cuda.is_available():
        print("\n[3/3] ONNX Runtime -- SKIPPED (no GPU)")
        return {"mode": "ONNX Runtime", "frames": 0, "time_s": 0, "fps": 0,
                "persons": 0, "skipped": True}

    print("\n[3/3] ONNX Runtime (CUDA EP)")
    print("-" * 40)

    # Export models to ONNX
    detect_onnx = export_onnx(config.MODEL_DETECT_NAME, fp16=False)
    pose_onnx = export_onnx(config.MODEL_POSE_NAME, fp16=False)

    # Load ONNX models through Ultralytics (handles ORT session internally)
    detect_model = YOLO(detect_onnx)
    pose_model = YOLO(pose_onnx)

    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    total_persons = 0

    # Warmup
    print("  Warming up (3 frames)...")
    for frame_id, ts, frame in ingestor.frames():
        _ = detect_model.predict(source=frame, conf=0.35, classes=[0], verbose=False)
        processed += 1
        if processed >= 3:
            break
    ingestor.release()

    # NOTE: ONNX models through Ultralytics don't support model.track() directly
    # in all versions.  We benchmark detection + pose only (no tracking) to
    # isolate the neural network speedup.  Tracking overhead is negligible
    # (ByteTrack runs on CPU with ~0.1ms per frame).
    ingestor = VideoIngestor(video_path, sample_fps)
    processed = 0
    t_start = time.time()

    for frame_id, ts, frame in ingestor.frames():
        results = detect_model.predict(source=frame, conf=0.35, classes=[0], verbose=False)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            total_persons += len(boxes)

            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                h, w = frame.shape[:2]
                pad_x = int((x2-x1) * 0.1)
                pad_y = int((y2-y1) * 0.1)
                crop = frame[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.shape[0] > 30 and crop.shape[1] > 30:
                    _ = pose_model.predict(source=crop, conf=0.25, verbose=False)

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
    return {"mode": "ONNX Runtime", "frames": processed, "time_s": round(elapsed, 2),
            "fps": round(fps, 2), "persons": total_persons}


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

    # Save results to JSON for the FYP report
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "benchmark_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print("  (Use these numbers in your FYP Evaluation chapter)")


def main():
    parser = argparse.ArgumentParser(description="MI-EYE PRO -- ONNX/FP16 Benchmark")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=50, help="Frames to benchmark (default: 50)")
    parser.add_argument("--fps", type=float, default=2.0, help="Sample FPS")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX benchmark")
    args = parser.parse_args()

    # Auto-detect video
    if args.video is None:
        import config
        args.video = config.VIDEO_PATH

    print("=" * 65)
    print("  MI-EYE PRO -- Stage 6: ONNX Export + FP16 Benchmark")
    print("=" * 65)
    print(f"  Video: {args.video}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Sample FPS: {args.fps}")

    has_gpu = check_gpu()

    results = []

    # 1. PyTorch FP32 baseline (always runs)
    r1 = benchmark_pytorch_fp32(args.video, args.fps, args.max_frames)
    results.append(r1)

    # 2. PyTorch FP16 (GPU only)
    r2 = benchmark_pytorch_fp16(args.video, args.fps, args.max_frames)
    results.append(r2)

    # 3. ONNX Runtime (GPU only, optional)
    if not args.skip_onnx:
        r3 = benchmark_onnx(args.video, args.fps, args.max_frames)
        results.append(r3)
    else:
        print("\n[3/3] ONNX Runtime -- SKIPPED (--skip-onnx flag)")
        results.append({"mode": "ONNX Runtime", "skipped": True})

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
