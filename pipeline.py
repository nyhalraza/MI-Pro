"""
pipeline.py -- Main orchestrator for the MI-EYE PRO perception pipeline.

Stages:
    Stage 1: Ingest only                   [DONE]
    Stage 2: + Detection                   [DONE]
    Stage 3: + Tracking (BoT-SORT/ByteTrack) [DONE - upgraded to BoT-SORT]
    Stage 4: + Pose estimation             [DONE - upgraded to full-frame]
    Stage 5: + JSON serialization          [DONE]
    Stage 6: + ONNX / FP16 optimization   [DONE]

v2 CHANGES:
    - GPU auto-detection: model sizes adapt to available VRAM
    - BoT-SORT: appearance re-ID for stable IDs through occlusion
    - Full-frame pose: one inference call instead of N per-crop calls (3-5x faster)
    - Higher FPS: 6 FPS (T4) / 4 FPS (1650) / 2 FPS (CPU)
"""

import os
import time

import config
from video_ingest import VideoIngestor
from tracker import TrackedDetector
from pose_estimator import PoseEstimator
from serializer import FrameSerializer
from schemas import FrameResult, Track
from utils import ensure_dir


class PerceptionPipeline:
    """Orchestrates: ingest -> detect+track (BoT-SORT) -> pose (full-frame) -> JSON."""

    def __init__(
        self,
        video_source: str = config.VIDEO_PATH,
        sample_fps: float = config.SAMPLE_FPS,
        session_id: str = config.SESSION_ID,
        camera_id: str = config.CAMERA_ID,
    ):
        self.video_source = video_source
        self.sample_fps = sample_fps
        self.session_id = session_id
        self.camera_id = camera_id

        ensure_dir(config.OUTPUT_DIR)

        # Print active configuration
        print("=" * 60)
        print("MI-EYE PRO -- Configuration")
        print("=" * 60)
        config.print_config()
        print("=" * 60)

        # Initialize tracked detector
        self.tracked_detector = TrackedDetector(
            model_name=config.MODEL_DETECT_NAME,
            tracker_config=config.TRACKER_CONFIG,
            conf_threshold=0.35,
        )

        # Initialize full-frame pose estimator
        self.pose_estimator = PoseEstimator(
            model_name=config.MODEL_POSE_NAME,
        )

        # Initialize JSON serializer
        self.serializer = FrameSerializer(
            output_dir=config.OUTPUT_DIR,
        )

    def run(self, max_frames: int = None):
        """Run the full pipeline.

        Args:
            max_frames: Stop after N sampled frames (None = process entire video).
        """
        ingestor = VideoIngestor(self.video_source, self.sample_fps)
        info = ingestor.info()
        print("=" * 60)
        print("MI-EYE PRO -- Pipeline Running")
        print("=" * 60)
        for k, v in info.items():
            print(f"  {k:>20s}: {v}")
        print("=" * 60)

        processed = 0
        total_detections = 0
        total_keypoints_detected = 0
        all_track_ids = set()
        t_start = time.time()

        for frame_id, timestamp, frame in ingestor.frames():
            # ---- Detection + Tracking ----
            tracked_dets = self.tracked_detector.track(frame, persist=True)

            # ---- Full-frame Pose Estimation ----
            # Extract bboxes for batch pose matching
            bboxes = [det["bbox"] for det in tracked_dets]
            all_keypoints = self.pose_estimator.estimate_batch(frame, bboxes)

            # ---- Assemble Track objects ----
            tracks = []
            for i, det in enumerate(tracked_dets):
                keypoints = all_keypoints[i]
                kp_detected = sum(1 for kp in keypoints if kp.conf > 0)
                total_keypoints_detected += kp_detected

                track = Track(
                    track_id=det["track_id"],
                    cls="person",
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    keypoints=keypoints,
                    face_crop_path=None,
                )
                tracks.append(track)
                all_track_ids.add(det["track_id"])

            # ---- Serialize to JSON ----
            result = FrameResult(
                frame_id=frame_id,
                frame_ts=timestamp,
                session_id=self.session_id,
                camera_id=self.camera_id,
                tracks=tracks,
            )
            json_path = self.serializer.write(result)

            total_detections += len(tracks)

            # ---- Log ----
            kp_summary = []
            for t in tracks:
                kp_count = sum(1 for kp in t.keypoints if kp.conf > 0)
                kp_summary.append(f"T{t.track_id}:{kp_count}/13")

            print(
                f"[Frame {result.frame_id:>5d}]  "
                f"ts={result.frame_ts:>8.2f}s  "
                f"persons={len(tracks):>2d}  "
                f"kps=[{', '.join(kp_summary)}]  "
                f"-> {os.path.basename(json_path)}"
            )

            processed += 1
            if max_frames and processed >= max_frames:
                print(f"\n[STOP] Stopped after {max_frames} frames (max_frames limit).")
                break

        elapsed = time.time() - t_start
        throughput = processed / elapsed if elapsed > 0 else 0
        avg_kp = total_keypoints_detected / total_detections if total_detections > 0 else 0

        print("=" * 60)
        print(f"  Frames processed  : {processed}")
        print(f"  JSON files written: {self.serializer.frames_written}")
        print(f"  Total detections  : {total_detections}")
        print(f"  Unique track IDs  : {len(all_track_ids)} -> {sorted(all_track_ids)}")
        print(f"  Avg kpts/person   : {avg_kp:.1f}/13")
        print(f"  Wall time         : {elapsed:.2f}s")
        print(f"  Throughput        : {throughput:.1f} sampled frames/sec")
        print(f"  Output directory  : {config.OUTPUT_DIR}")
        print("=" * 60)
        print(f"\nRun 'python validate_output.py' to verify contract compliance.")

        ingestor.release()
        return processed
