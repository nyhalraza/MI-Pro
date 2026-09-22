"""
pipeline.py -- Main orchestrator for the MI-EYE PRO perception pipeline.

This file grows with each stage:
    Stage 1: Ingest only                   [DONE]
    Stage 2: + Detection                   [DONE]
    Stage 3: + Tracking                    [DONE]
    Stage 4: + Pose estimation             [DONE]
    Stage 5: + JSON serialization          [CURRENT]
    Stage 6: + ONNX / FP16 optimization

Each stage is a method that transforms the frame data, and run() chains them.
"""

import json
import time

import config
from video_ingest import VideoIngestor
from tracker import TrackedDetector
from pose_estimator import PoseEstimator
from serializer import FrameSerializer
from schemas import FrameResult, Track, Keypoint, KEYPOINT_NAMES
from utils import ensure_dir


class PerceptionPipeline:
    """Orchestrates the full perception pipeline:
    ingest -> detect+track -> pose -> serialize to JSON.

    Stage 5 adds JSON serialization: each processed frame is written to a
    JSON file matching the frozen output contract.
    """

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

        # Ensure output dirs exist
        ensure_dir(config.OUTPUT_DIR)

        # ---- Stage 3: Tracked detector ----
        self.tracked_detector = TrackedDetector(
            model_name=config.MODEL_DETECT_NAME,
            tracker_config=config.TRACKER_CONFIG,
            conf_threshold=0.35,
        )

        # ---- Stage 4: Pose estimator ----
        self.pose_estimator = PoseEstimator(
            model_name=config.MODEL_POSE_NAME,
        )

        # ---- Stage 5: JSON serializer ----
        self.serializer = FrameSerializer(
            output_dir=config.OUTPUT_DIR,
        )

    def run(self, max_frames: int = None):
        """Run the pipeline over the video source.

        Args:
            max_frames: Stop after processing this many sampled frames (None = all).
        """
        ingestor = VideoIngestor(self.video_source, self.sample_fps)
        info = ingestor.info()
        print("=" * 60)
        print("MI-EYE PRO -- Core Perception Pipeline (Stage 5: Full)")
        print("=" * 60)
        for k, v in info.items():
            print(f"  {k:>20s}: {v}")
        print(f"  {'detection_model':>20s}: {config.MODEL_DETECT_NAME}")
        print(f"  {'pose_model':>20s}: {config.MODEL_POSE_NAME}")
        print(f"  {'tracker':>20s}: {config.TRACKER_CONFIG}")
        print(f"  {'output_dir':>20s}: {config.OUTPUT_DIR}")
        print("=" * 60)

        processed = 0
        total_detections = 0
        total_keypoints_detected = 0
        all_track_ids = set()
        t_start = time.time()

        for frame_id, timestamp, frame in ingestor.frames():
            # ---- Stage 3: Detection + Tracking ----
            tracked_dets = self.tracked_detector.track(frame, persist=True)

            # ---- Stage 4: Pose estimation per tracked person ----
            tracks = []
            for det in tracked_dets:
                keypoints = self.pose_estimator.estimate(frame, det["bbox"])
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

            # Build the frame result
            result = FrameResult(
                frame_id=frame_id,
                frame_ts=timestamp,
                session_id=self.session_id,
                camera_id=self.camera_id,
                tracks=tracks,
            )

            # ---- Stage 5: Serialize to JSON ----
            json_path = self.serializer.write(result)

            total_detections += len(tracks)

            # Print summary
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


# Need os for basename in the print statement
import os
