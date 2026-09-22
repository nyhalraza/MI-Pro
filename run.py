"""
run.py — Entry point for the MI-EYE PRO perception pipeline.

Usage:
    python run.py                           # Process full video at defaults
    python run.py --video path/to/clip.mp4  # Custom video
    python run.py --max-frames 20           # Quick smoke test (20 sampled frames)
    python run.py --fps 1.0                 # Override sample rate
"""

import argparse
import config
from pipeline import PerceptionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="MI-EYE PRO — Core Perception & Streaming Engine"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=config.VIDEO_PATH,
        help="Path to video file or RTSP URI",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=config.SAMPLE_FPS,
        help="Frames per second to sample from the source video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N sampled frames (for quick testing)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=config.SESSION_ID,
        help="Session identifier embedded in JSON output",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=config.CAMERA_ID,
        help="Camera identifier embedded in JSON output",
    )

    args = parser.parse_args()

    pipeline = PerceptionPipeline(
        video_source=args.video,
        sample_fps=args.fps,
        session_id=args.session_id,
        camera_id=args.camera_id,
    )

    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
