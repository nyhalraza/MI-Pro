"""
video_ingest.py — Frame-by-frame video reader with configurable sampling rate.

DESIGN DECISIONS:
    • OpenCV (cv2.VideoCapture) over ffmpeg-python / decord:
        - Ships with pip, zero system-library hassle on Colab
        - Supports .mp4, .avi, RTSP, GStreamer URIs — good enough for our FYP scope
        - If we later need hardware-accelerated decode we can swap to GStreamer backend
          via cv2.VideoCapture(uri, cv2.CAP_GSTREAMER) — same API, no code changes.

    • Frame sampling (yield every Nth frame) instead of processing every frame:
        - A classroom camera at 30 FPS produces far more data than our GPU can handle
          in real-time with detection + tracking + pose on every frame.
        - Sampling at 2 FPS gives ~2 detections/second — plenty for attendance/behavior
          analysis, which don't need sub-second granularity.
        - GPU-CONSTRAINT FLAG: On a T4 (16 GB VRAM) 2 FPS is conservative; we benchmark
          and raise this in Stage 6.

    • Generator pattern (yield, not return-a-list):
        - Keeps memory constant regardless of video length.
        - Downstream pipeline pulls frames lazily.
"""

import cv2
import time
from typing import Generator, Tuple

import numpy as np


class VideoIngestor:
    """Opens a video source and yields sampled frames as (frame_id, timestamp, ndarray).

    Args:
        source: Path to .mp4 file, or an RTSP/GStreamer URI for live streams.
        sample_fps: How many frames per second to actually yield. Must be > 0.
                    If the source FPS is lower, every frame is yielded instead.
    """

    def __init__(self, source: str, sample_fps: float = 2.0):
        self.source = source
        self.sample_fps = sample_fps

        # Open the video
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        # Read source metadata
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback if metadata missing
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute how many source frames to skip between samples
        # e.g. source=30fps, sample=2fps → keep every 15th frame
        self._frame_interval = max(1, int(round(self.source_fps / self.sample_fps)))

    def info(self) -> dict:
        """Return a summary dict for logging / smoke tests."""
        duration_sec = self.total_frames / self.source_fps if self.source_fps > 0 else 0
        return {
            "source": self.source,
            "source_fps": self.source_fps,
            "total_frames": self.total_frames,
            "resolution": f"{self.width}x{self.height}",
            "duration_sec": round(duration_sec, 2),
            "sample_fps": self.sample_fps,
            "frame_interval": self._frame_interval,
            "estimated_samples": int(self.total_frames / self._frame_interval),
        }

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_id, timestamp, frame) tuples at the configured sample rate.

        frame_id: sequential counter of *sampled* frames (0, 1, 2, ...)
        timestamp: seconds since video start (computed from source frame index / fps)
        frame:     H×W×3 BGR numpy array (OpenCV native format)
        """
        raw_index = 0        # tracks the source video's frame counter
        sample_id = 0        # our sequential output frame counter

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # end of video (or stream dropped)

            # Only yield frames at the sampling interval
            if raw_index % self._frame_interval == 0:
                # Timestamp relative to video start (for recorded files)
                # For live streams, replace with time.time()
                timestamp = raw_index / self.source_fps
                yield sample_id, timestamp, frame
                sample_id += 1

            raw_index += 1

    def release(self):
        """Release the OpenCV capture. Call when done."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()
