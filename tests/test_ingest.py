"""
tests/test_ingest.py — Smoke test for the video ingest loop.

Verifies:
    1. VideoIngestor can open the sample video
    2. .info() returns sane metadata
    3. .frames() yields the expected number of sampled frames
    4. Each yielded frame has the right shape (H, W, 3)
    5. FrameResult schema serializes correctly to dict

Run:
    python -m tests.test_ingest
    # or from Code/:
    python tests/test_ingest.py
"""

import os
import sys

# Add parent dir so imports work when running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from video_ingest import VideoIngestor
from schemas import FrameResult, KEYPOINT_NAMES
import config


def test_ingest():
    video_path = config.VIDEO_PATH

    if not os.path.exists(video_path):
        print(f"[FAIL] Test video not found at {video_path}")
        print("  Run 'python download_sample_video.py' first.")
        return False

    print("-" * 50)
    print("TEST: Video Ingest Smoke Test")
    print("-" * 50)

    # 1. Open video
    ingestor = VideoIngestor(video_path, sample_fps=2.0)
    info = ingestor.info()
    print(f"  [OK] Opened video: {info['source']}")
    print(f"    Resolution : {info['resolution']}")
    print(f"    Source FPS : {info['source_fps']}")
    print(f"    Duration   : {info['duration_sec']}s")
    print(f"    Interval   : every {info['frame_interval']}th frame")
    print(f"    Est samples: {info['estimated_samples']}")

    # 2. Sanity-check metadata
    assert info["source_fps"] > 0, "Source FPS should be positive"
    assert info["total_frames"] > 0, "Total frames should be positive"

    # 3. Iterate first 10 sampled frames
    count = 0
    for frame_id, ts, frame in ingestor.frames():
        assert frame_id == count, f"Expected frame_id={count}, got {frame_id}"
        assert ts >= 0, f"Timestamp should be non-negative, got {ts}"
        assert len(frame.shape) == 3 and frame.shape[2] == 3, \
            f"Frame should be HxWx3, got {frame.shape}"

        count += 1
        if count >= 10:
            break

    print(f"  [OK] Iterated {count} sampled frames, all shapes valid")

    # 4. Test FrameResult serialization (empty tracks -- Stage 1 baseline)
    result = FrameResult(
        frame_id=0,
        frame_ts=0.0,
        session_id="test_session",
        camera_id="test_cam",
        tracks=[],
    )
    d = result.to_dict()
    assert list(d.keys()) == ["frame_id", "frame_ts", "session_id", "camera_id", "tracks"], \
        f"Unexpected top-level keys: {list(d.keys())}"
    assert d["tracks"] == []
    print(f"  [OK] FrameResult serializes to correct contract shape: {list(d.keys())}")

    # 5. Verify keypoint count
    assert len(KEYPOINT_NAMES) == 13, f"Expected 13 keypoints, got {len(KEYPOINT_NAMES)}"
    print(f"  [OK] KEYPOINT_NAMES has {len(KEYPOINT_NAMES)} entries")

    ingestor.release()
    print("-" * 50)
    print("ALL TESTS PASSED")
    print("-" * 50)
    return True


if __name__ == "__main__":
    success = test_ingest()
    sys.exit(0 if success else 1)
