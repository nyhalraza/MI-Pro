"""
tracker.py -- Multi-object tracking wrapper for the MI-EYE PRO pipeline.

TRACKER CHOICE RATIONALE (for FYP committee):

    ByteTrack vs BoT-SORT:
    -----------------------------------------------------------------------
    ByteTrack (chosen):
        - Uses only Kalman filter (motion-based) -- no appearance re-ID model
        - Zero additional VRAM cost, since it runs on CPU
        - Key innovation: associates BOTH high-confidence AND low-confidence
          detections in a two-stage matching process, which recovers occluded
          persons that single-threshold trackers would lose
        - Ideal for FIXED-CAMERA scenarios like classrooms where people don't
          cross paths frequently

    BoT-SORT (alternative):
        - Adds a lightweight appearance re-identification (re-ID) model on top
          of Kalman filter motion
        - Better at maintaining IDs through heavy occlusion and camera motion
        - Costs ~200-500 MB extra VRAM for the re-ID feature extractor
        - Overkill for our fixed-camera classroom setup

    GPU-CONSTRAINT FLAG:
        We chose ByteTrack partly because it leaves more VRAM for pose estimation
        (Stage 4).  On a T4 with 16 GB, this gives us:
            ~3 GB (YOLOv8s detect) + 0 GB (ByteTrack) + ~2 GB (pose) = ~5 GB
            leaving ~11 GB headroom for batch processing and PyTorch overhead.

    WHY ULTRALYTICS BUILT-IN TRACKER?
        Ultralytics ships ByteTrack and BoT-SORT as built-in options via
        model.track().  This means:
        - No extra dependencies to install
        - Tracker runs in the same inference call as detection (fewer data copies)
        - Switching to BoT-SORT is a one-line config change (bytetrack.yaml -> botsort.yaml)
        - The tracker is maintained by the Ultralytics team alongside the detector

INTEGRATION APPROACH:
    Instead of running model.predict() then a separate tracker, we use
    model.track() which combines detection + tracking in one call.
    This is both simpler and faster than feeding detections into a
    standalone tracker library.
"""

import numpy as np
from ultralytics import YOLO

import config


class TrackedDetector:
    """Combined person detector + ByteTrack tracker using Ultralytics model.track().

    This replaces the separate PersonDetector from Stage 2.  By using model.track()
    instead of model.predict(), we get stable track_ids across frames for free.

    The track_id assignment is handled internally by ByteTrack's Kalman filter:
        1. High-confidence detections are matched to existing tracks first (IoU)
        2. Unmatched low-confidence detections are matched in a second pass
        3. New tracks are spawned for unmatched detections
        4. Lost tracks are kept alive for a configurable number of frames
           before being deleted

    Attributes:
        model: The loaded YOLO model instance (same as Stage 2).
        tracker_config: Path to the tracker YAML config (bytetrack.yaml).
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = config.MODEL_DETECT_NAME,
        tracker_config: str = config.TRACKER_CONFIG,
        conf_threshold: float = 0.35,
    ):
        self.model_name = model_name
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold

        print(f"[Tracker] Loading {model_name} with {tracker_config}...")
        self.model = YOLO(model_name)
        print(f"[Tracker] Model loaded. Tracker: {tracker_config}")

    def track(self, frame: np.ndarray, persist: bool = True) -> list:
        """Run detection + tracking on a single frame.

        Args:
            frame: H x W x 3 BGR numpy array.
            persist: If True, maintain tracks across calls (required for
                     multi-frame tracking).  Set False only for single-frame
                     testing.

        Returns:
            List of dicts, each with:
                - 'track_id': int (stable across frames, assigned by ByteTrack)
                - 'bbox': [x1, y1, x2, y2] as floats
                - 'confidence': float
                - 'class_id': int (always 0)

            Empty list if no persons detected/tracked.
        """
        # model.track() runs detection + tracking in one call
        # persist=True tells ByteTrack to maintain its internal state
        # across consecutive calls (essential for ID consistency)
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            tracker=self.tracker_config,
            persist=persist,
            verbose=False,
        )

        tracked = []
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4)
                confs = result.boxes.conf.cpu().numpy()    # (N,)
                class_ids = result.boxes.cls.cpu().numpy() # (N,)

                # Track IDs: result.boxes.id is None if tracker lost the object
                # in this frame (rare edge case on first frame or heavy occlusion)
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)  # (N,)
                else:
                    # Fallback: no IDs assigned yet (usually only frame 0)
                    track_ids = np.full(len(boxes), -1, dtype=int)

                for i in range(len(boxes)):
                    tracked.append({
                        "track_id": int(track_ids[i]),
                        "bbox": boxes[i].tolist(),
                        "confidence": float(confs[i]),
                        "class_id": int(class_ids[i]),
                    })

        return tracked

    def reset(self):
        """Reset the tracker state.  Call between separate video sessions."""
        # Re-instantiate the model to clear ByteTrack's internal state
        self.model = YOLO(self.model_name)
