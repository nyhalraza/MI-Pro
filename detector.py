"""
detector.py -- YOLO-based person detector for the MI-EYE PRO pipeline.

MODEL CHOICE RATIONALE (for FYP committee):
    We evaluated the YOLOv8 model family (n/s/m/l/x) against our hardware constraint:
    a Google Colab T4 GPU with 16 GB VRAM.

    +-------+--------+----------+----------+------------------------------+
    | Model | Params | VRAM Est | mAP50-95 | Notes                        |
    +-------+--------+----------+----------+------------------------------+
    | v8n   |  3.2M  | ~1.5 GB  |   37.3   | Fast, fits easily on T4      |
    | v8s   | 11.2M  |  ~3 GB   |   44.9   | Good accuracy/speed balance  |
    | v8m   | 25.9M  |  ~6 GB   |   50.2   | Still fits T4, slower        |
    | v8l   | 43.7M  | ~10 GB   |   52.9   | Tight on T4 with pose too    |
    | v8x   | 68.2M  | ~14 GB   |   53.9   | Risky: leaves <2GB for pose  |
    +-------+--------+----------+----------+------------------------------+

    DECISION: YOLOv8s (small)
    WHY NOT NANO?
        - Nano (v8n) is the fastest but its mAP is 7.6 points below small.
          In a classroom with partial occlusion (students behind desks, teacher
          at whiteboard edge), that accuracy gap means missed detections that
          would cascade into missed attendance records.
    WHY NOT MEDIUM/LARGE?
        - We need to run detection + pose on the SAME GPU.  YOLOv8s uses ~3 GB,
          leaving ~13 GB for pose estimation, face crops, and PyTorch overhead.
        - Medium would also work, but small gives us 2x the throughput for a
          modest accuracy tradeoff -- and at 2 FPS sampling we're not bottlenecked.
    GPU-CONSTRAINT FLAG:
        - If running on a 4 GB card, drop to v8n (nano).

    WHY YOLOv8 OVER YOLOv11?
        - YOLOv11 (YOLO11) is newer but has fewer community benchmarks as of our
          development date.  YOLOv8 is battle-tested, widely documented, and the
          Ultralytics API is identical -- we can swap to v11 with a one-line
          config change if benchmarks justify it.

DETECTION STRATEGY:
    - We detect ALL 80 COCO classes but filter to class 0 ("person") only.
    - Confidence threshold: 0.35 (lower than default 0.5 to catch partially
      occluded students; false positives are cheaper than false negatives for
      attendance tracking).
    - We run detection on the FULL frame, not on crops -- YOLO is designed for
      single-pass whole-image detection.
"""

import numpy as np
from ultralytics import YOLO

import config


class PersonDetector:
    """Runs YOLOv8 person detection on frames.

    Attributes:
        model: The loaded YOLO model instance.
        conf_threshold: Minimum confidence to keep a detection.
        device: 'cuda' or 'cpu' (auto-detected by Ultralytics).
    """

    # COCO class index for "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = config.MODEL_DETECT_NAME,
        conf_threshold: float = 0.35,
    ):
        """Load the YOLO detection model.

        Args:
            model_name: Ultralytics model identifier (e.g. 'yolov8s.pt').
                        Auto-downloads on first use.
            conf_threshold: Detections below this confidence are discarded.
                           0.35 is intentionally lower than default (0.5) to
                           catch partially occluded students behind desks.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold

        # Ultralytics auto-selects GPU if available; falls back to CPU
        print(f"[Detector] Loading {model_name}...")
        self.model = YOLO(model_name)
        print(f"[Detector] Model loaded. Device will be auto-selected at inference.")

    def detect(self, frame: np.ndarray) -> list:
        """Run person detection on a single frame.

        Args:
            frame: H x W x 3 BGR numpy array (OpenCV format).

        Returns:
            List of dicts, each with:
                - 'bbox': [x1, y1, x2, y2] as floats (pixel coords)
                - 'confidence': float
                - 'class_id': int (always 0 for person)

            Empty list if no persons detected.
        """
        # Run inference
        # verbose=False suppresses per-frame logging from Ultralytics
        # classes=[0] filters to person-only at the model level (faster than
        # post-filtering because YOLO skips NMS on other classes)
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]  # single image -> single result

            if result.boxes is not None and len(result.boxes) > 0:
                # Extract boxes as numpy arrays
                boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4)
                confs = result.boxes.conf.cpu().numpy()    # (N,)
                class_ids = result.boxes.cls.cpu().numpy() # (N,)

                for i in range(len(boxes)):
                    detections.append({
                        "bbox": boxes[i].tolist(),       # [x1, y1, x2, y2]
                        "confidence": float(confs[i]),
                        "class_id": int(class_ids[i]),
                    })

        return detections
