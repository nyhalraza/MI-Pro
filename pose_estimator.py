"""
pose_estimator.py -- Per-person pose estimation for the MI-EYE PRO pipeline.

APPROACH:
    We use a TWO-MODEL strategy (not a single pose model on the full frame):
        1. YOLOv8s (detection + tracking) runs on the full frame    -> bounding boxes
        2. YOLOv8s-pose runs on CROPPED regions per tracked person  -> keypoints

    WHY NOT a single YOLO-pose model for everything?
        - A single yolov8s-pose.pt CAN do detection + pose in one pass, but
          it doesn't integrate with Ultralytics' built-in tracking (model.track()
          only works with detection models, not pose models, for track ID assignment).
        - We'd have to write custom tracker integration, which is fragile and
          defeats the purpose of using Ultralytics' battle-tested tracker.
        - The two-model approach cleanly separates concerns:
            * TrackedDetector owns detection + tracking (stable IDs)
            * PoseEstimator owns keypoint extraction (accurate poses)

    ALTERNATIVE CONSIDERED: Run yolov8s-pose on full frame, then manually match
    pose detections to tracked bboxes via IoU.  Rejected because:
        - Matching is error-prone when people overlap
        - Running pose on the full frame wastes compute on background regions
        - Crop-based inference focuses the model's attention on each person

    MODEL CHOICE: yolov8s-pose (small)
        - Matches our detection model size (yolov8s) for consistency
        - ~3 GB VRAM, fitting within our T4 budget alongside the detector
        - 13 of the 17 COCO keypoints are used (contract drops ears + ankles)

    GPU-CONSTRAINT FLAG:
        Combined VRAM: ~3 GB (detect) + ~3 GB (pose) = ~6 GB on T4 (16 GB).
        This leaves comfortable headroom.  On a 4 GB card, switch both to nano.

KEYPOINT MAPPING:
    YOLO-pose outputs 17 COCO keypoints.  Our contract requires 13 (no ears/ankles).
    The mapping is defined in schemas.COCO_TO_CONTRACT_INDICES.

    COCO-17 order:           Our contract-13 order:
    0  nose                  0  nose           (COCO 0)
    1  left_eye              1  left_eye       (COCO 1)
    2  right_eye             2  right_eye      (COCO 2)
    3  left_ear       [DROP] 3  left_shoulder  (COCO 5)
    4  right_ear      [DROP] 4  right_shoulder (COCO 6)
    5  left_shoulder         5  left_elbow     (COCO 7)
    6  right_shoulder        6  right_elbow    (COCO 8)
    7  left_elbow            7  left_wrist     (COCO 9)
    8  right_elbow           8  right_wrist    (COCO 10)
    9  left_wrist            9  left_hip       (COCO 11)
    10 right_wrist           10 right_hip      (COCO 12)
    11 left_hip              11 left_knee      (COCO 13)
    12 right_hip             12 right_knee     (COCO 14)
    13 left_knee
    14 right_knee
    15 left_ankle     [DROP]
    16 right_ankle    [DROP]
"""

import cv2
import numpy as np
from ultralytics import YOLO

import config
from schemas import Keypoint, KEYPOINT_NAMES, COCO_TO_CONTRACT_INDICES


class PoseEstimator:
    """Runs YOLO-pose on cropped person regions to extract keypoints.

    Usage:
        estimator = PoseEstimator()
        keypoints = estimator.estimate(frame, bbox=[x1, y1, x2, y2])
        # Returns list of 13 Keypoint objects matching the contract
    """

    def __init__(self, model_name: str = config.MODEL_POSE_NAME):
        print(f"[Pose] Loading {model_name}...")
        self.model = YOLO(model_name)
        print(f"[Pose] Model loaded.")

    def estimate(self, frame: np.ndarray, bbox: list) -> list:
        """Run pose estimation on a cropped region of the frame.

        Args:
            frame: Full H x W x 3 BGR frame.
            bbox: [x1, y1, x2, y2] bounding box of the person (pixel coords).

        Returns:
            List of 13 Keypoint objects in contract order.
            If pose estimation fails (e.g. crop too small), returns 13 keypoints
            with conf=0.0 (the contract requires all 13 always present).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp bbox to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Skip tiny crops -- pose model can't extract meaningful keypoints
        # from a region smaller than ~30x30 pixels
        if crop_w < 30 or crop_h < 30:
            return self._empty_keypoints()

        # Crop the person region
        # PADDING: Add 10% padding around the bbox to give the pose model
        # context (shoulders/hips near edges get cut off without padding)
        pad_x = int(crop_w * 0.1)
        pad_y = int(crop_h * 0.1)
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)

        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        # Run pose estimation on the crop
        results = self.model.predict(
            source=crop,
            conf=0.25,      # lower threshold for pose -- we want all visible keypoints
            verbose=False,
        )

        if not results or len(results) == 0:
            return self._empty_keypoints()

        result = results[0]

        # The pose model may detect multiple people in the crop (unlikely with
        # a tight bbox, but possible).  Take the detection with highest confidence.
        if result.keypoints is None or len(result.keypoints) == 0:
            return self._empty_keypoints()

        # Get the best detection's keypoints
        # result.keypoints.data shape: (num_detections, 17, 3) where 3 = (x, y, conf)
        kp_data = result.keypoints.data.cpu().numpy()

        if len(kp_data) == 0:
            return self._empty_keypoints()

        # Pick the detection with highest average keypoint confidence
        if len(kp_data) > 1:
            avg_confs = kp_data[:, :, 2].mean(axis=1)
            best_idx = int(np.argmax(avg_confs))
        else:
            best_idx = 0

        kp17 = kp_data[best_idx]  # shape: (17, 3) -> (x, y, conf) in CROP coordinates

        # Convert crop-local coordinates back to full-frame coordinates
        # The keypoints are relative to the padded crop, so we add the crop offset
        kp17[:, 0] += x1_pad  # x offset
        kp17[:, 1] += y1_pad  # y offset

        # Map COCO-17 to our contract-13
        keypoints = []
        for contract_idx, coco_idx in enumerate(COCO_TO_CONTRACT_INDICES):
            kp = kp17[coco_idx]
            keypoints.append(Keypoint(
                name=KEYPOINT_NAMES[contract_idx],
                x=round(float(kp[0]), 2),
                y=round(float(kp[1]), 2),
                conf=round(float(kp[2]), 4),
            ))

        return keypoints

    def _empty_keypoints(self) -> list:
        """Return 13 zero-confidence keypoints (contract-compliant placeholder)."""
        return [
            Keypoint(name=name, x=0.0, y=0.0, conf=0.0)
            for name in KEYPOINT_NAMES
        ]
