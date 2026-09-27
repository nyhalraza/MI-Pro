"""
pose_estimator.py -- Full-frame pose estimation for the MI-EYE PRO pipeline.

ARCHITECTURE CHANGE (v2):
    v1 (Stage 4, old): Crop each tracked person -> run pose model N times
    v2 (current):      Run pose model ONCE on full frame -> match to tracks via IoU

    WHY THIS IS FASTER:
        With 7 people in a classroom, v1 made 7 separate YOLO-pose inference calls.
        Each call has GPU overhead (memory allocation, kernel launch, data transfer).
        v2 makes ONE call that detects all 7 poses simultaneously.

        Expected speedup: 3-5x on the same hardware, because:
        - One inference call instead of N (eliminates N-1 kernel launch overheads)
        - YOLO-pose is designed for multi-person full-frame inference
        - GPU utilization is much higher with one large batch vs N tiny crops

    HOW MATCHING WORKS:
        After running pose on the full frame, we get M pose detections (may differ
        from the N tracked persons).  We match them using IoU (Intersection over
        Union) between:
        - The tracked person's bbox (from detection + tracking)
        - The pose detection's bbox (from the pose model)

        The Hungarian algorithm finds the optimal 1-to-1 assignment.
        Unmatched tracks get zero-confidence placeholder keypoints.

KEYPOINT MAPPING:
    Same as v1: COCO-17 -> contract-13 (drop ears + ankles).
    See schemas.COCO_TO_CONTRACT_INDICES for the mapping.
"""

import numpy as np
from ultralytics import YOLO

import config
from schemas import Keypoint, KEYPOINT_NAMES, COCO_TO_CONTRACT_INDICES


def _compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _compute_iou_matrix(boxes_a, boxes_b):
    """Compute NxM IoU matrix between two sets of boxes."""
    n = len(boxes_a)
    m = len(boxes_b)
    iou_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = _compute_iou(boxes_a[i], boxes_b[j])
    return iou_matrix


def _greedy_match(iou_matrix, threshold=0.3):
    """Greedy IoU matching: assign each track to the best-matching pose detection.

    WHY GREEDY INSTEAD OF HUNGARIAN?
        - For typical classroom sizes (5-15 people), greedy matching is sufficient
          and avoids the scipy dependency.
        - Hungarian would be better for dense crowds where many boxes overlap,
          but in a classroom setting people are mostly spatially separated.

    Args:
        iou_matrix: NxM matrix where N=tracks, M=pose detections
        threshold: Minimum IoU to consider a match valid

    Returns:
        dict mapping track_index -> pose_detection_index
    """
    n_tracks, n_poses = iou_matrix.shape
    matches = {}
    used_poses = set()

    # Sort all (track, pose) pairs by IoU descending
    pairs = []
    for i in range(n_tracks):
        for j in range(n_poses):
            if iou_matrix[i, j] >= threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for _, track_idx, pose_idx in pairs:
        if track_idx not in matches and pose_idx not in used_poses:
            matches[track_idx] = pose_idx
            used_poses.add(pose_idx)

    return matches


class PoseEstimator:
    """Runs YOLO-pose on the FULL FRAME, then matches poses to tracked bboxes.

    Usage:
        estimator = PoseEstimator()
        all_keypoints = estimator.estimate_batch(frame, tracked_bboxes)
        # Returns list of 13-keypoint lists, one per tracked bbox
    """

    def __init__(self, model_name: str = config.MODEL_POSE_NAME):
        print(f"[Pose] Loading {model_name}...")
        self.model = YOLO(model_name)
        print(f"[Pose] Model loaded (full-frame mode).")

    def estimate_batch(self, frame: np.ndarray, tracked_bboxes: list) -> list:
        """Run pose estimation on the full frame and match to tracked bounding boxes.

        Args:
            frame: Full H x W x 3 BGR frame.
            tracked_bboxes: List of [x1, y1, x2, y2] bboxes from the tracker.

        Returns:
            List of keypoint lists (one per tracked bbox, same order).
            Each is 13 Keypoint objects in contract order.
            Unmatched tracks get zero-confidence placeholders.
        """
        n_tracks = len(tracked_bboxes)

        if n_tracks == 0:
            return []

        # Run pose model ONCE on the full frame
        results = self.model.predict(
            source=frame,
            conf=0.25,
            verbose=False,
        )

        # If no pose detections, return placeholders for all tracks
        if not results or len(results) == 0:
            return [self._empty_keypoints() for _ in range(n_tracks)]

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0 or result.boxes is None:
            return [self._empty_keypoints() for _ in range(n_tracks)]

        # Extract pose detection bboxes and keypoints
        pose_boxes = result.boxes.xyxy.cpu().numpy()   # (M, 4)
        kp_data = result.keypoints.data.cpu().numpy()   # (M, 17, 3)

        # Compute IoU between tracked bboxes and pose bboxes
        iou_matrix = _compute_iou_matrix(tracked_bboxes, pose_boxes)

        # Match tracks to pose detections
        matches = _greedy_match(iou_matrix, threshold=0.3)

        # Build keypoints list in track order
        all_keypoints = []
        for track_idx in range(n_tracks):
            if track_idx in matches:
                pose_idx = matches[track_idx]
                kp17 = kp_data[pose_idx]  # (17, 3) in full-frame coordinates
                keypoints = self._map_keypoints(kp17)
            else:
                # No pose matched this track — zero-confidence placeholders
                keypoints = self._empty_keypoints()
            all_keypoints.append(keypoints)

        return all_keypoints

    def _map_keypoints(self, kp17: np.ndarray) -> list:
        """Map COCO-17 keypoints to contract-13 keypoints.

        Args:
            kp17: (17, 3) array of (x, y, conf) in full-frame coordinates.

        Returns:
            List of 13 Keypoint objects in contract order.
        """
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
