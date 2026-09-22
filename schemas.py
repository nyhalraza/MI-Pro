"""
schemas.py — Python dataclasses that mirror the frozen JSON output contract EXACTLY.

CONTRACT OWNER: The JSON structure is shared with 3 downstream teammates (face
recognition, attendance, behavior analysis).  Field names, order, and nullability
are FROZEN.  Do not rename or remove any field without a team-wide review.

WHY DATACLASSES (not plain dicts)?
    1. Typos in field names are caught at construction time, not at JSON parse time
       in a teammate's code two days later.
    2. The `to_dict()` methods guarantee the JSON shape every time.
    3. Easy to unit-test: construct → serialize → validate against JSON schema.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ─── The 13 keypoints our contract requires, in FIXED order ──────────────
# NOTE: COCO/YOLO-pose produces 17 keypoints.  Our contract uses 13 — we
# intentionally DROP left_ear(3), right_ear(4), left_ankle(15), right_ankle(16).
# The mapping from COCO-17 indices to our 13 is handled in Stage 4.

KEYPOINT_NAMES: List[str] = [
    "nose",            # COCO index 0
    "left_eye",        # COCO index 1
    "right_eye",       # COCO index 2
    "left_shoulder",   # COCO index 5
    "right_shoulder",  # COCO index 6
    "left_elbow",      # COCO index 7
    "right_elbow",     # COCO index 8
    "left_wrist",      # COCO index 9
    "right_wrist",     # COCO index 10
    "left_hip",        # COCO index 11
    "right_hip",       # COCO index 12
    "left_knee",       # COCO index 13
    "right_knee",      # COCO index 14
]

# COCO-17 → contract-13 index mapping (used in Stage 4 to extract the right keypoints)
COCO_TO_CONTRACT_INDICES: List[int] = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


@dataclass
class Keypoint:
    """A single named keypoint with pixel coordinates and confidence."""
    name: str
    x: float
    y: float
    conf: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "conf": self.conf,
        }


@dataclass
class Track:
    """One tracked person in a single frame."""
    track_id: int
    cls: str                              # always "person"
    bbox: List[float]                     # [x1, y1, x2, y2] in pixels
    confidence: float
    keypoints: List[Keypoint]             # exactly 13, in contract order
    face_crop_path: Optional[str] = None  # null until face crop is saved

    def to_dict(self) -> dict:
        """Serialize to the frozen JSON contract.

        NOTE: the JSON key is "class" (Python reserved word), so we store as
        `cls` internally but emit "class" here.
        """
        return {
            "track_id": self.track_id,
            "class": self.cls,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "face_crop_path": self.face_crop_path,
        }


@dataclass
class FrameResult:
    """The top-level output for one processed frame — matches the JSON contract."""
    frame_id: int
    frame_ts: float                # unix epoch seconds
    session_id: str
    camera_id: str
    tracks: List[Track] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "frame_ts": self.frame_ts,
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "tracks": [t.to_dict() for t in self.tracks],
        }
