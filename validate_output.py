"""
validate_output.py -- Contract validation script for MI-EYE PRO JSON output.

Run this BEFORE handing output to teammates to guarantee the JSON matches
the frozen contract.  Catches:
    - Missing or extra top-level keys
    - Wrong types (e.g. frame_id as string instead of int)
    - Missing or extra keypoints (must be exactly 13)
    - Wrong keypoint names or order
    - Missing required Track fields
    - null where not allowed, or non-null where null expected

Usage:
    python validate_output.py                           # validate all files in output/
    python validate_output.py --dir path/to/output      # custom directory
    python validate_output.py --file path/to/frame.json # single file
"""

import json
import os
import sys
import argparse
from typing import List, Tuple

# The 13 contract keypoint names IN ORDER
EXPECTED_KEYPOINTS = [
    "nose", "left_eye", "right_eye",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]

# Top-level keys in exact order
EXPECTED_TOP_KEYS = ["frame_id", "frame_ts", "session_id", "camera_id", "tracks"]

# Track keys in exact order
EXPECTED_TRACK_KEYS = ["track_id", "class", "bbox", "confidence", "keypoints", "face_crop_path"]

# Keypoint keys in exact order
EXPECTED_KP_KEYS = ["name", "x", "y", "conf"]


def validate_frame(data: dict) -> List[str]:
    """Validate a single frame's JSON against the contract.

    Returns a list of error strings.  Empty list = valid.
    """
    errors = []

    # ---- Top-level structure ----
    actual_keys = list(data.keys())
    if actual_keys != EXPECTED_TOP_KEYS:
        errors.append(f"Top-level keys mismatch. Expected {EXPECTED_TOP_KEYS}, got {actual_keys}")

    # Type checks
    if not isinstance(data.get("frame_id"), int):
        errors.append(f"frame_id should be int, got {type(data.get('frame_id')).__name__}")

    if not isinstance(data.get("frame_ts"), (int, float)):
        errors.append(f"frame_ts should be float, got {type(data.get('frame_ts')).__name__}")

    if not isinstance(data.get("session_id"), str):
        errors.append(f"session_id should be string, got {type(data.get('session_id')).__name__}")

    if not isinstance(data.get("camera_id"), str):
        errors.append(f"camera_id should be string, got {type(data.get('camera_id')).__name__}")

    if not isinstance(data.get("tracks"), list):
        errors.append(f"tracks should be array, got {type(data.get('tracks')).__name__}")
        return errors  # can't validate tracks if it's not a list

    # ---- Validate each track ----
    for t_idx, track in enumerate(data["tracks"]):
        prefix = f"tracks[{t_idx}]"

        # Key order
        track_keys = list(track.keys())
        if track_keys != EXPECTED_TRACK_KEYS:
            errors.append(f"{prefix}: keys mismatch. Expected {EXPECTED_TRACK_KEYS}, got {track_keys}")

        # Type checks
        if not isinstance(track.get("track_id"), int):
            errors.append(f"{prefix}.track_id should be int, got {type(track.get('track_id')).__name__}")

        if track.get("class") != "person":
            errors.append(f'{prefix}.class should be "person", got {track.get("class")!r}')

        bbox = track.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{prefix}.bbox should be [x1,y1,x2,y2] (4 floats), got {bbox}")
        elif not all(isinstance(v, (int, float)) for v in bbox):
            errors.append(f"{prefix}.bbox values should all be numbers")

        if not isinstance(track.get("confidence"), (int, float)):
            errors.append(f"{prefix}.confidence should be float, got {type(track.get('confidence')).__name__}")

        # face_crop_path: string or null
        fcp = track.get("face_crop_path")
        if fcp is not None and not isinstance(fcp, str):
            errors.append(f"{prefix}.face_crop_path should be string or null, got {type(fcp).__name__}")

        # ---- Validate keypoints ----
        kps = track.get("keypoints")
        if not isinstance(kps, list):
            errors.append(f"{prefix}.keypoints should be array, got {type(kps).__name__}")
            continue

        if len(kps) != 13:
            errors.append(f"{prefix}.keypoints should have exactly 13 entries, got {len(kps)}")
            continue

        for kp_idx, kp in enumerate(kps):
            kp_prefix = f"{prefix}.keypoints[{kp_idx}]"

            # Key order
            kp_keys = list(kp.keys())
            if kp_keys != EXPECTED_KP_KEYS:
                errors.append(f"{kp_prefix}: keys mismatch. Expected {EXPECTED_KP_KEYS}, got {kp_keys}")

            # Name must match contract order
            expected_name = EXPECTED_KEYPOINTS[kp_idx]
            if kp.get("name") != expected_name:
                errors.append(f"{kp_prefix}.name should be '{expected_name}', got '{kp.get('name')}'")

            # Type checks
            if not isinstance(kp.get("x"), (int, float)):
                errors.append(f"{kp_prefix}.x should be float, got {type(kp.get('x')).__name__}")
            if not isinstance(kp.get("y"), (int, float)):
                errors.append(f"{kp_prefix}.y should be float, got {type(kp.get('y')).__name__}")
            if not isinstance(kp.get("conf"), (int, float)):
                errors.append(f"{kp_prefix}.conf should be float, got {type(kp.get('conf')).__name__}")

    return errors


def validate_file(filepath: str) -> Tuple[bool, List[str]]:
    """Validate a single JSON file. Returns (is_valid, errors)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    errors = validate_frame(data)
    return len(errors) == 0, errors


def validate_directory(dirpath: str) -> Tuple[int, int, List[Tuple[str, List[str]]]]:
    """Validate all .json files in a directory.

    Returns:
        (total_files, valid_files, list of (filename, errors) for invalid files)
    """
    json_files = sorted([f for f in os.listdir(dirpath) if f.endswith(".json")])

    if not json_files:
        print(f"[WARNING] No .json files found in {dirpath}")
        return 0, 0, []

    total = len(json_files)
    valid = 0
    invalid_files = []

    for fname in json_files:
        fpath = os.path.join(dirpath, fname)
        is_valid, errors = validate_file(fpath)
        if is_valid:
            valid += 1
        else:
            invalid_files.append((fname, errors))

    return total, valid, invalid_files


def main():
    parser = argparse.ArgumentParser(description="Validate MI-EYE PRO JSON output against the contract")
    parser.add_argument("--dir", type=str, default=None, help="Directory of JSON files to validate")
    parser.add_argument("--file", type=str, default=None, help="Single JSON file to validate")
    args = parser.parse_args()

    if args.file:
        print(f"Validating: {args.file}")
        is_valid, errors = validate_file(args.file)
        if is_valid:
            print(f"  [PASS] Contract-compliant")
        else:
            print(f"  [FAIL] {len(errors)} error(s):")
            for e in errors:
                print(f"    - {e}")
        sys.exit(0 if is_valid else 1)

    # Default: validate output directory
    output_dir = args.dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    if not os.path.isdir(output_dir):
        print(f"[ERROR] Directory not found: {output_dir}")
        sys.exit(1)

    print(f"Validating all JSON files in: {output_dir}")
    print("-" * 50)

    total, valid, invalid_files = validate_directory(output_dir)

    print(f"\nResults: {valid}/{total} files passed validation")

    if invalid_files:
        print(f"\n[FAIL] {len(invalid_files)} file(s) have errors:")
        for fname, errors in invalid_files:
            print(f"\n  {fname}:")
            for e in errors:
                print(f"    - {e}")
        sys.exit(1)
    else:
        if total > 0:
            print("[PASS] All files are contract-compliant")
        sys.exit(0)


if __name__ == "__main__":
    main()
