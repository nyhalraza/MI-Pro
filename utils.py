"""
utils.py — Shared utility functions for the MI-EYE PRO perception pipeline.

Kept deliberately small: only things that two or more modules need.
"""

import os
import time


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Returns the path for chaining."""
    os.makedirs(path, exist_ok=True)
    return path


def current_epoch() -> float:
    """Return current time as a Unix epoch float (seconds).

    WHY time.time()?
        The contract specifies frame_ts as 'unix epoch seconds'.  For recorded
        video we'll compute this from the video's own timestamps, but for live
        streams time.time() is the fallback.
    """
    return time.time()
