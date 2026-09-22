"""
serializer.py -- JSON output writer for the MI-EYE PRO perception pipeline.

Writes one JSON file per processed frame to the output directory.
Each file matches the frozen contract exactly.

OUTPUT FORMAT:
    output/
      frame_000000.json
      frame_000001.json
      ...

    Each file contains the FrameResult serialized via .to_dict().

WHY ONE FILE PER FRAME (not a single giant JSON array)?
    1. Downstream modules can process frames independently / in parallel
    2. If the pipeline crashes mid-video, all previously written frames are safe
    3. Easier to inspect/debug individual frames
    4. The validation script can check one file at a time
    5. For production, these could be replaced with a message queue (Redis, Kafka)
       but files are simpler and sufficient for our FYP demo
"""

import json
import os

from schemas import FrameResult
from utils import ensure_dir
import config


class FrameSerializer:
    """Writes FrameResult objects to JSON files matching the output contract."""

    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        self.output_dir = ensure_dir(output_dir)
        self._frames_written = 0

    def write(self, result: FrameResult) -> str:
        """Serialize a FrameResult to a JSON file.

        Args:
            result: The FrameResult to serialize.

        Returns:
            The path to the written JSON file.
        """
        filename = f"frame_{result.frame_id:06d}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        self._frames_written += 1
        return filepath

    @property
    def frames_written(self) -> int:
        return self._frames_written
