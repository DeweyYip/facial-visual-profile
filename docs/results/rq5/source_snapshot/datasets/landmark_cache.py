"""Read cached MTCNN landmarks without running the detector."""

import hashlib
import json
import sqlite3
from pathlib import Path

import torch


# the Python reader that reads our stored predicted lanmarks in SQLite
"""
it goes to SQLite and does stuff like:
SELECT name, result
FROM detections
WHERE split = 0
"""
class MTCNNLandmarkCache:
    """Load one official split and return original-image coordinates."""

    POINT_ORDER = [
        "lefteye", "righteye", "nose", "leftmouth", "rightmouth"
    ]
    SPLIT_CODES = {"train": "0", "valid": "1", "test": "2"}

    def __init__(self, cache_path, partition_path, split="train"):
        if split not in self.SPLIT_CODES:
            raise ValueError("split must be train, valid, or test")

        cache_path = Path(cache_path).resolve()
        partition_path = Path(partition_path)

        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)

        split_code = self.SPLIT_CODES[split]
        partition_bytes = partition_path.read_bytes()
        partition_hash = hashlib.sha256(partition_bytes).hexdigest()

        expected_names = {
            name
            for name, code in (
                line.split()
                for line in partition_bytes.decode("utf-8-sig").splitlines()
            )
            if code == split_code
        }

        # Store encoded records in memory; do not keep a SQLite connection.
        with sqlite3.connect(
            cache_path.as_uri() + "?mode=ro", uri=True
        ) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'config'"
            ).fetchone()
            if row is None:
                raise ValueError("Cache configuration is missing")

            self.config = json.loads(row[0])

            if self.config.get("schema_version") != 1:
                raise ValueError("Unsupported cache schema")
            if self.config.get("point_order") != self.POINT_ORDER:
                raise ValueError("Cache point order mismatch")
            if self.config.get("coordinates") != "original RGB image pixels":
                raise ValueError("Cache coordinate system mismatch")
            if self.config.get("partition_sha256") != partition_hash:
                raise ValueError("Cache partition hash mismatch")

            self.records = dict(connection.execute(
                "SELECT name, result FROM detections WHERE split = ?",
                (split_code,),
            ))

        actual_names = set(self.records)
        if actual_names != expected_names:
            missing = len(expected_names - actual_names)
            extra = len(actual_names - expected_names)
            raise ValueError(
                f"Cache split mismatch: missing={missing}, extra={extra}"
            )

        self.split = split

    def __len__(self):
        return len(self.records)

    def get(self, name, image_size):
        """Return (raw points or None, status), without GT substitution."""
        if name not in self.records:
            raise KeyError(f"No {self.split} cache record for {name}")

        result = json.loads(self.records[name])

        if result.get("image_size") != list(image_size):
            raise ValueError(f"Cache image-size mismatch for {name}")

        status = result.get("status")

        if status in ("no_face", "invalid_output"):
            if result.get("landmarks") is not None:
                raise ValueError(f"Failed detection contains points: {name}")
            return None, status

        if status != "ok":
            raise ValueError(f"Unknown cache status for {name}: {status}")

        points = torch.tensor(
            result["landmarks"], dtype=torch.float32
        )
        if points.shape != (5, 2):
            raise ValueError(f"Invalid landmark shape for {name}")
        if not torch.isfinite(points).all():
            raise ValueError(f"Non-finite landmarks for {name}")

        return points, status
