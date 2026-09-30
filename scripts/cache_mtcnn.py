"""Cache MTCNN results with resumable SQLite storage."""
"""
Run MTCNN over the dataset and cache face boxes and estimated landmarks
in a resumable SQLite database.
by using the detector defined in mtcnn.py 
"""

"""
This code is ran on Colab for saving time 
The report is stored in Google cloud
"""

import argparse
import hashlib
import json
import sqlite3
import time
from importlib import metadata
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# the real geer 
from detectors.mtcnn import MTCNNDetector

# assign a number to each image to better recognize them later on
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    # Read the related data
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    split_path = args.root / "list_eval_partition.txt"
    rows = [line.split() for line in split_path.read_text().splitlines()]
    if any(len(row) != 2 or row[1] not in ("0", "1", "2") for row in rows):
        raise ValueError("Invalid partition file")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")

    if args.limit:
        rows = rows[:args.limit]

    # The official MTCNNDetecor instance
    detector = MTCNNDetector(device=args.device)
    package_root = Path(__import__("facenet_pytorch").__file__).parent
    weights = {
        name: sha256(package_root / "data" / name)
        for name in ("pnet.pt", "rnet.pt", "onet.pt")
    }

    # Use Config for reproducibility in the future
    config = {
        "schema_version": 1,
        "detector": "facenet-pytorch MTCNN",
        "versions": {
            name: metadata.version(name)
            for name in (
                "facenet-pytorch", "torch", "torchvision", "numpy", "Pillow"
            )
        },
        "device": str(detector.device),
        "settings": detector.CONFIG,
        "point_order": list(detector.POINT_ORDER),
        "coordinates": "original RGB image pixels",
        "partition_sha256": sha256(split_path),
        "detector_source_sha256": sha256(Path("detectors/mtcnn.py")),
        "weight_sha256": weights,
    }
    encoded_config = json.dumps(config, sort_keys=True)

    # a SQLite instance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    try:
        # Create a table for metadata
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        # Create a table for MTCNN of each image 
        connection.execute(
            "CREATE TABLE IF NOT EXISTS detections "
            "(name TEXT PRIMARY KEY, split TEXT NOT NULL, "
            "result TEXT NOT NULL)"
        )
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key = 'config'"
        ).fetchone()

        # Just in case we mix up the experienments
        if existing is not None and existing[0] != encoded_config:
            raise ValueError(
                "Cache configuration mismatch; use a new output file."
            )
        if existing is None:
            connection.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("config", encoded_config),
            )
        connection.commit()

        # For continuity - enabling our model to keep going under disconnection
        completed = {
            row[0] for row in connection.execute("SELECT name FROM detections")
        }
        pending = [(name, split) for name, split in rows if name not in completed]

        print("Requested images:", len(rows), flush=True)
        print("Already cached:", len(rows) - len(pending), flush=True)
        print("Pending images:", len(pending), flush=True)

        started = time.perf_counter()
        processed = 0
        try:
            for name, split in tqdm(pending, desc="Caching MTCNN"):
                image_path = args.root / "img_align_celeba" / name
                with Image.open(image_path) as image:
                    result = detector.detect(image)
                result["image_sha256"] = sha256(image_path)

                """ Format:
                    status = "ok"
                    face_count = 1
                    box =
                    [x1, y1, x2, y2]
                    landmarks =
                    [
                    [left_eye_x, left_eye_y],
                    [right_eye_x, right_eye_y],
                    [nose_x, nose_y],
                    [left_mouth_x, left_mouth_y],
                    [right_mouth_x, right_mouth_y]
                    ]
                """

                # Put the result into SQLite
                connection.execute(
                    "INSERT INTO detections VALUES (?, ?, ?)",
                    (name, split, json.dumps(result, allow_nan=False)),
                )
                processed += 1
                if processed % 100 == 0:
                    connection.commit()
        finally:
            connection.commit()

        elapsed = time.perf_counter() - started
        total = connection.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        print("Newly processed:", processed)
        print("Total cached:", total)
        print(f"Elapsed seconds: {elapsed:.1f}")
        if processed:
            print(f"Seconds per image: {elapsed / processed:.4f}")
        print("Saved:", args.output)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
