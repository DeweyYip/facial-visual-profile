"""Fixed MTCNN detector for raw RGB images."""

import numpy as np
import torch
from facenet_pytorch import MTCNN


class MTCNNDetector:
    """Return five landmarks in original-image pixel coordinates."""

    CONFIG = {
        "keep_all": True,
        "min_face_size": 20,
        "thresholds": [0.6, 0.7, 0.7],
        "factor": 0.709,
        "selection": "highest_probability",
    }

    POINT_ORDER = (
        "lefteye",
        "righteye",
        "nose",
        "leftmouth",
        "rightmouth",
    )

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = MTCNN(
            keep_all=True,
            min_face_size=self.CONFIG["min_face_size"],
            thresholds=list(self.CONFIG["thresholds"]),
            factor=self.CONFIG["factor"],
            device=self.device,
        ).eval()

    @torch.inference_mode()
    def detect(self, image):
        """Return a JSON-serializable result; never use GT for selection."""
        image = image.convert("RGB")
        boxes, scores, points = self.model.detect(image, landmarks=True)

        result = {
            "status": "no_face",
            "image_size": list(image.size),
            "face_count": 0 if boxes is None else len(boxes),
            "selected_index": None,
            "probability": None,
            "box": None,
            "landmarks": None,
        }

        if result["face_count"] == 0:
            return result

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        count = result["face_count"]

        if (
            boxes.shape != (count, 4)
            or scores.shape != (count,)
            or points.shape != (count, 5, 2)
            or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or not np.isfinite(points).all()
        ):
            result["status"] = "invalid_output"
            return result

        selected = int(np.argmax(scores))
        result.update(
            status="ok",
            selected_index=selected,
            probability=float(scores[selected]),
            box=boxes[selected].tolist(),
            landmarks=points[selected].tolist(),
        )
        return result
