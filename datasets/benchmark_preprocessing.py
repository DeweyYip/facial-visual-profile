"""Expose pre-normalization canvas without changing established prepare()."""
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from .preprocessing import SIZE, MEAN, STD


def prepare_canvas(image, landmarks):
    """Same letterbox and pixel-center mapping as current preprocessing.py.

    Returns RGB PIL canvas, GT pixel coordinates [5,2]. No normalization.
    """
    width, height = image.size
    scale = SIZE / max(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (SIZE - new_width) // 2, (SIZE - new_height) // 2
    resized = image.convert("RGB").resize(
        (new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(resized, (left, top))
    points = landmarks.clone().reshape(5, 2).float()
    points[:, 0] = (points[:, 0] + 0.5) * new_width / width - 0.5 + left
    points[:, 1] = (points[:, 1] + 0.5) * new_height / height - 0.5 + top
    return canvas, points


def normalize_canvas(canvas):
    if canvas.mode != "RGB" or canvas.size != (SIZE, SIZE):
        raise ValueError("Expected final unnormalized RGB 224x224 canvas")
    return TF.normalize(TF.to_tensor(canvas), MEAN, STD)


def normalize_points(pixel_points):
    """For transformed GT or newly detected final-canvas points; no resizing."""
    points = torch.as_tensor(pixel_points, dtype=torch.float32)
    if points.shape not in ((5, 2), (10,)) or not torch.isfinite(points).all():
        raise ValueError("Expected finite five-point pixel coordinates")
    return (2 * points.reshape(5, 2) / (SIZE - 1) - 1).flatten()
