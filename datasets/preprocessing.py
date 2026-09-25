"""Shared image and landmark preprocessing for every model."""
from PIL import Image
import torch
from torchvision.transforms import functional as TF

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prepare(image: Image.Image, landmarks: torch.Tensor):
    """Return normalized image [3,224,224] and five points [10] in [-1,1]."""
    width, height = image.size
    scale = SIZE / max(width, height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    left = (SIZE - new_width) // 2
    top = (SIZE - new_height) // 2

    resized = image.convert("RGB").resize(
        (new_width, new_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(resized, (left, top))

    points = landmarks.clone().reshape(5, 2).float()
    points[:, 0] = (points[:, 0] + 0.5) * new_width / width - 0.5 + left
    points[:, 1] = (points[:, 1] + 0.5) * new_height / height - 0.5 + top
    points = 2 * points / (SIZE - 1) - 1

    image_tensor = TF.normalize(TF.to_tensor(canvas), MEAN, STD)
    return image_tensor, points.flatten()
