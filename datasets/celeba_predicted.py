"""CelebA attributes with cached predicted five-point landmarks."""

from PIL import Image

from datasets.celeba import CelebAAttributes
from datasets.landmark_cache import MTCNNLandmarkCache
from datasets.preprocessing import prepare


class LandmarkUnavailableError(RuntimeError):
    """A sample has no usable predicted landmarks."""


class CelebAPredictedAttributes(CelebAAttributes):
    """Preserve baseline images and labels; use predicted geometry.

    Successful samples return:
        image_tensor, labels, predicted_points, filename

    Failed detections raise explicitly during integration checks.
    The final training-time failure policy is not implemented here.
    """

    def __init__(self, cache_path, split="train", root="data/raw"):
        # Reuse official partitions, attribute order, and label loading.
        super().__init__(split=split, root=root)

        self.predicted_cache = MTCNNLandmarkCache(
            cache_path=cache_path,
            partition_path=self.root / "list_eval_partition.txt",
            split=split,
        )

        if set(self.names) != set(self.predicted_cache.records):
            raise ValueError("Dataset and cache image IDs differ")

    def __getitem__(self, index):
        name = self.names[index]
        image_path = self.root / "img_align_celeba" / name

        with Image.open(image_path) as image:
            raw_points, status = self.predicted_cache.get(
                name,
                image_size=image.size,
            )

            if raw_points is None:
                raise LandmarkUnavailableError(
                    f"No predicted landmarks for {name}: {status}"
                )

            # Use the exact same image and coordinate transformation.
            image_tensor, predicted_points = prepare(image, raw_points)

        return (
            image_tensor,
            self.targets[name],
            predicted_points,
            name,
        )
