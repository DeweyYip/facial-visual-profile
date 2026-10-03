"""CelebA attributes with cached predicted five-point landmarks."""
"""
it puts celbeA image, celebA labels, and MTCNN predicted landmarks together
celeba.py gives image, labels, GT landmarks and filename
here in celeba_predicted.py, landmarks = MTCNN prediction

MTCNN -> calculate the landmarks
SQLite -> store the landmarks
celeba_predicted.py -> get the landmarks and process them
"""
# SQLite results is stored in Google drive: MyDrive/FacialVisualProfile/cache/mtcnn_clean.sqlite

"""
CelebAPredictedAttributes don't just simply retreive the landmarks
the original landmarks in sqlite is in raw format: [5, 2]
CelebAPredictedAttributes is responsible to change them into [10] and it deals with resizing
"""

from PIL import Image
import torch

from datasets.celeba import CelebAAttributes
from datasets.landmark_cache import MTCNNLandmarkCache
from datasets.preprocessing import prepare


class LandmarkUnavailableError(RuntimeError):
    """A sample has no usable predicted landmarks."""


# it inherit CelebAAttributes, cuz we only change the source of landmark here
class CelebAPredictedAttributes(CelebAAttributes):

    """Preserve baseline images and labels; use predicted geometry.

    Successful samples return:
        image_tensor, labels, predicted_points, filename

    failure_policy="error": return four items and raise on failure.
    failure_policy="mask": return a fifth boolean geometry_valid flag;
    retain failed samples with zero normalized coordinate placeholders.
    The model must mask their geometry features after the MLP.
    """

    """
    when we initiate an instance:
    train_dataset = CelebAPredictedAttributes(
        cache_path="/content/drive/MyDrive/FacialVisualProfile/cache/mtcnn_clean.sqlite",
        split="train",
        root="data/raw",
    )
    """

    def __init__(
        self, cache_path, split="train", root="data/raw",
        failure_policy="error",
    ):
        if failure_policy not in ("error", "mask"):
            raise ValueError("failure_policy must be error or mask")
        self.failure_policy = failure_policy
        # Reuse official partitions, attribute order, and label loading.
        super().__init__(split=split, root=root)

        # this is just a SQLite reader. It reads an image's MTCNN result
        # MTCNNLandmarkCache is just a reader
        # e.g. self.predicted_cache.get("000001.jpg", ...)
        self.predicted_cache = MTCNNLandmarkCache(
            cache_path=cache_path,
            partition_path=self.root / "list_eval_partition.txt",
            split=split,
        )

        # data integrity check
        if set(self.names) != set(self.predicted_cache.records):
            raise ValueError("Dataset and cache image IDs differ")

    """
    this dataset will be constantly used by DataLoader
    dataset[100] = dataset.__getitem__(100)
    """
    def __getitem__(self, index):

        # using index to get the name of the file of the image
        name = self.names[index]
        image_path = self.root / "img_align_celeba" / name

        with Image.open(image_path) as image:

            # use self.predicted_cache.get(...) to go to the SQLite to get the data
            raw_points, status = self.predicted_cache.get(
                name,
                image_size=image.size,
            )

            geometry_valid = raw_points is not None

            if not geometry_valid:
                if self.failure_policy == "error":
                    raise LandmarkUnavailableError(
                        f"No predicted landmarks for {name}: {status}"
                    )

                # Prepare the image identically, then use normalized zeros.
                # These are placeholders, not detected or GT coordinates.
                image_tensor, _ = prepare(image, torch.zeros(5, 2))
                predicted_points = torch.zeros(10, dtype=torch.float32)
            else:
                # Use the exact same image and coordinate transformation.
                image_tensor, predicted_points = prepare(image, raw_points)

        sample = (
            image_tensor,
            self.targets[name],
            predicted_points,
            name,
        )
        if self.failure_policy == "mask":
            return (*sample, geometry_valid)
        return sample
