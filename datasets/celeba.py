"""CelebA images, 24 selected attributes, and aligned GT landmarks."""
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml

from datasets.preprocessing import prepare


class CelebAAttributes(Dataset):
    def __init__(self, split="train", root="data/raw"):
        split_codes = {"train": "0", "valid": "1", "test": "2"}
        if split not in split_codes:
            raise ValueError("split must be train, valid, or test")

        self.root = Path(root)
        with open("configs/attributes.yaml", encoding="utf-8") as f:
            self.attribute_names = yaml.safe_load(f)["attributes"]

        with (self.root / "list_attr_celeba.txt").open(
            encoding="utf-8-sig"
        ) as f:
            next(f)  # Number of images
            all_names = next(f).split()
            selected = [all_names.index(name) for name in self.attribute_names]
            self.targets = {}
            for line in f:
                name, *values = line.split()
                self.targets[name] = torch.tensor(
                    [int(values[i]) == 1 for i in selected],
                    dtype=torch.float32,
                )

        with (self.root / "list_landmarks_align_celeba.txt").open(
            encoding="utf-8-sig"
        ) as f:
            next(f)  # Number of images
            next(f)  # Ten coordinate names
            self.landmarks = {}
            for line in f:
                name, *values = line.split()
                self.landmarks[name] = torch.tensor(
                    [float(value) for value in values], dtype=torch.float32
                )

        with (self.root / "list_eval_partition.txt").open(
            encoding="utf-8-sig"
        ) as f:
            self.names = [
                name
                for line in f
                for name, code in [line.split()]
                if code == split_codes[split]
            ]

        if any(
            name not in self.targets or name not in self.landmarks
            for name in self.names
        ):
            raise ValueError("Some image IDs are missing labels or landmarks")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / "img_align_celeba" / name) as image:
            image_tensor, points = prepare(image, self.landmarks[name])

        return image_tensor, self.targets[name], points, name
