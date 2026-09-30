"""CelebA images, 24 selected attributes, and aligned GT landmarks."""
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml

from datasets.preprocessing import prepare


"""
when we inherit Dataset and want to create our own dataset
we need to override __len__ and __getitem__
"""

"""
information about the 24 attributes is stored in attrubutes.yaml
"""

class CelebAAttributes(Dataset):

    # when creating an instance:
    # we need to specify which dataset we need 
    # and root tell us the place

    """
    The initialization of celebA dataset is about:
    reading the three txt files from CelebA
    and build a image -> label / landmark relationship
    self below represents all the image int the data set; not a single image
    """

    def __init__(self, split="train", root="data/raw"):
        
        """
        The format we have in list_eval_partition.txt is:
        000001.jpg 0
        000002.jpg 2
        000003.jpg 1
        ...
        It specifies well that which picure is in what category
        """

        split_codes = {"train": "0", "valid": "1", "test": "2"}

        if split not in split_codes:
            raise ValueError("split must be train, valid, or test")

        self.root = Path(root)

        # open the yaml file and build a dictionary
        """
        {
            "attributes":
            ["Big_Nose",
            "Smiling",
            ...]
        }
        """
        with open("configs/attributes.yaml", encoding="utf-8") as f:
            self.attribute_names = yaml.safe_load(f)["attributes"]
        """
        Dealing with the attributes:
        5_o_Clock_Shadow Arched_Eyebrows Attractive...
            000001.jpg  -1  1 -1 ...
            000002.jpg   1 -1  1 ...
        """
        with (self.root / "list_attr_celeba.txt").open(
            encoding="utf-8-sig"
        ) as f:
            next(f)  # Jump over the first line bc it's about the number of images
            all_names = next(f).split() # Getting all the attributes names and split them ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive",...]
            selected = [all_names.index(name) for name in self.attribute_names]
            self.targets = {} # Find the 24 target attributes 
            # looping thru all lines in list_attr.txt
            for line in f:
                name, *values = line.split()    # split the name and values in [000001.jpg -1 1 -1 1 ...]
                # create image name -> label tensor 
                self.targets[name] = torch.tensor(
                    [int(values[i]) == 1 for i in selected],
                    dtype=torch.float32,
                )
        """
        we created: 
        slef.targets
        {
            "000001.jpg": tensor([24]),
            "000002.jpg": tensor([24]),
            ...
        }
        """

        # Everything works the same with the landmarks
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
        """
        self.landmarks = {
            "000001.jpg": tensor([...]),
            "000002.jpg": tensor([...]),
            "000003.jpg": tensor([...]),
        }
        """

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
        
        """
        Different sets have different names
        For valid set it might be:
        self.names = [
        "162771.jpg",
        "162772.jpg"
        ]
        """


    def __len__(self):
        return len(self.names)

    """
    getitem gets a single image for us
    index  = 100 : i want the 100th training sample。
    """
    def __getitem__(self, index):
        name = self.names[index]
        # Prepare the image tensor by using prepare function
        with Image.open(self.root / "img_align_celeba" / name) as image:
            image_tensor, points = prepare(image, self.landmarks[name])

        # What it returns
        return image_tensor, self.targets[name], points, name
