"""Visually inspect 50 CelebA images and their transformed GT landmarks."""
"""
Just a data sanity check
The size of our image has been changed after preprocessing
So we expect the corresponding landmark will change too
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from datasets.celeba import CelebAAttributes
from datasets.preprocessing import MEAN, STD, SIZE

dataset = CelebAAttributes("train")    # training set; dataset[0] gives image, target, coords, name
indices = random.Random(2026).sample(range(len(dataset)), 50) # get 50 images from training samples
out = Path("outputs/figures") # place we store the result
out.mkdir(parents=True, exist_ok=True)

"""
Reverse Normalization to restore images 
image * std + mean
if we directly use plt.imshow(image), the color would be weird
"""
mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)
point_labels = ("LE", "RE", "N", "LM", "RM") # E: eyes; N: noese; M: mouth
label_lines = []
outside = 0

# Each page has 10 images
for page in range(5):
    # Subplots has 2 rows, 5 columns - 10 images included
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), layout="constrained")
    for ax, index in zip(axes.flat, indices[page * 10:(page + 1) * 10]):
        image, target, coords, name = dataset[index]
        # reverse normalization
        pixels = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        # Prepare landmarks
        points = ((coords.reshape(5, 2) + 1) * (SIZE - 1) / 2).numpy()
        # positive facial attributes
        positives = [
            label for label, value in zip(dataset.attribute_names, target)
            if value.item() == 1
        ]
        label_lines.append(f"{name}: {', '.join(positives)}")
        outside += int(((points < 0) | (points >= SIZE)).any())

        # Show the image 
        ax.imshow(pixels)
        # Show the landmarks
        ax.scatter(points[:, 0], points[:, 1], s=25, c="red")
        # Annotate the landmarks
        for (x, y), label in zip(points, point_labels):
            ax.annotate(label, (x, y), xytext=(3, 3),
                        textcoords="offset points", color="yellow", fontsize=7)
        ax.set_title(f"{name}\n{', '.join(positives[:2])}", fontsize=9)
        ax.axis("off")

    filename = out / f"gt_landmarks_page_{page + 1:02d}.png"
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print("已保存：", filename)

(out / "gt_landmarks_labels.txt").write_text(
    "\n".join(label_lines) + "\n", encoding="utf-8"
)
print("检查样本：50")
print("关键点超出 224×224 画面的样本数：", outside)
print("完整阳性标签：outputs/figures/gt_landmarks_labels.txt")
