# CelebA data inventory — Day 2

Source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Local files (under data/raw/):
- img_align_celeba.zip — aligned and cropped images
- img_align_celeba/ — extracted images
- list_attr_celeba.txt — 40 attribute labels
- list_landmarks_align_celeba.txt — aligned 5-point landmarks
- list_eval_partition.txt — official train/validation/test split

Checks completed:
- ZIP integrity check passed.
- 202,599 images extracted; image filenames have no duplicates.
- All four sources contain the same 202,599 image filenames.
- Official split: train 162,770; validation 19,867; test 19,962.
- All 24 attributes in configs/attributes.yaml exist in the label file.
- First image (000001.jpg) opens as RGB, size 178 × 218.
- First image matches its attribute, landmark and split records.
- Git ignores files under data/.

Next: implement Dataset/DataLoader and visually inspect images with GT landmarks.
