# Experiment log

## Day 1
- Scope: repository structure and local Python environment.
- Environment check: record the terminal result after setup.
- No dataset download or training run yet.

## Day 3 — 2026-09-25
- Implemented a CelebA Dataset for the official train, validation, and test splits.
- Selected 24 attributes by name and mapped CelebA labels to 0/1.
- Added shared 224 × 224 preprocessing that preserves the full image and transforms GT landmarks with it.
- DataLoader shapes: images [4, 3, 224, 224], targets [4, 24], landmarks [4, 10].
- Visually inspected 50 randomly selected training images with GT landmark overlays.
- All 50 overlays appeared correctly placed; no sampled landmarks fell outside the image.
- No training was run. Review preprocessing again before the full training runs.

## Day 4: ResNet18 baseline and GPU benchmark

- Model: ImageNet-pretrained ResNet18 with 24 output logits.
- Loss function: BCEWithLogitsLoss.
- CPU forward check: passed.
- Eight-sample overfit check: passed at step 20.
- Overfit evaluation loss: 0.003705.
- Overfit exact match: 100% on the same eight training samples.
- GPU: Tesla T4.
- PyTorch: 2.11.0+cu128.
- Full training benchmark: one epoch, 162,770 samples.
- Configuration: FP32, Adam, learning rate 0.0001, batch size 64.
- Mean training loss: 0.213719.
- Epoch time: 9.00 minutes.
- Throughput: 301.3 images/second.
- Peak allocated GPU memory: 1.64 GiB.
- Checkpoint and JSON report: saved to Google Drive and checked for existence and nonzero size; JSON parsed successfully.
- Drive run directory: `FacialVisualProfile/runs/day4_r18_20260927T115428425737Z/`.
- No validation or test evaluation was performed.
- This checkpoint is a training benchmark result, not a selected final model.

## MTCNN clean-image detection and quality assessment

- Detector: facenet-pytorch 2.6.0 MTCNN; Tesla T4.
- Environment: Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128.
- Installed with --no-deps outside the declared dependency range; sampled and full-dataset inference completed.
- Settings: min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True.
- Face selection: highest probability, without GT-based selection.
- Output: original-image pixel coordinates, ordered as left eye, right eye, nose, left mouth, right mouth.
- Visually inspected 50 sampled training images and the multiple-face example.
- Cached 202,599 images: 202,306 valid outputs and 293 no-face results.
- Multiple-face images: 3,754. Target-face selection is not guaranteed for every image.
- Full cache run: 79.7 minutes; persistent backup saved to Drive.
- NME: mean five-point Euclidean error divided by GT inter-eye distance, evaluated on successful detections only.
- Train: 162,770 images; 249 no-face results; 2,979 multiple-face images.
- Train NME: mean 6.41%, median 5.40%, 95th percentile 11.34%.
- Validation: 19,867 images; 32 no-face results; 355 multiple-face images.
- Validation NME: mean 6.21%, median 5.37%, 95th percentile 10.78%.
- Invalid GT count: zero in training and validation.
- Test predictions cached; test NME not evaluated.
- Drive cache: `FacialVisualProfile/cache/mtcnn_clean.sqlite`.
- Drive report: `FacialVisualProfile/cache/mtcnn_clean_quality.json`.
- Augmented-image handling and geometry-branch detection-failure handling remain to be implemented.
- GT must not replace missing predictions.
