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

## ResNet18 seed-0 training and validation

- Model: ImageNet-pretrained ResNet18 with 24 attribute logits.
- Official split: 162,770 training images and 19,867 validation images.
- Seed: 0. Trained for 10 epochs with batch size 64, FP32, Adam, and learning rate 0.0001.
- Preprocessing: shared 224 × 224 pipeline; no random augmentation in this run.
- Validation macro-F1 used a fixed probability threshold of 0.5.
- Validation mAP is the mean of per-attribute average precision scores.
- Checkpoint selection rule: highest validation mAP.
- Selected checkpoint: `epoch_03.pt`; validation mAP 0.8177, macro-F1 0.7273, validation BCE loss 0.1907.
- Epoch 10: training BCE loss 0.0416, validation BCE loss 0.3669, validation mAP 0.7698.
- Training loss fell while validation loss rose after epoch 3, indicating overfitting.
- Checkpoints, config, history, best.json, epoch_metrics.csv, and training_curves.png are stored in Drive under `FacialVisualProfile/runs/r18_seed0/`.
- Training source commit: `43e51ca99bf98743ba2d7c4e2481e2b7751e97f7`.
- The test split was not evaluated.

## Day 8: ResNet50 seed-0 training

- Model: ImageNet-pretrained ResNet50 with 24 output logits.
- Parameters: 23,557,208.
- Official split: 162,770 training images and 19,867 validation images.
- Configuration: seed 0, 10 epochs, batch size 64, Adam, learning rate 0.0001, FP32.
- Shared preprocessing, attribute order, and metric source were restored from the R18 source snapshot.
- R50 model source commit: `f601fb6c140c790a35ff9b5564ad0d7a0eb562d5`.
- GPU: Tesla T4.
- Model and Adam restoration checks passed; resumed after epoch 8 and completed epochs 9 and 10.
- Selected checkpoint: `epoch_03.pt`, using validation mAP.
- Selected validation mAP: 0.818835.
- Selected macro-F1 at threshold 0.5: 0.731168.
- Selected validation BCE loss: 0.190218.
- Validation loss increased and mAP declined after epoch 3, indicating overfitting.
- All 10 epochs and saved run artifacts were checked.
- Checkpoints and source snapshot remain in Drive: `FacialVisualProfile/runs/r50_seed0/`.
- Small result files and training curves: `docs/results/r50_seed0/`.
- The test split was not evaluated.
- This is a single-seed clean-validation result; robustness has not been evaluated.

## Day 9 — Geometry model implementation

Date: 2026-10-02

Implemented `models/resnet18_geometry.py` with
`ResNet18GeometryAttributes`.

Architecture:
- Image branch: ResNet18 with its FC replaced by Identity; 512 features.
- Geometry branch: Linear(10, 32), ReLU, Linear(32, 32), ReLU.
- Fusion: concatenate image and geometry features; 544 features.
- Classifier: Linear(544, 24), returning logits.

Mac CPU smoke check:
- Seed: 0; batch size: 2; pretrained=False.
- Random images, normalized random coordinates, and binary labels.
- Image features: (2, 512).
- Geometry features: (2, 32).
- Combined features: (2, 544).
- Logits: (2, 24).
- BCEWithLogitsLoss: 0.684515.
- All trainable parameters received finite gradients.
- Each branch and the classifier had a nonzero gradient absolute sum.
- Total parameters: 11,191,000.
- Result: PASS.

Scope:
- This checks model wiring and gradient propagation only.
- No formal training or attribute-performance evaluation was performed.
- No MTCNN cache or real-coordinate integration was performed.

Next: inspect SQLite cache schema and integrate predicted coordinates,
checking point order, shared preprocessing, and missing/invalid records.
