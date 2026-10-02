# Facial Visual Profile — Current State v1

Updated: 2026-10-02 (Asia/Shanghai). Handoff after Day 8.

Evidence: code and execution results shared in the development conversation. The Mac repository and Google Drive were not directly inspected when preparing this document. Paths and completion claims below refer to those reported results. This workspace contains the handoff document, not a checkout of the project.

## 1. Project Goal / Research Question

Build a reproducible system for predicting 24 facial attributes from CelebA images, with image-only baselines and a subsequent image-plus-five-landmark model.

The working research question is whether detected facial geometry improves attribute prediction and, in later experiments, robustness to image corruption. The baselines are established; geometry and robustness experiments are not yet implemented or validated. This project supports a master's application, so fair comparisons, provenance, and reproducible results matter more than isolated headline scores.

## 2. Model Design: Implemented vs. Planned

| Model | Design | Current status |
|---|---|---|
| R18 baseline | ImageNet-pretrained ResNet18; replace FC with `Linear(512, 24)` | Implemented; seed-0 training and validation analysis complete |
| R50 baseline | ImageNet-pretrained ResNet50; replace FC with `Linear(2048, 24)` | Implemented; seed-0 training complete |
| Geometry model | R18 image features (512) + MLP embedding of five points (32); concatenate to 544 features and predict 24 logits | Planned Day 9; not implemented |

The geometry model is referred to as M3 in the schedule. Its required interface is `[B, 544] -> [B, 24]`. MLP hidden layers, activation choices, and the classification head remain to be settled in code. Do not present these as finalized architecture details.

MTCNN is a separate pretrained detector that supplies coordinates; the baseline models do not consume those coordinates. Detector joint training has not been implemented.

## 3. Dataset and Attribute Order

CelebA files: `img_align_celeba.zip`, `list_attr_celeba.txt`, `list_landmarks_align_celeba.txt`, and `list_eval_partition.txt`. Use aligned images and their aligned five-point annotations.

| Split | Official code | Images |
|---|---:|---:|
| Train | 0 | 162,770 |
| Validation | 1 | 19,867 |
| Test | 2 | 19,962 |
| Total | | 202,599 |

All four filename sets matched exactly, with no duplicates. The archive integrity check passed. A checked original image was RGB, 178 × 218 pixels. Attribute labels are converted from `-1/1` to float32 `0/1`.

The following order is mandatory across labels, output logits, checkpoints, and thresholds:

```text
Smiling, Mouth_Slightly_Open, Eyeglasses, Bangs,
Wearing_Hat, Mustache, No_Beard, Black_Hair,
Bushy_Eyebrows, Arched_Eyebrows, High_Cheekbones, Bags_Under_Eyes,
Big_Lips, Big_Nose, Double_Chin, Oval_Face,
Receding_Hairline, Blond_Hair, Brown_Hair, Gray_Hair,
Bald, Straight_Hair, Wavy_Hair, Sideburns
```

Shared preprocessing preserves aspect ratio, resizes the longest side to 224, and centers the image on a black 224 × 224 canvas. Normalize RGB with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.

Landmarks use the same resize and padding, including the pixel-center mapping `(coord + 0.5) * new_dimension / original_dimension - 0.5 + padding`. Normalize coordinates with `2 * coord / 223 - 1`, then flatten to 10 values. Point order: left eye, right eye, nose, left mouth, right mouth, each as `(x, y)`.

## 4. Important Engineering Files

This is the verified important-file inventory, not an exhaustive repository listing.

| Path | Role |
|---|---|
| `configs/attributes.yaml` | Ordered 24-attribute definition |
| `datasets/celeba.py` | Official split loading; returns image, labels, GT points, filename |
| `datasets/preprocessing.py` | Shared image and coordinate transformation |
| `models/resnet18.py`, `models/resnet50.py` | Image-only classifiers |
| `detectors/mtcnn.py` | MTCNN wrapper and face selection |
| `scripts/cache_mtcnn.py` | Resumable SQLite detection cache |
| `scripts/inspect_celeba.py` | GT landmark visual inspection |
| `scripts/overfit_resnet18.py` | Eight-sample training check |
| `evaluation/attributes.py` | Macro-F1 and per-attribute AP/mAP |
| `training/__init__.py` | Package exists; reusable trainer implementation is unconfirmed |
| `docs/data_inventory.md`, `docs/experiment_log.md` | Data verification and experiment record |
| `docs/day7_r18_validation.md` | R18 validation analysis |
| `configs/thresholds/r18_seed0.json` | R18 validation-tuned thresholds |
| `docs/results/r18_seed0/`, `docs/results/r50_seed0/` | Imported small experiment artifacts |
| `notebooks/day4_resnet18_benchmark.ipynb` | Initial GPU benchmark |
| `notebooks/day5_mtcnn_inspection.ipynb` | Detector inspection and cache assessment |
| `notebooks/day6_resnet18_seed0.ipynb` | Formal R18 run |
| `notebooks/day7_resnet18_validation.ipynb` | Reproduction, thresholds, error analysis |
| `notebooks/day8_resnet50_seed0.ipynb` | R50 run, recovery, and results |

Large datasets, weights, and caches belong outside Git. `/data/` is ignored. Clean notebook outputs and execution metadata before committing.

## 5. Completed Work by Day

| Day | Verified work |
|---|---|
| 1 | Repository/environment foundation exists; exact Day 1 tasks are not available in the visible record |
| 2 | Downloaded and checked CelebA images, labels, aligned landmarks, and official partitions |
| 3 | Dataset and preprocessing; batch checks; visually inspected 50 GT overlays |
| 4 | R18 forward check, small-sample overfit check, and one-epoch T4 benchmark |
| 5 | MTCNN inspection, full clean-image cache, train/validation landmark quality assessment |
| 6 | R18 seed-0 training for 10 epochs; validation F1/mAP; checkpoint selection |
| 7 | Reproduced selected R18 validation results; fitted thresholds; inspected attribute errors |
| 8 | R50 seed-0 training for 10 epochs, including checkpoint/optimizer recovery; archived results |

## 6. Successful Verification

- Dataset batches: images `[B, 3, 224, 224]`, labels `[B, 24]`, GT points `[B, 10]`; binary labels and correct split sizes.
- All 50 sampled GT overlays appeared correctly placed; no sampled points outside the canvas.
- R18: 11,188,824 parameters; finite forward/loss. Eight-sample overfit passed at step 20, evaluation BCE 0.003705 and exact match 100% on those same samples.
- Day 4 T4 benchmark: 162,770 images in 9.00 minutes, mean training BCE 0.213719, throughput 301.3 images/s, peak allocated GPU memory 1.64 GiB.
- R50: 23,557,208 parameters. GPU batch-64 forward, backward, and Adam update passed; peak allocated memory 5.30 GiB.
- Metric perfect-prediction check returned macro-F1 = mAP = 1.0.
- Cache rerun skipped all 50 existing smoke records, confirming resume behavior.
- R50 resumed from epoch 8 with Adam step count 20,352 and completed epochs 9–10.

## 7. Training, Checkpoints, and Evaluation

Shared baseline configuration: seed 0; 10 epochs; batch 64; Adam; learning rate 0.0001; weight decay 0; FP32; two workers; epoch seed rule `seed + epoch`; all backbone parameters trainable; BCEWithLogitsLoss; no augmentation. No AMP, scheduler, or positive-class weighting was used.

Select the checkpoint by highest validation mAP; retain the earliest epoch on ties. Baseline F1 uses probability threshold 0.5.

| Selected result | R18 | R50 |
|---|---:|---:|
| Epoch | 3 | 3 |
| Validation BCE | 0.190717 | 0.190218 |
| Macro-F1 at 0.5 | 0.727259 | 0.731168 |
| mAP | 0.817737 | 0.818835 |
| Epoch-10 training BCE | 0.0416 | 0.046406 |
| Epoch-10 validation BCE | 0.3669 | 0.364135 |
| Epoch-10 mAP | 0.7698 | 0.777675 |

Both runs overfit after approximately epoch 3. R50's selected mAP gain is about 0.11 percentage points; one seed does not establish a reliable advantage.

R18 selected-checkpoint validation was independently reproduced on all 19,867 images, with zero reported differences in BCE, F1, and mAP. Per-attribute validation threshold tuning increased macro-F1 to 0.762372; mAP remained unchanged. This is an in-sample validation tuning score, not an independent generalization estimate. R50's independent full-validation reproduction and threshold tuning are not yet completed. Neither model has been evaluated on the test split.

Persistent Drive root: `/content/drive/MyDrive/FacialVisualProfile/`.

- `runs/r18_seed0/`: checkpoints, config, source snapshot, history, selected-checkpoint summary, and curves; selected `epoch_03.pt`.
- `runs/r18_seed0/day7/`: validation predictions, reproduction report, thresholds, and attribute CSV.
- `runs/r50_seed0/`: `epoch_00.pt` through `epoch_10.pt`, config, source snapshot, history, `best.json`, metrics CSV, curves, and recovery check. Selected `epoch_03.pt`; training checkpoints approximately 270 MiB each.
- `cache/mtcnn_clean.sqlite` and `cache/mtcnn_clean_quality.json`: persistent detector results.
- `data/`: original image archive and three annotation files.

## 8. Important Decisions and Rationale

- Preserve official splits and attribute order to prevent leakage and output mismatches.
- Keep identical preprocessing and training settings for baseline comparisons. R50 used the saved R18 shared-source snapshot because current GitHub dataset/metric files differed from that run.
- Treat saved source bytes and SHA-256 hashes as authoritative experiment provenance. A Git commit alone does not describe the mixed-source R50 workspace.
- Save Adam state and history with training checkpoints to support epoch-boundary recovery.
- Cache MTCNN predictions on original aligned images in native pixel coordinates. Select the highest-confidence face without using GT landmarks.
- Transform cached points together with geometric augmentation. For future corruption experiments, rerun the detector on corrupted images; clean predictions cannot represent detector behavior under corruption.
- Never silently substitute GT coordinates when detection fails. A concrete failure-handling policy is still pending.
- Keep official labels during error analysis. `Bags_Under_Eyes` means under-eye bags, not simply dark circles; apparent disagreements are not automatically annotation errors.

## 9. Unfinished Work

Geometry MLP and concat model; coordinate/cache reader integration; detector-failure handling; M3 smoke/overfit checks and formal training; additional seeds; corruption evaluation; final test evaluation; reusable training-script extraction if needed.

MTCNN cache: 202,599 records; 202,306 `ok`, 293 `no_face` (0.14%); 3,754 multiple-face images. Train NME mean/median/95th percentile: 6.41% / 5.40% / 11.34%; validation: 6.21% / 5.37% / 10.78%. NME is normalized by GT eye distance and excludes failed detections. Test detections were cached, but test landmark quality was not evaluated.

## 10. Next Development Order

1. Day 9: inspect existing sources on Mac; implement the geometry MLP and R18 concat model; verify shapes, gradients in both branches, and parameter count. No full training yet.
2. Day 10: inspect the actual SQLite schema and wrapper; integrate predicted coordinates; verify point order, normalization, augmentation consistency, and missing/invalid records.
3. Day 11: specify failure handling and run M3 small-batch checks before seed-0 training.
4. Complete M3 and compare clean validation results under the established checkpoint-selection rule. R50 training is already finished.
5. Reproduce R50 validation if required; then plan additional seeds and corruption experiments. Reserve test evaluation for a fixed protocol.

## 11. Most Important Files to Read First

Read `docs/experiment_log.md`, `configs/attributes.yaml`, `datasets/preprocessing.py`, `datasets/celeba.py`, both baseline model files, and `evaluation/attributes.py`. For geometry integration, also read `detectors/mtcnn.py` and `scripts/cache_mtcnn.py`. For exact baseline provenance, consult each Drive run's `config.json`, `source/`, `history.json`, and `best.json`.

## 12. New Chatbox Handoff Notes

Mac repository: `/Users/deweye/Desktop/Facial Visual Profile`; GitHub: `DeweyYip/facial-visual-profile`. Latest confirmed commit: `e91dc65`, synchronized with `origin/main`.

The worktree is not clean: modified `models/resnet50.py` and untracked `notebooks/day8_resnet50_seed0(upper_half).ipynb` were deliberately excluded from the results commit. Inspect and preserve them; do not overwrite or delete automatically.

Colab `/content` files are temporary. Restore from Drive source snapshots and data after runtime replacement; do not restart completed baseline training. The previous R50 workspace was `/content/r50_seed0_workspace`. The user has Colab Pro; last confirmed GPU was Tesla T4. Runtime availability must be checked afresh.

Working style: explain each code block in Chinese before giving it, including Mac/Colab location, purpose, outputs, expected duration, and whether training starts. Code, comments, logs, and project documents use English. Proceed one logical step at a time and inspect results. Avoid bare shell comments in pasted Mac zsh commands; use Python comments inside heredocs. Use `python3` or the active virtual environment.

Do not infer implementation from schedules, confuse cached detection with model training, claim checkpoint existence proves evaluation, or compare validation-tuned F1 against fixed-threshold F1 without labeling the difference.
