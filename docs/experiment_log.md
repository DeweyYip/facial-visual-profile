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

## Day 10 — Predicted landmark cache reader

Date: 2026-10-03

Inspected the persistent MTCNN cache in Colab:
- Schema version: 1.
- Coordinates: original RGB image pixels.
- Point order: lefteye, righteye, nose, leftmouth, rightmouth.
- Train: 162,521 ok; 249 no_face.
- Validation: 19,835 ok; 32 no_face.
- Test: 19,950 ok; 12 no_face.
- Total: 202,599 records.

Implemented `datasets/landmark_cache.py`:
- Loads one official split into memory using read-only SQLite.
- Checks schema, coordinate system, point order, partition hash,
  and exact split membership.
- Returns float32 original-image coordinates with shape (5, 2).
- Returns None and the failure status for unsuccessful detections.
- Never substitutes GT coordinates.

Mac smoke check using a temporary synthetic database:
- Official split selection: PASS.
- Original coordinates preserved: PASS.
- Detection failure remains explicit: PASS.
- Missing record rejection: PASS.
- Image-size mismatch rejection: PASS.
- Invalid coordinate shape rejection: PASS.
- Non-finite coordinate rejection: PASS.
- Incomplete split rejection: PASS.
- Partition hash mismatch rejection: PASS.

Pending:
- Real-cache reader verification and image/coordinate preprocessing checks.
- Dataset integration and explicit training-time failure policy.
- No formal geometry-model training has started.

### Day 10 — Coordinate alignment and predicted Dataset

Date: 2026-10-03

Colab checks using source commit c4fd5b6:
- Real-cache reader passed for all 162,770 training and 19,867
  validation records, including explicit no_face results.
- Sampled 3 training and 3 validation images with seed 0:
  101140.jpg, 110412.jpg, 010626.jpg,
  171267.jpg, 179547.jpg, 178715.jpg.
- All six image hashes and image sizes matched cache records.
- Image tensors were unchanged by the coordinate input.
- Maximum coordinate mapping error was approximately 0.00000763 pixels.
- Visual inspection found no obvious landmark misalignment in these
  six samples. This is a sampled check, not a full-dataset guarantee.

Implemented `datasets/celeba_predicted.py`:
- Reuses official splits, labels, attribute order, and shared preprocessing.
- Returns image, labels, normalized predicted points, and filename.
- Raises LandmarkUnavailableError for failed detections.
- Inherited GT annotations are loaded but are not used by __getitem__.
- Final training-time failure handling remains pending.

Mac synthetic-data smoke check:
- Sample membership and order: PASS.
- Baseline image and label equality: PASS.
- Predicted coordinates used instead of GT: PASS.
- Output independent of inherited GT: PASS.
- Explicit failed-detection handling: PASS.
- Successful-sample DataLoader shapes: PASS.

Pending: real Dataset/DataLoader/model integration check.
No formal geometry-model training has started.

### Day 10 — Real integration completed and archived

Date: 2026-10-03
Verified source: 41d36c4de3ea61e3e2f0fe8e93cc5fb72eebeff7.

- Sampled baseline/predicted Dataset images and labels matched.
- Train and validation batches passed with two DataLoader workers.
- Logits: (3, 24); finite outputs and BCE.
- Untrained eval-mode BCE: train 0.766356; validation 0.942679.
- Real no_face sample 000199.jpg raised LandmarkUnavailableError.
- No optimizer updates or formal training were performed.
- Drive archive: runs/m3_preparation/day10/integration_report.json,
  sampled_landmark_alignment.png, and source/.
- Notebook: notebooks/day10_predicted_geometry_integration.ipynb.
- Temporary Colab raw data contains only seven images.
- Next: failure policy, baseline source compatibility, full-data
  preparation, and small-sample training checks.

## Day 11 — Geometry failure masking

Date: 2026-10-03

Implemented explicit missing-geometry handling:
- Dataset failure_policy="error" preserves the Day 10 four-item
  interface and raises on failed detections.
- Dataset failure_policy="mask" retains all samples and returns
  image, labels, normalized points, filename, geometry_valid.
- Failed detections use zero normalized coordinate placeholders
  and geometry_valid=False. No GT substitution is performed.
- Model accepts an optional boolean geometry_valid tensor [B].
- Invalid coordinates are replaced with zeros before the MLP.
- Invalid geometry features are zeroed after the MLP, including biases.
- Missing mask means all geometry is valid.
- Parameter count remains 11,191,000.
- Cache integrity errors remain errors.

Mac model checks:
- Mixed valid/invalid geometry features: PASS.
- Invalid-coordinate changes and NaN isolation: PASS.
- Omitted mask equals all-valid mask: PASS.
- Mixed-batch geometry gradients: PASS.
- All-invalid geometry gradients: zero.
- All-invalid image/classifier gradients: finite and nonzero.

Mac synthetic Dataset checks:
- Default strict behavior preserved: PASS.
- Failed sample retained with zero placeholders: PASS.
- Both samples preserve baseline images and labels: PASS.
- Successful and failed geometry independent of GT: PASS.
- DataLoader boolean validity mask [True, False]: PASS.
- Mixed Dataset batch to model: PASS.

No optimizer updates or formal training were performed.
Pending: baseline source compatibility, real mixed-sample checks,
and small-sample overfit training in Colab.

### Day 11 — Overfit passed; formal seed-0 training pending

Date: 2026-10-03
Verified source: a6722d6453e6106fdf07c871b5ea9ea02743018b.

- Baseline saved-source hashes verified.
- Shared computational source and attribute order matched baselines.
- Eight real training images: seven successful detections, one no_face.
- ImageNet-pretrained M3; Adam; smoke learning rate 0.001; FP32.
- Completed updates: 10.
- Final eval BCE: 0.017194.
- Exact match: 100%; failed sample exact match: True.
- This is an overfit check on training samples, not generalization.
- Reports: docs/results/m3_preparation/day11/.
- Drive artifacts: runs/m3_preparation/day11/.
- overfit_smoke.pt must not initialize the formal experiment.
- Day 11 remains in progress: full-data preparation and formal M3
  seed-0 training have not started.
- Formal training must restart from ImageNet weights, using the
  established baseline settings, including learning rate 0.0001.

## Day 11 — M3 seed-0 formal training and verification

- Source commit: 44e8ac21946d1db1146d1efdbdba36482b3ecfad.
- Model: ResNet18 with predicted five-point geometry.
- Completed all 10 formal training epochs with seed 0.
- Training samples per epoch: 162,770.
- Validation samples per epoch: 19,867.
- Missing geometry retained through feature masking:
  249 training samples and 32 validation samples.
- Best checkpoint selected by validation mAP: epoch_02.pt.
- Best validation BCE: 0.1881912535008375.
- Best validation macro-F1: 0.7439260119349894.
- Best validation mAP: 0.8185643813256119.
- F1 threshold: 0.5.
- Fresh-model checkpoint verification with num_workers=0: PASS.
- Recomputed BCE, macro-F1, and mAP exactly matched saved values.
- Later epochs showed decreasing training BCE and worsening validation
  performance; the epoch-2 checkpoint remains selected.
- Multiprocessing DataLoader cleanup exceptions appeared during epoch 4.
  Subsequent epochs completed, and independent verification passed.
- This is a single-seed result, not evidence of a stable geometry benefit.
- No test evaluation was performed.

Artifacts:
- Drive: runs/m3_seed0/, including checkpoints, source snapshot,
  configuration, history, best selection, verification report,
  and validation_predictions.npz.
- Git reports: docs/results/m3_seed0/.
- Notebook: notebooks/day11_m3_seed0.ipynb.

Next:
- Per-attribute validation comparison with the RGB baselines.
- Oracle geometry and planned control/multiple-seed experiments.
- Keep the test split untouched until the evaluation protocol is finalized.

## Day 12 — Clean validation comparison and branch checks

- R50 epoch_03.pt reloaded using hash-verified training source.
- All 19,867 validation image hashes verified against the cache.
- R50 checkpoint verification: PASS within tolerance 1e-6.
- M3 epoch_00.pt versus epoch_02.pt parameter comparison: PASS.
- Image encoder, geometry encoder, classifier, and classifier connections
  to both branches have parameter updates.
- Parameter updates alone do not prove a geometry performance benefit.
- R18, R50, and M3 predictions have identical official validation
  filenames, targets, and attribute order.
- Shared training protocol fields checked.
- Full validation set retained; F1 threshold fixed at 0.5.

| Model | BCE | Macro-F1 | mAP |
|---|---:|---:|---:|
| R18 | 0.190717 | 0.727259 | 0.817737 |
| R50 | 0.190218 | 0.731168 | 0.818835 |
| M3 | 0.188191 | 0.743926 | 0.818564 |

- M3 minus R18 mAP: +0.082783 percentage points.
- M3 minus R50 mAP: -0.027029 percentage points.
- Per-attribute AP and F1 comparisons saved for all 24 attributes.
- Improvements are mixed across attributes.
- Higher fixed-threshold F1 does not necessarily imply higher AP.
- Single-seed, validation-only findings; stable gains remain unproven.
- No training updates or test evaluation performed during Day 12.

Artifacts:
- Drive comparison reports: runs/comparisons/day12/.
- R50 predictions and verification: runs/r50_seed0/day12/.
- Git reports: docs/results/comparisons/day12/.
- Notebook: notebooks/day12_clean_validation_comparison.ipynb.

Next: oracle geometry and planned control/multiple-seed experiments.

## Day 13 — Training protocol and three-model reload audit

- Shared seed-0 training configurations: identical for audited fields.
- All three runs completed 10 epochs.
- Saved checkpoint selection matches maximum validation mAP.
- Selected epochs: R18=3, R50=3, M3=2.
- No maximum-score ties occurred.
- Fresh-model strict loading from hash-verified saved source: PASS.
- Four real validation images used, including one failed detection.
- Identical image tensors across the three models: PASS.
- Reload logits matched saved predictions within recorded tolerances.
- No optimizer updates, new full-validation run, or test evaluation.
- Rules documented in docs/training_protocol_v1.md.
- Reports: docs/results/protocol_audit/day13/.
- Notebook: notebooks/day13_training_protocol_and_reload_audit.ipynb.
- Drive reports: runs/protocol_audit/day13/.

## Day 14 — Corruption implementation and validation previews

- Implemented blur, brightness, rotation, occlusion, and JPEG corruption.
- Four candidate severity levels per corruption; configuration remains draft.
- Preserved existing preprocessing; clean tensor and coordinate equality: PASS.
- Synthetic deterministic generation and rotation marker alignment: PASS.
- Generated 80 corrupted images from four fixed official validation samples.
- Real-sample clean equivalence, repeatability, and saved PNG pixel checks: PASS.
- Two uploaded montages reviewed; no obvious visual implementation issues.
- Shared generation is ready; actual multi-model corrupted inference is pending.
- Real-face GT overlays and corrupted-image MTCNN integration remain pending.
- No training, model evaluation, or test-image evaluation performed.
- Reports: docs/results/corruption_preparation/day14/.
- Implementation notes: docs/day14_implementation_draft.md.
- Local preview images remain under runs/; do not add CelebA images to Git.

## Day 15 — Real-image GT overlays and rotation audit

- Checked 100 conditions across four fixed validation images.
- Saved image reconstruction and independent GT transformation checks: PASS.
- Coordinate-formula difference below 1e-8 pixels; no GT points out of frame.
- Two real-image montages visually reviewed, covering both rotation directions.
- No obvious transform-induced landmark misalignment in reviewed images.
- GT remains available beneath occlusion; this is privileged Oracle information.
- User comments preserved in benchmark_preprocessing.py and corruptions.py.
- Documentation differences accepted only after matching the Day 14 source
  hashes and comparing computational ASTs; current hashes recorded.
- Report: docs/results/corruption_preparation/day15/gt_overlay_audit.json.
- Script: scripts/day15_gt_overlay_audit.py.
- Overlay images remain under runs/.
- No detector execution, model inference, training, or test evaluation.
- Next: Day 16 detector status, NME, coverage, and small end-to-end checks.

<!-- day16-pipeline-smoke -->
## Day 16 — Detector quality and M3 pipeline smoke

- Used four fixed official-validation images: 162771.jpg, 169393.jpg,
  176015.jpg, and 182637.jpg.
- Checked 84 unique inputs: four clean inputs and 80 corrupted inputs.
- Verified input image hashes and packaged source hashes.
- Ran fresh MTCNN detection on each final 224x224 RGB canvas.
- Detection: 82 valid, two no_face, zero runtime errors.
  Both failures were 182637.jpg under occlusion severity 3 and 4.
- Clean mean NME: 4.42%; NME uses successful detections only.
- Strictly loaded M3 epoch_02.pt with its verified training source.
- All 84 inputs produced finite logits with shape [84, 24].
- Retained both failed samples; their geometry features were exactly zero.
- Changing invalid coordinates to large finite values or NaN changed
  logits by 0.0.
- Reports: docs/results/corruption_preparation/day16/.
- Notebook: notebooks/day16_detector_quality_and_pipeline_smoke.ipynb.
- This is a four-image validation pipeline smoke, not an attribute
  performance evaluation. Benchmark remains unfrozen; test not evaluated.
- CelebA images and the input-image ZIP remain under runs/ outside Git.
