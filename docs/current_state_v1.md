# Facial Visual Profile — Current State v1

## Latest verified state

Day 16 completed and pushed as 09b41c51d43ddd63d76db10cdfea8bd604029030.
Day 17 historical clean-validation thresholds were reproduced/fitted and
archived locally. Benchmark protocol/manifest freeze is the next execution
step; it is not complete merely because this document has been installed.
Use the Day 17 freeze audit to confirm completion after running the script.

## Goal and models

Research whether explicit predicted five-point geometry improves facial
attribute prediction and robustness compared with image-only depth.
The project is an engineering/research portfolio, not evidence of established
geometry gains. Demo/language features remain later work.

| Model | Architecture | Seed-0 status |
|---|---|---|
| R18 | ImageNet ResNet18; 512 -> 24 | 10 epochs complete; epoch 3 selected |
| R50 | ImageNet ResNet50; 2048 -> 24 | 10 epochs complete; epoch 3 selected |
| M3 Predicted Geometry | R18 512 + MLP 10 -> 32 -> 32; concat 544 -> 24 | 10 epochs complete; epoch 2 selected |
| Oracle Geometry | Same fusion design, independently trained using GT | Pending |

Outputs are logits; BCEWithLogitsLoss trains all model parameters.
MTCNN is a fixed pretrained detector, not jointly trained with M3.
Failure policy="mask" retains samples, replaces missing coordinate inputs
with zeros and zeroes geometry features after the MLP. No GT substitution.
Optional strict Dataset failure_policy="error" still raises on failures.

## Dataset, attributes and preprocessing

CelebA aligned RGB images and matching aligned five-point annotations.
Official counts: train 162,770; validation 19,867; test 19,962.
Validation IDs: 162771.jpg through 182637.jpg. Use official partition metadata
as the membership authority. Targets convert -1/1 to 0/1.

Mandatory 24-attribute order:

```text
Smiling, Mouth_Slightly_Open, Eyeglasses, Bangs, Wearing_Hat, Mustache,
No_Beard, Black_Hair, Bushy_Eyebrows, Arched_Eyebrows, High_Cheekbones,
Bags_Under_Eyes, Big_Lips, Big_Nose, Double_Chin, Oval_Face,
Receding_Hairline, Blond_Hair, Brown_Hair, Gray_Hair, Bald,
Straight_Hair, Wavy_Hair, Sideburns
```

Preserve aspect ratio, resize longest side to 224, centered black letterbox,
PIL BILINEAR. Pixel-center mapping:
`(coord + 0.5) * resized_dimension / original_dimension - 0.5 + padding`.
Normalize coordinates `2*coord/223-1`, flatten five (x,y) pairs to 10 values.
Point order: lefteye, righteye, nose, leftmouth, rightmouth.
ImageNet RGB mean (.485,.456,.406), std (.229,.224,.225).

## Training and clean-validation evidence

Seed 0, 10 epochs, batch 64, Adam lr 0.0001, weight decay 0, FP32,
two workers, no augmentation, all parameters trainable. Select highest
validation mAP; exact ties choose earliest epoch. No checkpoint reselection.

| Model | BCE | F1 at 0.5 | Validation-selected F1 | mAP |
|---|---:|---:|---:|---:|
| R18 | 0.190717 | 0.727259 | 0.762372 | 0.817737 |
| R50 | 0.190218 | 0.731168 | 0.763569 | 0.818835 |
| M3 | 0.188191 | 0.743926 | 0.764527 | 0.818564 |

All three have complete, verified clean-validation predictions on identical
filenames, targets and attribute order. M3 retains 32 validation failures.
Fitted F1 uses the same validation data for fitting and scoring; it is not an
independent estimate. One seed and small score differences do not establish
stable gains. Do not compare tuned F1 with fixed-0.5 F1 without labeling.

Thresholds: configs/thresholds/{r18_seed0,r50_seed0,m3_seed0}.json.
Grid 0.05..0.95, step 0.01, maximize per-attribute F1; ties closest to 0.5,
then lower. R18 existing thresholds and checkpoint SHA reproduced exactly;
R50/M3 fitted from saved clean logits. Do not rerun threshold fitting merely
because corruption is introduced. Reuse each checkpoint's threshold vector.

Drive root: /content/drive/MyDrive/FacialVisualProfile.
- runs/r18_seed0/epoch_03.pt; day7/validation_predictions.npz.
- runs/r50_seed0/epoch_03.pt; day12/validation_predictions.npz.
- runs/m3_seed0/epoch_02.pt; validation_predictions.npz.
- runs/benchmark_preparation/day17/: threshold reproduction and new JSONs.
- Historical clean cache: cache/mtcnn_clean.sqlite.

## Verified development milestones

- Days 1-3: environment, data inventory, official Dataset, shared preprocessing.
- Day 4: R18 eight-image overfit and one-epoch T4 benchmark passed.
- Days 5-7: clean MTCNN cache, R18 10-epoch training, full validation
  reproduction and per-attribute threshold fitting.
- Day 8: R50 10 epochs with optimizer recovery; epoch 3 selected.
- Days 9-10: geometry model and predicted Dataset/cache integration passed.
- Day 11: failure masking, gradient/NaN checks, eight-image overfit, formal
  M3 10-epoch training and full validation reproduction passed.
- Day 12: R50 full validation reproduction, M3 branch update checks, and
  aligned three-model clean-validation comparison passed.
- Day 13: shared training protocol and four-sample strict reload audit passed.
- Day 14: five corruptions, four levels each; exact clean preprocessing
  equality and synthetic checks; four real validation previews generated.
- Day 15: 100 image/GT conditions checked, rotation alignment passed;
  two montages manually reviewed; comments preserved by AST equivalence.
- Day 16: 84 inputs from four validation images, 82 valid detections and two
  no_face, zero runtime errors; all M3 outputs finite [84,24]; failed geometry
  features exactly zero; invalid coordinate changes affected logits by 0.0.
- Day 17: historical clean thresholds reproduced/added and archived; rule
  freeze execution follows installation of the corrected bundle.

## Corruption benchmark and pipeline distinction

After letterbox and before normalization: blur sigma 1/2/3/4; brightness
.8/.6/.4/.2; rotation +/-5/10/15/20 degrees; lower-face occlusion side
fractions .1/.2/.3/.4; JPEG qualities 75/50/25/10. One corruption at a time.
Rotate GT with image; other corruptions preserve GT even under occlusion.
Oracle is privileged input. Preserve VERSION=day14_draft_v1 because it is
part of the deterministic rotation-sign hash; renaming changes directions.

Historical training/validation geometry: detect original aligned RGB images,
then map points with shared preprocessing. New benchmark geometry: detect
the final 224x224 RGB canvas for BOTH clean and corrupted conditions.
Keep historical thresholds fixed and report this inference pipeline change.
Generate a matching new clean baseline; do not use historical clean metrics
as the baseline for new-pipeline corruption degradation.

Planned main scope: full official test, 19,962 IDs crossed with 21 conditions
(one clean + 20 corrupted), 419,202 planned inputs. Factorized sample CSV
and condition config fix all pairs and rotation signs. Not yet generated
merely by manifest creation. All models/seeds share identical RGB inputs.
Report F1/AP per attribute, macro F1/mAP, sample counts, detection coverage,
failure statuses and successful-only NME/counts. Retain failures in attribute
metrics. Exceptions are execution errors, not ordinary no_face detections.

Earlier MTCNN cache includes test detections (19,950 ok / 12 no_face).
Therefore test images HAVE been accessed for clean detection. Archived
evidence reports no test attribute evaluation or test NME evaluation.
Do not claim test images were never accessed. Do not tune using test scores.

## Important files and current next steps

- configs/attributes.yaml; configs/thresholds/: label order and saved decisions.
- datasets/preprocessing.py: historical shared preprocessing, unchanged.
- datasets/benchmark_preprocessing.py, corruptions.py: reviewed transforms.
- datasets/celeba_predicted.py, landmark_cache.py: historical predicted inputs.
- detectors/mtcnn.py: frozen detector/selection wrapper.
- evaluation/attributes.py: unchanged historical scalar-threshold metrics.
- evaluation/benchmark_attributes.py: supplied-vector benchmark metrics only.
- docs/training_protocol_v1.md: historical training rules.
- docs/benchmark_protocol_v1.md: corrected formal benchmark rules.
- scripts/day17_freeze_benchmark.py: bind metadata, thresholds and source hashes.
- docs/results/: small provenance artifacts; experiment_log.md: chronology.

Next: run corrected Day 17 freeze; verify audit and commit explicit files.
Then independently train/select Oracle and fit its clean-validation thresholds;
prepare final-canvas clean/corrupted detector caches and RGB provenance;
evaluate selected models under frozen rules; additional seeds/controls follow.
No full corrupted attribute evaluation, Oracle training or stable-gain claim
has been completed in the currently verified record.

## Handoff discipline

Mac: /Users/deweye/Desktop/Facial Visual Profile.
GitHub: DeweyYip/facial-visual-profile. Drive checkpoints/source snapshots
are authoritative; preserve mixed-source experiment provenance and hashes.
Colab /content is temporary. Do not restart completed training.
Preserve user comments and preexisting worktree changes, especially
models/resnet50.py and older notebooks. Do not git add everything.
Never commit CelebA images, caches or weights. Keep them under data/runs/Drive.
Before proposing work, check logs/artifacts for prior completion. Distinguish
unconfirmed from unfinished. Explain each block, location, inputs, outputs,
expected runtime and whether training starts; proceed one logical step at a time.
English code/docs/comments, Chinese user explanations. Validate evidence
before replacing old state; do not infer execution from schedules or filenames.

<!-- day17-benchmark-freeze -->
## Day 17 — Benchmark protocol v1 frozen

- Official test manifest: 19,962 unique IDs; 21 conditions; 419,202 planned inputs.
- CSV fixes test IDs/order and deterministic rotation signs; config fixes conditions.
- Saved R18/R50/M3 thresholds are hash-bound; grid 0.05..0.95;
  closest to 0.50 then smaller on ties. Existing R18 values reproduced.
- Also report fixed-0.5 F1; preserve historical training records.
- M3 benchmark clean baseline uses fresh final-canvas detection, matching
  corruption. Historical thresholds remain fixed; pipeline change is explicit.
- M3 retains failed detections with masked geometry; no GT substitution or deletion.
- This freeze accessed partition metadata only. Earlier clean MTCNN test cache
  exists; test attribute evaluation remains pending.
- Evidence: docs/results/benchmark_freeze/day17/freeze_audit.json.
- Rules: docs/benchmark_protocol_v1.md; configs/benchmark_v1.json.
- Next: independent Oracle training/validation thresholds, final-canvas landmark
  caches, then evaluation under the frozen protocol.
