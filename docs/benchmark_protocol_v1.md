# Benchmark protocol v1

## Freeze scope

Day 17 fixes sample selection, conditions, metrics and threshold-selection
rules. R18/R50/M3 historical clean-validation thresholds have been verified and saved.
Oracle training and thresholds, new final-canvas benchmark caches, and test
attribute evaluation remain pending. Historical training records are preserved.

## Samples and factorized manifest

Use all 19,962 official CelebA test images, in ascending filename order.
No subset or sampling by label, detector status, or outcome is permitted.
The archived test_samples.csv records filename, official split ID (2), and
rotation sign. The 21 ordered conditions in benchmark_v1.json are crossed
with every row: 419,202 planned inputs. This factorized manifest specifies
every sample-condition pair without duplicating the same sample 21 times.
Validate all 202,599 partition IDs, uniqueness and official split counts.
Read partition metadata during freeze; do not open test images or test
attribute/landmark annotations, run test detection, or compute test metrics.
An earlier original-image clean MTCNN cache already includes test images.
This does not constitute test attribute evaluation, but it must be disclosed.
The freeze script accesses partition metadata only; do not claim test images
have never been accessed. Archived records report no test attribute evaluation.

## Image and geometry pipeline

Use shared aspect-preserving 224x224 letterbox and pixel-center mapping.
Apply one corruption to the RGB canvas before ImageNet normalization.
Clean is a single condition implemented via blur severity 0 (identity).
Preserve corruption implementation VERSION=day14_draft_v1: its identifier
participates in random direction selection and must not be renamed.
Freeze candidate levels and all rendering settings from the reviewed draft.
Direction is +1 when the first SHA256 byte of
`day14_draft_v1|0|<filename>|rotation` has its low bit set, otherwise -1;
one direction per filename across all four rotation severities.
Rotate GT with the image, without swapping point labels or clipping points.
Other corruptions leave GT unchanged, including occlusion.
Record out-of-frame GT occurrences rather than silently removing samples.
Oracle receives transformed GT even under occlusion and is privileged input.

All models and seeds share identical final RGB inputs. Future cache generation
must verify the planned manifest and record actual parameters, RGB hashes,
GT transformations, detector statuses, environment and detector-weight hash.
Fresh clean detection uses the same final 224x224 canvas pipeline as corruption.
Do not substitute the historical original-image landmark cache for this clean
benchmark. Freeze detector configuration, highest-probability face selection,
point order, and implementation source through the Day 16 evidence.

## Checkpoints and thresholds

Retain clean-validation mAP selection with earliest-epoch exact-tie handling.
Seed-0 selected checkpoints: r18_seed0/epoch_03.pt,
r50_seed0/epoch_03.pt, m3_seed0/epoch_02.pt. Oracle is independently trained
under the shared clean training protocol and selected on clean validation.
Never choose a checkpoint, seed, threshold or condition using test outcomes.

Reuse configs/thresholds/r18_seed0.json, r50_seed0.json and m3_seed0.json.
R18 was fitted earlier and independently reproduced in Day 17; R50/M3 were
fitted from already verified full clean-validation predictions in Day 17.
All use 19,867 samples, grid 0.05..0.95 in steps of 0.01, maximizing each
attribute's F1. Equal scores prefer closest to 0.50, then the smaller value.
Comparison is probability >= threshold. Bind checkpoint SHA256, threshold
file SHA256 and the 24-attribute order. Verify these before evaluation.
No threshold refitting is permitted per corruption, severity, detector status,
new final-canvas pipeline or test outcomes. For Oracle, fit thresholds once on
clean validation using the same historical rule after independent training;
seal its checkpoint/threshold provenance before Oracle test evaluation.

M3 thresholds were fitted using the historical original-image landmark cache
and subsequent mapping. The formal benchmark uses fresh detection on the
final 224x224 canvas for BOTH clean and corruption, keeping those historical
thresholds fixed. This intentionally measures the trained model with a changed
inference pipeline; describe that change explicitly. Report historical clean
validation separately and do not treat its score as the benchmark clean
baseline. A robustness degradation compares matching benchmark pipelines.

Formal reports include validation-selected-threshold F1 and fixed-0.5 F1,
clearly named. AP/mAP do not depend on classification thresholds.
Validation-selected F1 is an in-sample tuning score, not independent validation
performance. Preserve historical fixed-0.5 training numbers. Do not relabel
four-image inference smoke checks as attribute-performance evaluation.

## Metrics and failure handling

Report each condition separately, with sample count, 24 F1/AP values,
macro F1 (unweighted mean over 24 attributes), mAP (unweighted mean AP),
and positive counts. Use sklearn average_precision_score (non-interpolated
AP); do not substitute trapezoidal PR AUC. Use zero_division=0 for F1;
require both label classes per attribute or stop and report the issue.
For degradation, compare clean and corrupted on the exact same samples;
report clean minus corrupted score in percentage points. Never average clean
into a corruption aggregate. Optional aggregate: equal mean over 20 corrupted
conditions, explicitly named; preserve all individual condition results.

Detection success is finite valid-shaped output from the frozen wrapper,
not an NME threshold or a new probability cutoff. Retain no_face and
invalid_output samples, supply zero coordinate placeholders and a False mask,
and zero geometry features after the MLP. No GT substitution, sample deletion
or R18 fallback in the M3 benchmark. This masked M3 is not the R18 baseline.
Runtime exceptions are execution errors: record them and stop completeness
claims until resolved. Do not silently classify exceptions as no_face.

Coverage = valid / all planned samples; detection failure rate =
(no_face + invalid_output) / all planned samples. Report each status count.
NME is mean five-point Euclidean pixel error divided by GT eye distance,
using transformed final-canvas GT; successful detections only. Save ratios,
display percentages, report NME count, mean and median. Save failed NME as
null; if no detections succeed, mean/median are null. Attribute metrics still
include every sample, including failed detection cases.

## Reproduction and interpretation

Verify frozen file hashes before generating caches or evaluating. Record
actual CPU/GPU device, package versions, pretrained detector weights and
encoder/decoder environment: matching source does not guarantee identical
RGB pixels across Pillow/libjpeg builds. Bind caches to actual RGB hashes.
Use the Day 16 environment as the tested starting point, not evidence of GPU
compatibility. Do not infer robustness gains from the four-image smoke.
Report actual completed seeds; seed-0 alone is not a multi-seed result.
Test-inspired new experiments must be versioned and labeled exploratory.
