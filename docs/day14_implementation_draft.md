# Day 14 corruption implementation draft

This additive package preserves existing datasets/preprocessing.py. Copy only
the five new files into the existing repository; do not replace training files.

## Pipeline

Original RGB -> established aspect-preserving resize and black letterbox ->
224x224 RGB -> one corruption -> MTCNN on final RGB -> image normalization.
For 178x218 inputs, resized content is 183x224 with left padding of 20 pixels.
Coordinates use the established pixel-center resize mapping.

Severity zero is clean. Every nonzero severity starts from the same clean
canvas, never from the previous severity. No composed corruptions.

The five severity arrays follow page 9 of V3_formal(1).pdf as candidates.
Implementation details and corruption seed 0 are validation-preview choices,
not a frozen test benchmark. Review real validation previews before freezing.
The configuration records implementation constants; the smoke script checks
agreement. Editing JSON alone does not change runtime behavior.

## Geometry

Predicted landmarks must be freshly detected on final RGB or come from an
exactly matching corruption cache. Never use clean cached points as corrupted
detector predictions, and never pass normalized RGB to MTCNN.
Final-canvas predicted pixels go directly through normalize_points(); do not
apply the original-image resize mapping a second time.

GT points move with rotation and retain anatomical indices. No horizontal flip
is included. GT coordinates remain unchanged for other corruptions, including
occluded points. Out-of-frame GT flags are reported; coordinates are not clipped.
Failure masking remains the established M3 policy. No detector integration or
cache is implemented in this package.

Pillow rotation uses edge center (112,112), corresponding to pixel-index center
(111.5,111.5). Positive angles are counterclockwise; the forward GT matrix is
recorded for each sample. Rotation sign is fixed by SHA256 of version, seed,
sample_id and type; it is independent of severity and worker order.

## Verification and next step

Run from the Mac project root:

    python scripts/day14_corruption_smoke.py

This checks clean tensor/coordinate equality with the current prepare(), all
five corruptions, deterministic pixels, unchanged nongeometric GT, and synthetic
marker alignment for both rotation directions. It reads no dataset images and
uses no GPU. Its output stays under runs/ pending review.

Next: use fixed validation sample IDs to preview all 20 conditions, inspect
occlusion position and semantic visibility, then proceed to real landmark
overlays. Official test selection and evaluation remain pending protocol freeze.

JPEG decoded pixels may depend on Pillow/libjpeg versions. Record environment
versions and compare final RGB hashes before sharing cached detector results.
Existing clean MTCNN cache detects on original aligned images; new final-canvas
detector results represent a different input pipeline. Establish matching clean
final-canvas detector results before computing corrupted-vs-clean degradation.

## Delivery validation limits

The delivery environment lacks PyTorch/torchvision. Pure Pillow/NumPy corruption
checks were run locally; exact shared-preprocessing tensor equivalence must be
confirmed by the included smoke script in the project's Python environment.
