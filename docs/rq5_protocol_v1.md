# RQ5 protocol and implementation audit

Status: implementation candidate; no formal training or empirical improvement claimed.

## Scope

N0/N1/MTL, seeds 0/1/2, nine runs maximum. No additional Oracle, R50,
augmentation, feature masking, architecture search or lambda sweep.

N0: R18(512) + fixed zeros(32), Linear(544,128), ReLU, Linear(128,24).
N1: R18(512) + predicted geometry MLP(10,32,32), same nonlinear head.
MTL: R18(512), Linear(512,24) and Linear(512,10); mean BCE + lambda * mean MSE.
GT enters the MTL auxiliary loss only. Attribute model forward takes RGB only.

## Inherited settings verified against archived RQ4 notebook

- Adam, LR 0.0001, WD 0, FP32, train batch 64, eval batch 128, workers 2.
- ImageNet1K V1 initialization; all parameters fine-tuned.
- Max 10 epochs, minimum 4, patience 2, strict clean-validation mAP improvement.
- Exact ties keep the earlier checkpoint. Epoch RNG seed = nominal seed + epoch.
- Identical worker/generator initialization and deterministic cuDNN settings.
- Clean-only inputs, original 24 attributes, official train/valid/test partitions.
- Pure RQ4 canvas/corruption/metric definitions extracted into rq5_shared.py.
- Validation thresholds: 0.05..0.95, step 0.01; ties within 1e-12 prefer
  closest to 0.5, then lower, matching actual RQ4 code.
- Clean excluded from the equal-weight 20-condition corruption average.
- Sample SD is descriptive, not a confidence interval.

The source notebook checksum is in rq5_source_audit.json. The historical
training_protocol_v1.md refers to earlier fixed-ten-epoch runs, not RQ4.
The RQ4 notebook's old 'not executed' prose is historical; current archived
RQ4 results already exist. Do not retrain A or overwrite RQ4.

## Input chains

N1 train/valid reads the existing MTCNNLandmarkCache reader and original-image
SQLite cache, then applies the shared pixel-center coordinate map.
N1 test reuses original day19/day20 ZIP shard records, verifies payload hashes
and final RGB hashes; failed detections stay in all attribute metrics and
geometry embeddings are masked after the MLP. Missing cache/hash disagreement
is an error, never a reason to fall back to GT or rerun a detector silently.
No facenet import or detector inference is needed by this package.

MTL NME is five-point mean Euclidean error / GT inter-eye distance, reported
as a ratio; multiply by 100 to display percent. GT transforms with rotations.

## Required local verification before official training

1. Run scripts/check_rq5.py in the existing project environment.
2. Run the real-data checks described in RQ5_LOCAL_README.md, including
   final-canvas cache/hash parity and historical preprocessing parity.
3. Compare actual RQ4 A configs, histories and thresholds for all three seeds.
   The connected Drive summary and A_seed0 folder were inspected, but this
   package does not certify every historical checkpoint byte or environment.
4. Perform the training-only four-batch loss-scale check. Freeze ONE lambda
   with its basis before any official run; no full trial or grid search.
5. Record environment and confirm the current Pillow/libjpeg outputs match
   frozen RGB hashes. This is not guaranteed by matching source code alone.

## Persistence and comparison

New outputs go to runs/rq5_training_v1/{model}_seed{seed}. Checkpoints include
optimizer state and best state; latest.pt commits an epoch. Interrupted epochs
repeat using the original reseed scheme. Do not use concurrent writers.
Config, source hashes, dependency versions and thresholds are bound to runs.

Report N1-N0, MTL-A and auxiliary N0-A per seed, for clean and corrupted
scores, with mean and sample SD. Old A is reused; environment/implementation
differences limit strict causal interpretation. The report script checks
completeness, not full historical protocol equivalence.

This environment lacks torch/torchvision. Python parsing/compilation and
source equivalence checks were performed here; model/data/CUDA checks must
run locally. No runtime PASS, frozen lambda, or training completion is claimed.

## Local verification update

Local Mac checks passed for all three synthetic model modes, failure masking,
MTL shared-backbone gradients and model/Adam continuation. Preprocessing
compatibility passed on one deterministic synthetic image across 21 conditions.
N0/MTL real training samples and batch-size-2 forward checks also passed.

N1 real-cache checks, frozen test-input verification, CUDA execution and MTL
lambda freezing remain pending. No formal RQ5 training has started.
This update supersedes earlier statements that local dynamic checks were pending.
