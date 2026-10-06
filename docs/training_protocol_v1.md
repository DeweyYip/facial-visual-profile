# Training Protocol v1

## Scope

This document records the shared seed-0 protocol used by the completed
R18, R50, and predicted-geometry M3 experiments. The configuration and
checkpoint selection records were audited during Day 13.

## Data and inputs

- CelebA official splits: train 162,770; validation 19,867; test 19,962.
- Use the same selected 24 attributes in the recorded order.
- Use the established shared image and coordinate preprocessing.
- Clean validation uses no added corruption.
- M3 uses cached MTCNN predicted landmarks.
- M3 retains failed detections and masks their geometry features to zero.
- Do not replace failed predicted landmarks with GT coordinates.
- Test evaluation has not been performed.

## Shared training configuration

| Setting | Recorded value |
|---|---|
| seed | 0 |
| epochs | 10 |
| batch_size | 64 |
| num_workers | 2 |
| optimizer | Adam |
| learning_rate | 0.0001 |
| weight_decay | 0.0 |
| precision | FP32 |
| augmentation | None; use existing shared preprocessing |
| f1_threshold | 0.5 |
| epoch_seed_rule | seed + epoch; epochs start at 1 |
| cudnn_deterministic | True |
| cudnn_benchmark | False |

## Model initialization

- R18: ResNet18_Weights.IMAGENET1K_V1.
- R50: ResNet50_Weights.IMAGENET1K_V1.
- M3: ResNet18_Weights.IMAGENET1K_V1 for the image backbone;
  newly initialized geometry MLP and fusion classifier.
- Documented training uses BCEWithLogitsLoss and trains all parameters.
- Verification loads trained checkpoints with pretrained=False.

## Checkpoint selection

- Complete the planned 10 epochs.
- Select the maximum validation mAP.
- For exactly equal maximum scores, select the earliest epoch.
- Do not select by training loss or macro-F1.
- Report macro-F1 at the fixed probability threshold 0.5.
- Threshold tuning is a separate analysis.
- All three saved selections match their full training histories.
- None of these histories had a tie at the maximum score;
  historical tie-handling execution was therefore not tested.

| Run | Selected epoch | Checkpoint | Validation mAP |
|---|---:|---|---:|
| r18_seed0 | 3 | epoch_03.pt | 0.817736554 |
| r50_seed0 | 3 | epoch_03.pt | 0.818834660 |
| m3_seed0 | 2 | epoch_02.pt | 0.818564381 |

## Reload verification

Fresh models were constructed from hash-verified saved source snapshots.
Best-checkpoint weights were loaded strictly and used for inference on
four identical validation images, including one failed MTCNN detection.
Outputs matched saved validation logits within atol=1e-4 and rtol=1e-4.

| Run | Maximum absolute logit difference | Result |
|---|---:|---|
| r18_seed0 | 0.00000858 | PASS |
| r50_seed0 | 0.00000572 | PASS |
| m3_seed0 | 0.00000811 | PASS |

This is a four-sample reload check. Full-validation evidence comes from
the earlier evaluation and verification artifacts.

## Evidence and limitations

- Reports: docs/results/protocol_audit/day13/.
- Notebook: notebooks/day13_training_protocol_and_reload_audit.ipynb.
- Drive: runs/protocol_audit/day13/.
- Source commits and record hashes are recorded in the audit JSON.
- Configuration agreement does not imply identical software environments.
- Single-seed validation results do not establish stable geometry gains.
- Keep future protocol changes explicit and versioned.
