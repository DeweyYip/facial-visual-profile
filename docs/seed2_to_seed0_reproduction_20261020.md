# Seed 2 through seed-0 reproduction: development record

Status checked against GitHub and Google Drive on 2026-10-20.
This record supplements older project handoffs; it does not replace historical results.

## Completed work and current status

| Work | Status | Evidence in this repository |
|---|---|---|
| Seed 2: R18, R50, M3 and Oracle training | All four completed 10 epochs; selected-checkpoint validation and threshold freeze complete | `docs/results/training_runs/*_seed2/`; original training archive |
| Seed 1/2 frozen test benchmark | Complete: 19,962 test images, 21 conditions, eight models | `docs/results/benchmark/multiseed_seed1_seed2/` |
| Original seeds 0/1/2 comparison | Existing comparison tables and plots retained | `docs/results/benchmark/three_seed_comparison/` |
| Three-seed NME grouping and paired F1 intervals | Complete; fixed trained seeds, image-resampling uncertainty only | `docs/results/benchmark/three_seed_nme_and_f1_intervals/` |
| Prediction case review | Complete: 24 illustrative cases, 288 prediction rows | `docs/results/benchmark/prediction_case_review/` |
| Small-scale inference reproduction | Passed: seed 1, four models, eight test images, four conditions; 128 model-sample evaluations | `docs/results/benchmark/small_scale_reproduction/` |
| Seed 3 training | All four completed 10 epochs; validation export and threshold freeze complete | `docs/results/training_runs/*_seed3/` |
| Seed-0 reproduction v1 training | All four completed 10 epochs; validation export and threshold freeze complete | `docs/results/training_runs/*_seed0_reproduction_v1/` |
| Seed-0 reproduction + seed 3 frozen test | Complete: eight models, 21 conditions, 1,638 shards, 168 overall and 4,032 attribute rows | `docs/results/benchmark/seed0_reproduction_v1_seed3/` |
| Seeds 4 and 5 | Discussed, not established as completed | No completion claim |

The previously committed seed-2 training and seed-1/2 test notebook code matches
the latest uploaded copies. Existing benchmark, NME, case-review and small-scale
reproduction files were compared with Drive and left unchanged. Exact matches
and checked filenames are recorded in
`docs/results/training_archives/existing_git_evidence_audit_20261020.json`.

## Selected clean-validation results

All values below come from verified training archive summaries. Epoch is the
selected checkpoint, not the number of training epochs. Every run completed ten.
The tuned F1 is fitted and scored on the same clean validation set; it is not
an independent evaluation. mAP is threshold-independent.

| Run | Selected epoch | F1 at 0.5 | Clean-val tuned F1 | mAP |
|---|---:|---:|---:|---:|
| r18_seed2 | 2 | 0.736760 | 0.764140 | 0.819413 |
| r50_seed2 | 2 | 0.735019 | 0.764860 | 0.820903 |
| m3_seed2 | 3 | 0.739927 | 0.763674 | 0.818906 |
| oracle_seed2 | 2 | 0.729663 | 0.764707 | 0.818790 |
| r18_seed3 | 2 | 0.736905 | 0.765540 | 0.819117 |
| r50_seed3 | 2 | 0.737576 | 0.765078 | 0.820018 |
| m3_seed3 | 2 | 0.747494 | 0.764549 | 0.819445 |
| oracle_seed3 | 2 | 0.748037 | 0.765043 | 0.820148 |
| r18_seed0_reproduction_v1 | 3 | 0.726520 | 0.761972 | 0.816857 |
| r50_seed0_reproduction_v1 | 3 | 0.729621 | 0.764718 | 0.819710 |
| m3_seed0_reproduction_v1 | 2 | 0.743926 | 0.764527 | 0.818564 |
| oracle_seed0_reproduction_v1 | 2 | 0.744359 | 0.765625 | 0.818609 |

Full-precision values and selected checkpoint SHA256 values are in
`docs/results/training_archives/validation_summary_20261020.csv`.

## Training and evaluation decisions

- Preserve the established four-model method: ImageNet-pretrained R18/R50 RGB
  baselines; R18 plus predicted five-point geometry (M3); independently trained
  R18 plus aligned GT geometry (Oracle). Oracle is a reference, not a guaranteed
  performance upper bound.
- Training uses the official 162,770/19,867 train/validation splits, 24 attributes,
  ten epochs, batch 64, Adam lr 1e-4, weight decay zero, FP32, two workers,
  no augmentation and full fine-tuning. Highest clean-validation mAP selects the
  checkpoint; exact ties retain the earliest epoch.
- Per-attribute thresholds use clean validation only, grid 0.05 through 0.95,
  step 0.01, maximum F1, ties closest to 0.5 then lower; decisions use >=.
  Keep frozen-threshold F1, F1 at 0.5 and mAP separately labeled.
- Original seed-0 RGB training recorded cu128; the later seed-1/2 environment
  checks and reproduction use cu130. Original M3 seed 0 used cu130. Recorded
  environment differences do not establish the cause of a robustness difference.
  Consult each archived config; do not describe all historical runs as an
  identical execution environment.
- Original seed 0 is retained. The new experiment uses random seed 0 with the
  independent tag `seed0_reproduction_v1`; it must not resume or overwrite the
  original `*_seed0` directories. It is a repeat of seed 0, not another
  independent training seed.
- Seed 0 showed different M3-minus-R18 robustness patterns from seeds 1/2,
  especially rotation and strong blur. Stable geometry gains are not established.
  No causal explanation or invalidation of an original run has been established.
- M3 clean-validation F1 and mAP reproduce the original seed-0 values at displayed
  precision. This does not prove identical weights or corrupted-test behavior.
- DataLoader subprocess cleanup exceptions appeared during the reproduction.
  The run proceeded through all sample-count, history, selected-validation replay
  and archive checks. Record the warnings; do not retroactively change workers
  or retrain solely to suppress their output.

## Files and reproducibility

- `notebooks/multiseed_training_and_validation_seed2.ipynb`: existing seed-2 runner.
- `notebooks/multiseed_training_and_validation_seed3.ipynb`: cleaned seed-3 replay
  recipe derived from the uploaded seed-2 code by changing only SEED to 3.
  It is not the captured executed seed-3 notebook; completed evidence is archived.
- `notebooks/seed0_reproduction_v1.ipynb`: seed-0 rerun with isolated run,
  workspace, threshold-export and archive names; historical reference paths kept.
- `notebooks/frozen_test_seed0_reproduction_v1_seed3.ipynb`: eight-model test
  runner with explicit run-tag mapping, isolated result directory and resumable
  checksum-bound shards. Original seed-0 metrics have separate comparison labels.
- `docs/results/training_runs/<run>/`: config, complete history, best checkpoint
  summary, data preparation and selected-validation export audit.
- `configs/thresholds/<run>.json`: the exact frozen threshold report for that run.
- `docs/results/training_archives/`: original small training ZIPs, including
  byte-exact source snapshots, plus import audit and validation summary CSV.
  Original bytes remain intact so archive manifests remain verifiable.

The new runner preserves the frozen 21-condition corruption protocol, batch 16,
shard 256, FP32 and verified original detector coordinates/masks. New model
logits are computed for each new shard; old RGB-model predictions are not
substituted for the selected new runs. M3 keeps detector failures with masked
geometry and no GT fallback. No test threshold fitting is performed.

## Drive locations and next actions

Root: `/content/drive/MyDrive/FacialVisualProfile/`.

- Weights and validation NPZ: `runs/<model>_seed2/`, `runs/<model>_seed3/`,
  `runs/<model>_seed0_reproduction_v1/`.
- Threshold exports: `runs/benchmark_preparation/seed2/`, `seed3/`,
  `seed0_reproduction_v1/`.
- Training exports: `runs/training_archives/<tag>_training_results_archive.zip`.
- New frozen test: `runs/benchmark/seed0_reproduction_v1_seed3/`.

1. Preserve the completed result export and cleaned runner. The executed Colab
   notebook with outputs has not been supplied for this Git update.
2. Extend the original seed-0/1/2 comparison with seed 3 and a separately labeled
   seed-0 reproduction; do not count the reproduction as an independent seed.
3. Decide whether seeds 4/5 are needed under the same fixed protocol. Do not
   select training seeds or checkpoints using favorable test results.

Checkpoint tensors, full image data, detector caches and prediction NPZ/shards
remain on Drive and are excluded from this Git update. Test completion is
distinct from training completion and from commit/push status.


## Completed frozen test update (2026-10-20)

Archive SHA256: `8a76c4b572bc6cb9957a08990c2215ce57dbfab3142e5a11f7e66e167da7ef13`.
All 31 manifest-listed payload hashes passed. Eight checkpoint/threshold bindings
match the imported training records; CSV row counts and report-binding hash pass.
The completed runner reports 1,638 verified shards. This archive import verifies
report payloads; it does not independently download and recheck the large shards.
CSV line endings are normalized only in extracted Git files. The untouched ZIP,
original manifest and separate local checksum manifest preserve provenance.

| Condition | Seed0 rerun: M3 - R18 mAP (pp) | Seed3: M3 - R18 mAP (pp) |
|---|---:|---:|
| clean_s0 | +0.0933 | -0.0981 |
| blur_s4 | +3.4174 | -2.0898 |
| brightness_s4 | -0.8651 | -0.0224 |
| rotation_s4 | +1.2445 | +0.0955 |

Across all 21 conditions, original and reproduced M3 aggregate F1 (both threshold
rules) and mAP match exactly in the saved reports (maximum absolute difference
zero). This establishes metric reproduction on these conditions, not identical
weights or every individual logit. The M3 seed-0 robustness pattern therefore
persists in the new run; it cannot be dismissed as an invalid old run on these
results alone. Seed 3 shows positive rotation-s4 mAP difference but negative
strong-blur mAP difference. Neither uniform gains nor a specific cause of
between-seed variation is established.
