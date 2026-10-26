# Notebook index

Use this index to distinguish **historical development notebooks**, **completed research benchmarks** and the **new RQ4 intervention study**. No archived notebook has been removed or renamed. Run status must be verified against its experiment reports, not inferred from a filename alone.

## Start here

| Notebook | What it is |
|---|---|
| [RQ4 — 18-run training with early stopping](rq4_18run_earlystopping_resume_v2.ipynb) | Six conditions (A–F) × three seeds; paired clean-only, corruption-aware and RGB-masking training; resume/checkpoint logic. [RQ4 result index](../docs/results/rq4/README.md). |
| [Multi-seed training and validation](multiseed_training_and_validation.ipynb) | Historical multi-model training and clean-validation workflow; inspect archived configs before reuse. |
| [Multi-seed frozen test benchmark](multiseed_frozen_test_benchmark.ipynb) | Test-time evaluation against the fixed 21-condition benchmark. |
| [Three-seed NME and F1 intervals](three_seed_nme_and_f1_intervals.ipynb) | Landmark reliability subgroups and paired image bootstrap intervals, not causal identification. |
| [Three-seed case review](three_seed_prediction_case_review.ipynb) | Selected qualitative examples; examples do not estimate dataset-level error prevalence. |
| [Small-scale inference reproduction](small_scale_inference_reproduction.ipynb) | Fixed-image inference replay; see the archived reproduction report for exact checks. |

## Development notebooks

| Milestone | Notebooks | Function |
|---|---|---|
| Day 4–8 | [Day 4](day4_resnet18_benchmark.ipynb), [Day 5](day5_mtcnn_inspection.ipynb), [Day 6](day6_resnet18_seed0.ipynb), [Day 7](day7_resnet18_validation.ipynb), [Day 8](day8_resnet50_seed0.ipynb) | R18/R50 setup, MTCNN inspection and initial validation |
| Day 10–13 | [Day 10](day10_predicted_geometry_integration.ipynb), [Day 11](day11_m3_seed0.ipynb), [Day 12](day12_clean_validation_comparison.ipynb), [Day 13](day13_training_protocol_and_reload_audit.ipynb) | Geometry branch integration and audit |
| Day 16–21 | [Day 16](day16_detector_quality_and_pipeline_smoke.ipynb), [Day 18](day18_oracle_seed0.ipynb), [Day 19](day19_clean_blur_brightness_benchmark.ipynb), [Day 20](day20_rotation_occlusion_jpeg_benchmark.ipynb), [Day 21](day21_oracle_benchmark_and_comparison.ipynb) | Corruptions, benchmark generation and GT Oracle analysis |

Additional archived notebooks in this directory (e.g. overfit/failure checks, seed-specific repeat runs, upper-half R50 work, seed-0 reproduction) are retained as historical records. See [experiment log](../docs/experiment_log.md) and [development record](../docs/seed2_to_seed0_reproduction_20261020.md) for provenance.

## Storage rules

- Notebooks and lightweight reproducibility metadata are tracked in Git; CelebA images, GPU checkpoints and large prediction shards stay outside this repository.
- Source snapshots in `docs/results/` may be deliberately duplicated for archival binding. Do not deduplicate without reviewing evidence manifests.
- RQ4 final results are in [`docs/results/rq4/`](../docs/results/rq4/README.md), while original model comparisons remain in [`docs/results/benchmark/`](../docs/results/benchmark/).
