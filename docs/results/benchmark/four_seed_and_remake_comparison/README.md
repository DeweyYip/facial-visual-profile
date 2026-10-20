# Four original seeds and seed-0 remake comparison

Coverage: 420 overall rows and 10,080 attribute rows = 21 conditions x 4 models x 5 labeled runs.
Independent nominal seeds: original seed0, seed1, seed2 and seed3. Seed0 remake is a repeated run and excluded from every mean/SD calculation.
No result is removed or replaced. Original seed0 remains included in the four-seed descriptive summaries.
SD uses ddof=1; it is not a confidence interval. Recorded historical environment differences remain a confound, so variation is not attributed solely to seed.
F1(frozen), F1@0.5 and mAP/AP are separate. Each run uses its own previously frozen validation thresholds.
Family summaries weight severities equally within each run before summarizing the four original seeds.
Five-run plots use dashed black for the seed0 remake. The legend and CSV retain it even when curves overlap.
Paired model differences match condition and nominal run. Seed0 reproduction differences compare remake minus original for each model.
Maximum absolute M3 original/remake aggregate F1/mAP difference: 0 percentage points. This does not imply identical weights or individual logits.
Oracle is independently trained and is not a guaranteed upper bound. Oracle minus M3 is not a causal intervention on detector accuracy.
No training, inference, threshold fitting, shard rereading, bootstrap intervals or significance tests performed.

## Figure guide

- paired_m3_r18_all_five_runs: direct replacement for the old three-seed comparison.
- all_five_run_curves_*: every model and run shown separately.
- four_seed_curves_*: four-original-seed mean with sample-SD shading.
- seed0_remake_vs_original: remake minus original, per model and condition.
- attribute_heatmap_*: all 24 attributes, four-original-seed mean.

## Reproduce

python3 scripts/compare_four_seeds_and_remake.py --project . --output new_empty_comparison_folder
