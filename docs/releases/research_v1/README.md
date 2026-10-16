# Research release v1

Status: local research-result snapshot; not a published paper or a GitHub Release.

## Research question

Does concatenating an explicit five-point geometry branch improve CelebA facial-attribute robustness relative to RGB baselines?

## Frozen experimental setup

- CelebA official splits: 162770 train, 19867 validation, 19962 test; 24 attributes.
- Four models: ImageNet-initialized R18, R50, R18 plus predicted MTCNN geometry (M3), and an independently trained GT-geometry reference (Oracle).
- Geometry branch: 10 coordinates -> 32 -> 32; concatenate with 512 RGB features. Invalid M3 geometry is masked after the MLP; samples are retained.
- Three training seeds (0, 1, 2); ten epochs per run; Adam, learning rate 0.0001, batch 64, FP32, no augmentation.
- Select highest clean-validation mAP, earliest epoch on ties. Fit per-attribute F1 thresholds on clean validation only; grid 0.05..0.95, step 0.01, tie closest to 0.5 then lower.
- Shared 224x224 letterbox preprocessing. Five corruption families x four severities, plus clean: 21 conditions. Same test inputs are paired across models and training seeds.
- Blur sigma 1/2/3/4; brightness factors 0.8/0.6/0.4/0.2; rotation magnitude 5/10/15/20 degrees with frozen deterministic signs; lower-face square occlusion side fractions 0.1/0.2/0.3/0.4; JPEG quality 75/50/25/10.
- F1(frozen), F1@0.5 and mAP remain separate. No test threshold fitting or test-based checkpoint selection.

## Descriptive summary

For corruptions, average the twenty conditions equally within each seed, then report mean +/- sample SD across the three resulting seed values. Clean is excluded from that average. SD is not a confidence interval.

| Scope | Model | F1(frozen) mean +/- SD | F1@0.5 mean +/- SD | mAP mean +/- SD |
|---|---|---|---|---|
| clean | r18 | 0.756943 +/- 0.000719 | 0.729237 +/- 0.005165 | 0.820533 +/- 0.000847 |
| clean | r50 | 0.756154 +/- 0.000164 | 0.726720 +/- 0.003268 | 0.819212 +/- 0.000191 |
| clean | m3 | 0.757145 +/- 0.000944 | 0.730369 +/- 0.007914 | 0.819474 +/- 0.000168 |
| clean | oracle | 0.757131 +/- 0.000674 | 0.729257 +/- 0.011074 | 0.820158 +/- 0.000265 |
| all_20_corruptions | r18 | 0.619917 +/- 0.000966 | 0.570292 +/- 0.007497 | 0.759282 +/- 0.003122 |
| all_20_corruptions | r50 | 0.624557 +/- 0.001357 | 0.577870 +/- 0.008813 | 0.762835 +/- 0.001745 |
| all_20_corruptions | m3 | 0.612086 +/- 0.001875 | 0.562734 +/- 0.012673 | 0.756334 +/- 0.002424 |
| all_20_corruptions | oracle | 0.612875 +/- 0.004829 | 0.563400 +/- 0.013318 | 0.757912 +/- 0.002938 |

## Supported interpretation and limits

- These three runs do not show a consistent, universal gain from the current predicted-geometry fusion over R18. This does not establish equivalence or prove geometry is generally useless.
- Oracle is separately trained and is neither a guaranteed upper bound nor a causal intervention on M3.
- NME subgroup results describe associations. Group difficulty, attribute prevalence and small subgroup counts can affect comparisons. They do not identify a causal effect of localization error.
- The 1000-replicate paired image-bootstrap F1 intervals are pointwise, unadjusted and conditional on fixed trained models. They exclude training-seed uncertainty. mAP confidence intervals were not computed.
- Seed0 and subsequent runs have recorded environment differences; variability cannot be attributed exclusively to the random seed.
- Twenty-four outcome-stratified qualitative cases illustrate successes, failures and possible annotation ambiguity. They do not estimate label-error prevalence or population model superiority. Dataset labels remain unchanged.
- R18 attribute performance does not demonstrate explicit five-point localization ability.

## Small-scale reproduction

- Seed1, four models x eight fixed test images x four conditions: 128 model-sample evaluations; all replay comparisons passed. Maximum absolute logit difference: 1.31130219e-05.
- Both threshold rules produced zero changed decisions. Reference GPU and replay CPU/hardware details are in the replay execution binding.
- Detector inputs were reused; this is not a fresh MTCNN execution or a full-benchmark replay.
- Runtime is recorded in the replay report. Monetary cost was not automatically measured and is not treated as zero.

## Evidence index

- [Overall metrics](../../results/benchmark/three_seed_comparison/all_seed_metrics.csv)
- [Frozen F1 curves](../../results/benchmark/three_seed_comparison/curves_f1_frozen.png)
- [F1 at 0.5 curves](../../results/benchmark/three_seed_comparison/curves_f1_at_0_5.png)
- [mAP curves](../../results/benchmark/three_seed_comparison/curves_map.png)
- [Per-seed M3 minus R18](../../results/benchmark/three_seed_comparison/paired_m3_r18.png)
- [Fixed-model image-bootstrap intervals](../../results/benchmark/three_seed_nme_and_f1_intervals/m3_r18_fixed_seed_image_intervals.png)
- [NME groups, frozen](../../results/benchmark/three_seed_nme_and_f1_intervals/nme_groups_frozen_thresholds.png)
- [NME groups, 0.5](../../results/benchmark/three_seed_nme_and_f1_intervals/nme_groups_fixed_0_5.png)
- [M3 minus R18 attribute heatmap](../../results/benchmark/three_seed_attribute_comparison/attribute_heatmap_m3_minus_r18.png)
- [Oracle minus M3 attribute heatmap](../../results/benchmark/three_seed_attribute_comparison/attribute_heatmap_oracle_minus_m3.png)
- [Attribute family differences](../../results/benchmark/three_seed_attribute_comparison/attribute_family_differences_pp.csv)
- [Selected cases and source records](../../results/benchmark/prediction_case_review/selected_cases.json)
- [Case predictions](../../results/benchmark/prediction_case_review/case_predictions.csv)
- [Replay report](../../results/benchmark/small_scale_reproduction/reproduction_report.json)

## Reusable notebooks

- [multiseed_training_and_validation.ipynb](../../../notebooks/multiseed_training_and_validation.ipynb)
- [multiseed_training_and_validation_seed2.ipynb](../../../notebooks/multiseed_training_and_validation_seed2.ipynb)
- [multiseed_frozen_test_benchmark.ipynb](../../../notebooks/multiseed_frozen_test_benchmark.ipynb)
- [three_seed_nme_and_f1_intervals.ipynb](../../../notebooks/three_seed_nme_and_f1_intervals.ipynb)
- [three_seed_prediction_case_review.ipynb](../../../notebooks/three_seed_prediction_case_review.ipynb)
- [small_scale_inference_reproduction.ipynb](../../../notebooks/small_scale_inference_reproduction.ipynb)

## Known metadata issue

The original case-review analysis_binding.json conditions field mistakenly contains official test IDs because of a variable-name collision. The four actual reviewed conditions are present in completion_audit.json, condition caches and selected_cases.json. Original files and hashes are preserved; no silent replacement is made.

## Remaining work

- Complete the detailed interpretation of minority-attribute results and the full case-review narrative.
- Record actual compute spending if required; the current monetary entry is unmeasured.
- Future reuse of case-selection code should fix the metadata field in a new version/output directory.
- Additional controls, training repetitions, detector reruns and broader datasets remain optional future studies, not completed experiments.
- Full paper writing, related-work sourcing, final paper figure selection and demo implementation are not completed by this release.
- October 26 tasks are not specified in the supplied schedule screenshot.
- Checkpoints and full prediction shards remain on Drive. This index does not package them or make the project executable without those prerequisites.

## Snapshot provenance

- Repository HEAD before this release was assembled: `5a8cbde4208b753b528f0ebef5e870f2a8449664`.
- See evidence_manifest.json for file hashes and pre-existing tracked modifications. The archived experiment snapshots, rather than unreviewed working-tree model edits, define the experiments.
- This command does not commit, push, tag, delete, train or run inference.
