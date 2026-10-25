# Paper figures and evidence

## Manuscript availability

The manuscript is undergoing final author formatting and is not yet published in this repository. This index links the existing public evidence; it does not advertise a missing PDF or an arXiv submission. Add the author-approved final PDF or a stable public manuscript URL here when it is published.

## Figure mapping

| Manuscript figure | Data or archived evidence | Existing plotting entry or status |
|---|---|---|
| 1 Fusion architecture | [model definition](../models/resnet18_geometry.py) | Manuscript diagram; exact authoring source not archived here |
| 2 Frozen F1 severity curves; B1 mAP curves | [four-seed results](results/benchmark/four_seed_and_remake_comparison/) | [comparison script](../scripts/compare_four_seeds_and_remake.py); `four_seed_curves_f1_frozen` and `four_seed_curves_map` outputs |
| 3 Detector reliability; Table 5 subgroups | [subgroup notebook](../notebooks/three_seed_nme_and_f1_intervals.ipynb) and its archived inputs | Notebook covers original seeds 0–2 subgroup analysis; detector inputs are shared across classifier seeds |
| 4 Attribute effects | [attribute differences CSV](results/benchmark/four_seed_and_remake_comparison/attribute_family_differences_pp.csv) | Comparison script produces three panels with a dynamic color limit; manuscript uses two panels and a fixed ±6 pp limit. Matching data does not imply identical figure layout |
| 5 RQ4 absolute performance | [RQ4 CSVs](results/rq4/README.md) | Manuscript-specific plotting code not yet archived; use corruption averages per seed, then sample SD |
| 6 RQ4 paired geometry effects | [paired contrasts](results/rq4/paired_seed_contrasts.csv) | Manuscript-specific plotting code not yet archived; plot B−A, D−C and F−E separately by seed |
| B2 Run-specific differences | [four-seed results](results/benchmark/four_seed_and_remake_comparison/) | Comparison script exports paired differences and broader comparison plots; manuscript panel arrangement differs |
| C1 Seed-0 reproduction | [reproduction differences](results/benchmark/four_seed_and_remake_comparison/seed0_remake_differences_pp.csv) | Comparison script; `seed0_remake_vs_original` output |
| D1 and D2 Qualitative cases | [case archive](results/benchmark/prediction_case_review/README.md) | [case-review notebook](../notebooks/three_seed_prediction_case_review.ipynb); Smiling image 191211 and Mouth_Slightly_Open image 185096 at blur severity 4 |

## Reproduce the archived comparison plots

From the repository root, install the base local dependencies and run:

```bash
python3 scripts/compare_four_seeds_and_remake.py --project . --output comparison_regenerated
```

Use a new or empty output directory. The script checks source hashes, attribute ordering, sample counts, threshold consistency and aggregate reconstruction before drawing. It does not rerun training, detection or inference. Keep the original archives intact. This command reproduces the archived analysis outputs, not every manuscript figure's exact typography or layout.
