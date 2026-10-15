# Three-seed NME subgroup and paired F1 analysis

NME groups: <=5%, (5%,10%], (10%,20%], >20%, detection failure.
All models are compared on the same detector-defined groups, including Oracle.
F1 uses all 24 attributes; undefined F1 is zero for nonempty groups. Empty groups remain missing.
AP is missing for attributes without positives. Both the positive-attribute-only mAP and
an all-24 convention assigning zero to absent-positive AP are explicitly labeled.
Do not compare tiny or different-prevalence subgroups as causal effects of NME.

Intervals: 95% percentile paired test-image bootstrap. Whole images, including all 24
labels, are resampled with replacement. The same resample is used across all models
and seeds within each condition. Intervals are provided for each fixed seed and the
mean difference across the three fixed trained seeds. No training-seed resampling is
performed. These intervals exclude training-run uncertainty and must not be interpreted
as confidence intervals over arbitrary future training seeds. Seed0 environment
changes are a further limitation. Intervals are exploratory, pointwise and unadjusted
for multiple comparisons; no p-values or blanket significance claims are produced.
mAP is reproduced and tabulated; mAP confidence intervals are not computed.

Existing inference thresholds are reused unchanged. No model inference, retraining,
threshold fitting or detector execution occurs. Resume verifies saved analysis payloads
and their frozen binding, then skips completed conditions; skipped conditions do not
re-read source shards. Remove no files and do not run concurrent writers.
