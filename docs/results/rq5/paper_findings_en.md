# RQ5 evidence and manuscript wording

## Result summary

- N1-N0, clean: mAP +0.0092 ± 0.0414 pp (2/3 positive seeds); Frozen-threshold macro-F1 -0.1141 ± 0.2306 pp (1/3 positive seeds).
- N1-N0, corrupted: mAP +0.0063 ± 0.2257 pp (2/3 positive seeds); Frozen-threshold macro-F1 -0.2856 ± 0.4423 pp (1/3 positive seeds).
- MTL-A, clean: mAP -0.0325 ± 0.0434 pp (1/3 positive seeds); Frozen-threshold macro-F1 +0.1619 ± 0.1951 pp (3/3 positive seeds).
- MTL-A, corrupted: mAP -0.4837 ± 0.3579 pp (0/3 positive seeds); Frozen-threshold macro-F1 +0.4666 ± 0.6267 pp (2/3 positive seeds).
- N0-A, clean: mAP -0.1039 ± 0.1213 pp (0/3 positive seeds); Frozen-threshold macro-F1 +0.0281 ± 0.0764 pp (1/3 positive seeds).
- N0-A, corrupted: mAP -0.0328 ± 0.1306 pp (1/3 positive seeds); Frozen-threshold macro-F1 +0.0061 ± 0.4390 pp (1/3 positive seeds).

## Suggested Results paragraph

Matching the nonlinear classification head did not establish a consistent benefit from predicted five-point geometry. On the corrupted-test average, N1 and N0 had nearly identical mAP, while N1 had lower frozen-threshold macro-F1 on average. Paired effects changed sign across seeds. Auxiliary landmark supervision also did not provide a uniform improvement over the historical RGB-only baseline: MTL reduced corrupted mAP in all three seeds, while its mean frozen-threshold F1 increased with an inconsistent seed-level effect. The nonlinear RGB-only head itself remained close to the linear baseline on corrupted averages. These findings do not support a general robustness benefit from either intervention under the tested protocol.

## Interpretation limits

F1 and AP measure different aspects of predictions. Their divergent changes do not by themselves identify calibration, ranking, or a causal mechanism. The experiments do not establish that five landmarks contain no useful information, or rule out other fusion designs or auxiliary-loss choices. MTL used one fixed training-only scale rule (lambda=1.89811), not a sweep. RQ5 is a test-informed follow-up on the same benchmark; avoid presenting it as untouched held-out confirmation. Family and attribute plots are exploratory. Error bars are sample SD across three seeds, not confidence intervals. Results are scoped to the measured CelebA 24 attributes, ResNet18, five-point geometry, frozen conditions and budget.

## Provenance and verification

Read audit_report.json and source_manifest.json. Macro metrics and archive summaries were independently reconciled against per-run aggregate JSON. Checkpoint binary bytes and per-image model predictions were not independently evaluated in this analysis. No existing model weights, lambda, thresholds, test scores or paper files were modified.
