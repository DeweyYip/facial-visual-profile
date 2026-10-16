# Prediction case review

This is an illustrative, outcome-stratified case review, not a new benchmark or a causal experiment.
One image-attribute pair is selected per category and condition using a deterministic seeded draw.
An image is used at most once within a condition. Empty categories are recorded and not replaced.
Help/hurt categories require the same correctness contrast in at least two of three trained seeds.
All-four-correct/wrong categories require that outcome in at least two seeds. High-NME disagreement
requires NME > 0.20 and an M3/R18 correctness contrast in at least two seeds. Failure cases require
invalid detector output; they do not imply that the attribute classifier failed.
Selection uses per-model frozen thresholds; both threshold rules and all twelve runs are displayed.
Category eligibility and counts are in selection_coverage.csv. Categories overlap before selection.
These examples cannot establish prevalence, causality, significance, or seed-generalization.
Sigmoid scores are not demonstrated calibrated probabilities. GT targets are dataset annotations.
Original JPEG hashes, regenerated RGB pixels, corruption parameters and transformed GT are checked.
Landmark error is reconstructed for successful detections. No model or detector is rerun.
Whole-condition metrics are reproduced from saved predictions before selection.
Condition selections resume from checksum-verified, source-bound cache files. An interrupted condition
restarts. Run only one writer. Do not change settings for an existing output directory.

## Interpretation and limitations draft

Describe only the attribute decisions and detector outputs actually present in each selected image.
Not established: replacing only geometry caused a decision change. Oracle and M3 were trained separately.
Review question: do the selected cases show success and failure at different landmark-error levels?
Not established: the RGB model explicitly learned to localize the five points.
Examples are selected after inspecting outcomes and do not substitute for full-test quantitative results.
Training-run variation, recorded environment differences, subgroup prevalence and small groups remain limitations.
No quantitative causal conclusion should be drawn from the displayed cases.
