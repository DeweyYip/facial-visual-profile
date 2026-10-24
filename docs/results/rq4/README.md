# RQ4 — Geometry utility under corruption

**Study date:** October 2026. **Source:** completed Google Colab 18-run study in the connected project's Google Drive under `MyDrive/FacialVisualProfile/runs/rq4_training_v1/`.

This folder archives the original five machine-generated **CSV result tables without editing their numeric contents**. We do not commit model weights, image datasets, or `latest.pt` checkpoints.

## Research question

> Why does accurate explicit five-point geometry fail to provide consistent gains in facial attribute recognition under image corruption?

The study examines whether **training distribution** or **RGB feature masking** changes the measured incremental advantage of ground-truth landmark geometry. It cannot by itself prove a unique mechanism, information-theoretic insufficiency or information redundancy.

## Experimental design

| Group | Input | Training data | RGB branch masking |
|---|---|---|---|
| A | R18 RGB only | clean | no |
| B | R18 + GT geometry | clean | no |
| C | R18 RGB only | 50% stochastic corruption augmentation | no |
| D | R18 + GT geometry | 50% stochastic corruption augmentation | no |
| E | R18 RGB only | clean | yes (20% sample-level probability) |
| F | R18 + GT geometry | clean | yes (20% sample-level probability) |

Three nominal seeds (0, 1, 2) per group, 18 independent training runs. All six use the official CelebA train/valid/test split with 24 attributes, ImageNet-pretrained R18 initialization, batch size 64 and LR 1e-4. The recorded policy is maximum 10 epochs, minimum 4 epochs, early-stopping patience 2 on **clean-validation mAP**. Each run fits frozen F1 thresholds on clean validation only. The final test covers 19,962 images, one clean plus 20 corruption conditions; the corruption mean excludes clean.

The six models are *newly trained*. Original R18, R50, predicted-landmark M3 and historical GT Oracle checkpoints are not secretly substituted into any A–F contrast. GT geometry is privileged input.

## Results: average of 20 corrupted conditions

| Model | mAP mean ± seed SD | Frozen macro-F1 mean ± seed SD |
|---|---:|---:|
| A | 76.048 ± 0.226% | 61.369 ± 0.047% |
| B | 75.975 ± 0.142% | 61.911 ± 0.139% |
| C | 79.776 ± 0.121% | 73.422 ± 0.162% |
| D | 79.820 ± 0.211% | 73.546 ± 0.085% |
| E | 75.798 ± 0.557% | 61.280 ± 0.373% |
| F | 75.967 ± 0.336% | 61.129 ± 0.141% |

Mean/SD: compute each seed's equal-weight average over 20 corruptions and then summarize three seeds using sample SD. Clean test frozen F1 for A–F is roughly 75.45–75.65%, not the 61–74% corrupted scores.

### Geometry value (GT fusion minus matched RGB-only)

| Contrast | ΔmAP (pp) | Δfrozen macro-F1 (pp) |
|---|---:|---:|
| B − A: clean training | −0.073 | +0.542 |
| D − C: corruption-aware training | +0.044 | +0.124 |
| F − E: RGB masking training | +0.169 | −0.150 |

The mAP effects are small and directionally inconsistent across seeds. Corruption-aware training improves **both** C and D in absolute performance; the benefit is not uniquely attributable to geometry. The masking results are metric-dependent, and F vs E alone cannot isolate how the fusion model learned to use landmarks.

### Interactions

- Training-distribution interaction `(D−C)−(B−A)`: approximately **+0.117 pp mAP / −0.417 pp F1**.
- RGB-masking interaction `(F−E)−(B−A)`: approximately **+0.242 pp mAP / −0.692 pp F1**.

The three seed-specific interactions must be checked in `paired_interactions.csv`; signs are not consistently positive for mAP. Three-seed descriptive differences are not statistical significance tests.

## Files and how to use them

| File | Rows / meaning |
|---|---|
| [`condition_results.csv`](condition_results.csv) | **378 rows** = 6 groups × 3 seeds × 21 conditions; columns: model, seed, condition, family, mAP, frozen_F1, F1_at_05. Most detailed aggregate source. |
| [`corruption_averages_by_seed.csv`](corruption_averages_by_seed.csv) | 18 rows: per-seed 20-condition corruption averages. |
| [`corruption_summary.csv`](corruption_summary.csv) | Mean and sample SD across three seeds; retains the original pandas two-line header. |
| [`paired_seed_contrasts.csv`](paired_seed_contrasts.csv) | 9 rows: B−A, D−C and F−E for seeds 0–2; percentage-point differences. |
| [`paired_interactions.csv`](paired_interactions.csv) | 6 rows: per-seed training-distribution and RGB-masking interaction differences. |

**Authoritative checkpoints and detailed training histories** remain in Google Drive `MyDrive/FacialVisualProfile/runs/rq4_training_v1/{A..F}_seed{0..2}/`. This GitHub copy is a source-linked research results archive, not an executable set of trained models.

**Related code:** [RQ4 Colab notebook](../../../notebooks/rq4_18run_earlystopping_resume_v2.ipynb).  
**Original research:** [research release v1](../../releases/research_v1/README.md) and [four-seed comparisons](../benchmark/four_seed_and_remake_comparison/README.md).

## Interpretation limits

- Strong absolute gain from corruption-aware training does not imply a strong *geometry-specific* gain.
- GT Oracle provides privileged landmarks. This does not measure deployment-time MTCNN reliability.
- mAP is threshold-free; frozen-threshold F1 is not. A gain in one does not guarantee a gain in the other.
- The same five corruption families are seen in augmentation and test; this is not a held-out-corruption-generalization claim.
- Repeated test inspection makes the mechanism analyses exploratory; use clean validation for checkpoint and threshold selection.
- Do not claim that five points contain no information, or that RGB feature dominance is definitively proven.
