# Three-seed overall comparison

Verified 252 overall rows: 21 conditions x 4 models x 3 training seeds.
F1(frozen), F1@0.5 and mAP are reported separately. SD uses ddof=1.
Model differences pair the same seed and condition. Values in difference tables are percentage points.
Family summaries average severity levels within each seed, then summarize the three seed averages.
These are descriptive results. No significance tests or confidence intervals were computed.
Test-image uncertainty, NME subgroup analysis and prediction-shard revalidation are not included.
Oracle is an independently trained GT-reference model, not a guaranteed upper bound.
Seed0 and later seeds have recorded environment differences; do not attribute all variation solely to random seed.

## Rotation severity 4: M3 minus R18

- f1_frozen: Seed0: +2.397 pp, Seed1: -1.951 pp, Seed2: -3.257 pp; mean -0.937 pp, SD 2.960 pp.
- f1_at_0.5: Seed0: +4.314 pp, Seed1: -5.611 pp, Seed2: -3.562 pp; mean -1.620 pp, SD 5.240 pp.
- map: Seed0: +1.186 pp, Seed1: -0.472 pp, Seed2: -0.683 pp; mean +0.010 pp, SD 1.023 pp.
