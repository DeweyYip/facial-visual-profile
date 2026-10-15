# Three-seed per-attribute review

Complete coverage: 6048 rows = 21 conditions x 4 models x 3 seeds x 24 attributes.
F1(frozen), F1@0.5 and AP are separate. mAP is the mean of the 24 attribute AP values.
Comparisons: Oracle - M3 and M3 - R18, matched by seed, attribute and condition.
Family differences first average the four severities within each seed, then summarize seeds.
SD uses ddof=1. All-3-seed direction consistency is descriptive, not statistical significance.
No test threshold fitting, retraining or inference was performed.
Oracle is an independently trained GT-reference model, not a guaranteed upper bound.
Oracle - M3 is not a causal detector-error effect: these are independently trained models.
Seed0 environment differences and only three training seeds limit generalization.
Do not select a new model/threshold using these test results and report it as an untouched test.

## Oracle - M3: all-corruption attribute differences
All attributes are reported in the predefined order, without post-hoc selection.

- Smiling: f1_frozen -0.909 pp (positive seeds 1/3); f1_at_0.5 -0.042 pp (positive seeds 1/3); ap -0.184 pp (positive seeds 1/3)
- Mouth_Slightly_Open: f1_frozen -1.251 pp (positive seeds 1/3); f1_at_0.5 +1.091 pp (positive seeds 1/3); ap -0.164 pp (positive seeds 1/3)
- Eyeglasses: f1_frozen +0.474 pp (positive seeds 2/3); f1_at_0.5 -1.396 pp (positive seeds 1/3); ap -0.184 pp (positive seeds 2/3)
- Bangs: f1_frozen +0.674 pp (positive seeds 2/3); f1_at_0.5 +0.526 pp (positive seeds 2/3); ap -0.014 pp (positive seeds 2/3)
- Wearing_Hat: f1_frozen +1.103 pp (positive seeds 2/3); f1_at_0.5 +0.713 pp (positive seeds 1/3); ap +0.367 pp (positive seeds 3/3)
- Mustache: f1_frozen -0.362 pp (positive seeds 2/3); f1_at_0.5 +0.119 pp (positive seeds 2/3); ap +0.950 pp (positive seeds 2/3)
- No_Beard: f1_frozen +0.087 pp (positive seeds 2/3); f1_at_0.5 +0.026 pp (positive seeds 1/3); ap -0.005 pp (positive seeds 1/3)
- Black_Hair: f1_frozen -0.902 pp (positive seeds 1/3); f1_at_0.5 -1.007 pp (positive seeds 1/3); ap +0.417 pp (positive seeds 3/3)
- Bushy_Eyebrows: f1_frozen +1.148 pp (positive seeds 1/3); f1_at_0.5 +4.562 pp (positive seeds 2/3); ap +0.502 pp (positive seeds 2/3)
- Arched_Eyebrows: f1_frozen +0.360 pp (positive seeds 2/3); f1_at_0.5 +0.325 pp (positive seeds 2/3); ap -0.294 pp (positive seeds 2/3)
- High_Cheekbones: f1_frozen -1.402 pp (positive seeds 0/3); f1_at_0.5 -2.257 pp (positive seeds 1/3); ap -0.444 pp (positive seeds 0/3)
- Bags_Under_Eyes: f1_frozen +1.651 pp (positive seeds 3/3); f1_at_0.5 -1.155 pp (positive seeds 0/3); ap +0.286 pp (positive seeds 2/3)
- Big_Lips: f1_frozen -3.245 pp (positive seeds 0/3); f1_at_0.5 +1.724 pp (positive seeds 2/3); ap +0.293 pp (positive seeds 2/3)
- Big_Nose: f1_frozen +1.434 pp (positive seeds 2/3); f1_at_0.5 -1.257 pp (positive seeds 0/3); ap -0.113 pp (positive seeds 2/3)
- Double_Chin: f1_frozen -0.173 pp (positive seeds 1/3); f1_at_0.5 -1.891 pp (positive seeds 2/3); ap -0.206 pp (positive seeds 0/3)
- Oval_Face: f1_frozen -0.969 pp (positive seeds 1/3); f1_at_0.5 -2.273 pp (positive seeds 1/3); ap +0.078 pp (positive seeds 2/3)
- Receding_Hairline: f1_frozen -0.037 pp (positive seeds 2/3); f1_at_0.5 -1.998 pp (positive seeds 0/3); ap +0.167 pp (positive seeds 2/3)
- Blond_Hair: f1_frozen +2.994 pp (positive seeds 3/3); f1_at_0.5 +0.643 pp (positive seeds 2/3); ap +0.142 pp (positive seeds 2/3)
- Brown_Hair: f1_frozen -0.687 pp (positive seeds 2/3); f1_at_0.5 +1.539 pp (positive seeds 3/3); ap +0.638 pp (positive seeds 3/3)
- Gray_Hair: f1_frozen +2.107 pp (positive seeds 3/3); f1_at_0.5 -2.085 pp (positive seeds 1/3); ap +0.096 pp (positive seeds 2/3)
- Bald: f1_frozen -2.111 pp (positive seeds 0/3); f1_at_0.5 +0.570 pp (positive seeds 1/3); ap +0.052 pp (positive seeds 2/3)
- Straight_Hair: f1_frozen +0.519 pp (positive seeds 2/3); f1_at_0.5 -1.990 pp (positive seeds 1/3); ap +1.031 pp (positive seeds 3/3)
- Wavy_Hair: f1_frozen +0.544 pp (positive seeds 3/3); f1_at_0.5 +4.230 pp (positive seeds 2/3); ap +0.207 pp (positive seeds 2/3)
- Sideburns: f1_frozen +0.845 pp (positive seeds 3/3); f1_at_0.5 +2.880 pp (positive seeds 3/3); ap +0.169 pp (positive seeds 2/3)
