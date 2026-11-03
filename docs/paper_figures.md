# Paper figures and evidence

## Manuscript status

This index maps the **26-page submission-candidate manuscript**, *Explicit Structure and Model Depth for Robust Facial Attribute Recognition* (Diwei Ye), to the repository's existing evidence. It covers RQ1–RQ5 and Appendices A–E. The author-approved PDF and a public arXiv identifier are **not currently linked here**; do not infer that a submission has been posted from this index.

The final manuscript uses Figures **1–8, B1–B2 and E1–E3**. Its appendices contain Tables A1–A2, B1, D1 and E1. Figures C1 and D1/D2 from earlier drafts are **not figures in the current 26-page manuscript**. Historical experiment archives remain available separately and should not be relabeled as current manuscript figures.

## Figure-to-evidence mapping

| Current manuscript figure | Data or archived evidence | Plotting source / important distinction |
|---|---|---|
| Figure 1 — M3 / Oracle fusion architecture | [ResNet18 geometry model](../models/resnet18_geometry.py) | Manuscript diagram; exact diagram-authoring source is not archived here |
| Figure 2 — initial benchmark, frozen-F1 severity curves | [Four-seed comparison](results/benchmark/four_seed_and_remake_comparison/README.md) | [Comparison script](../scripts/compare_four_seeds_and_remake.py); archived `four_seed_curves_f1_frozen` |
| Figure 3 — detector failure and NME | [NME subgroup notebook](../notebooks/three_seed_nme_and_f1_intervals.ipynb) | Shared detector inputs; subgroup attribute-model comparisons use original seeds 0–2, not four |
| Figure 4 — M3−R18 attribute-by-family differences | [Attribute differences CSV](results/benchmark/four_seed_and_remake_comparison/attribute_family_differences_pp.csv) | Manuscript shows two panels with a fixed ±6 pp color limit and **blue positive / red negative**; archived comparison plot has different layout and dynamic limits |
| Figure 5 — RQ4 absolute corruption performance | [RQ4 tables](results/rq4/README.md) | Six conditions A–F, three seeds; manuscript plotting layout is not claimed reproducible byte-for-byte from the archived plotting script |
| Figure 6 — RQ4 paired geometry effects | [RQ4 paired contrasts](results/rq4/paired_seed_contrasts.csv) | B−A, D−C, F−E by nominal seed; percentage-point differences |
| Figure 7 — RQ5 paired effects | [RQ5 archive](results/rq5/README.md) · [figure captions](results/rq5/figure_captions.md) | Archived `fig01_paired_effects`; N1−N0 and MTL−A are primary, N0−A supporting |
| Figure 8 — RQ5 corruption-severity curves | [RQ5 archive](results/rq5/README.md) · [figure captions](results/rq5/figure_captions.md) | Archived `fig02_corruption_curves`; clean reference is excluded from corrupted average |
| Figure B1 — initial benchmark mAP severity curves | [Four-seed comparison](results/benchmark/four_seed_and_remake_comparison/README.md) | [Comparison script](../scripts/compare_four_seeds_and_remake.py); archived `four_seed_curves_map` |
| Figure B2 — seed-specific M3−R18 differences | [Four-seed comparison](results/benchmark/four_seed_and_remake_comparison/README.md) | Archived comparison script generates related panels, but manuscript arrangement differs |
| Figure E1 — RQ5 paired effects by corruption family | [RQ5 archive](results/rq5/README.md) · [figure captions](results/rq5/figure_captions.md) | Archived `fig03_family_effects`; within-family four-severity mean, then paired effect |
| Figure E2 — RQ5 training curves | [RQ5 archive](results/rq5/README.md) · [figure captions](results/rq5/figure_captions.md) | Archived `fig05_training_curves`; right panel displays attribute BCE, not full MTL loss |
| Figure E3 — RQ5 per-attribute effects | [RQ5 attribute records](results/rq5/README.md) · [figure captions](results/rq5/figure_captions.md) | Archived `fig04_attribute_effects` uses **red positive / blue negative**. The manuscript Figure E3 reverses the palette to **blue positive / red negative**; it was formatted separately from the archived plot. Do **not** claim byte-identical plots or alter archived source data to force visual agreement |

### Table / protocol pointers

- **Tables 1–5, A1–A2, B1**: See the [initial benchmark records](results/benchmark/four_seed_and_remake_comparison/README.md), [benchmark protocol](benchmark_protocol_v1.md) and [attribute configuration](../configs/attributes.yaml).
- **Tables 6–9 and D1 (RQ4)**: See [RQ4 source summaries](results/rq4/README.md) and the [18-run notebook](../notebooks/rq4_18run_earlystopping_resume_v2.ipynb).
- **Tables 10–12 and E1 (RQ5)**: See [RQ5 archive](results/rq5/README.md), [paper main table](results/rq5/tables/paper_main_table.md), [paired contrasts](results/rq5/tables/paired_seed_contrasts.csv) and [audit limits](results/rq5/audit_report.json).
- **Appendix C** describes NME subgroup scope; it does not introduce an additional figure in this manuscript.

## Reproduce archived analysis (not all manuscript layout)

From the repository root:

```bash
python3 scripts/compare_four_seeds_and_remake.py --project . --output comparison_regenerated
python docs/results/rq5/scripts/verify_archive.py
```

The comparison command checks archived source hashes and regenerates its historical analysis plots; it does not rerun training or detector inference. The RQ5 verification command reconciles the archived summaries; it does **not** download and independently rerun checkpoint binaries or every per-image prediction. RQ5 analysis tables and plot exports can be regenerated with `python docs/results/rq5/scripts/build_evidence.py` when its documented dependencies are available.

Source figures and manuscript figures may differ in layout, axis annotations, or palette even when derived from the same archived summaries. Preserve original evidence and disclose those differences rather than silently replacing archived images.
