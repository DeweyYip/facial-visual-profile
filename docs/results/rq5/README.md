# RQ5: nonlinear geometry fusion and auxiliary landmark supervision

This archive records the completed RQ5 experiments. It adds nine trained runs
(N0/N1/MTL, seeds 0/1/2) and reuses three historical RQ4-A runs.

| ID | Model | Attribute head | Newly trained in RQ5 |
|---|---|---|---|
| A | Historical RGB-only R18 (RQ4-A) | Linear | No |
| N0 | RGB-only R18 with zero geometry placeholder | Nonlinear | Yes |
| N1 | R18 plus MTCNN-predicted five-point geometry | Matched nonlinear | Yes |
| MTL | R18 with attribute and auxiliary landmark regression heads | Linear attribute head | Yes |

Primary comparisons: N1-N0 and MTL-A. N0-A explains the head change.

## Results and evidence

- [Main paper table](tables/paper_main_table.md)
- [Paired seed contrasts](tables/paired_seed_contrasts.csv)
- [English findings draft](paper_findings_en.md)
- [Figure captions](figure_captions.md)
- [Chinese guide](README_zh.md)
- [Audit and verification limits](audit_report.json)

Figures are provided as vector PDF/SVG and 300 dpi PNG. Error bars are sample
SD across three seeds, not confidence intervals. Clean is excluded from the
20-condition corrupted average. Differences are in percentage points.

## Execution source provenance

`source_snapshot/` contains the exact eight files saved by the executed Colab
Notebook, independently downloaded from Drive and matched against every
SHA256 in `raw/notebook_execution_binding.json`. This includes the runtime
resume/logging fixes; it is not silently substituted with current root code.

The snapshot config intentionally retains `mtl_lambda=null`: it is the
original source template. The actually executed lambda (1.89811) and its
selection rule are in `raw/frozen_config.json` and `raw/lambda_scale_check.json`.

The historical baseline and all new run records are in `raw/`. Each source
file's Drive ID, original URL and downloaded byte hash appear in
`source_manifest.json` or `source_snapshot_manifest.json`.

The checkpoint binaries and CelebA assets remain on Drive and are not included.
Recorded checkpoint SHA256 references were cross-checked; weight binaries
and per-image predictions were not independently re-evaluated for this archive.
Historical A_seed0 ran six epochs despite the later four-epoch stopping gate;
both select epoch 2. See the audit report for this and environment differences.

## Verify and regenerate

From the repository root (verification requires Python standard library only):

```bash
python docs/results/rq5/scripts/verify_archive.py
```

Regenerate analysis tables/figures with numpy, pandas, matplotlib and reportlab:

```bash
python docs/results/rq5/scripts/build_evidence.py
```

`build_report.py` optionally creates a reading PDF; the report is not duplicated
in this Git archive. Rebuilding with different library versions may change
plot bytes; archive checksums describe this exact published snapshot.

The executed notebook belongs separately under `notebooks/`. No notebook or
existing model source is overwritten by this package.
