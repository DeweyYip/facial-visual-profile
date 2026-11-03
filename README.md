# Facial Visual Profile

**Research project:** Landmark-Guided Deep Learning for Robust Facial Attribute Analysis

Facial Visual Profile investigates whether explicit **five-point facial geometry** provides consistent value beyond RGB features for multi-label facial attribute recognition, particularly when images are corrupted. It is an experimental research repository, **not** a claim that geometry fusion universally improves robustness.

See the [paper figure and evidence index](docs/paper_figures.md) for manuscript-to-result mappings and the current manuscript availability status.

## Research at a glance

- **Task:** 24-attribute multi-label prediction on the official [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned-image splits: 162,770 train / 19,867 validation / 19,962 test.
- **Models:** ImageNet-initialized ResNet18, ResNet50, ResNet18 + MTCNN-predicted five-point geometry (**M3**), and a separately trained ResNet18 + ground-truth geometry (**Oracle**).
- **Fusion:** 512-dimensional RGB feature + 32-dimensional landmark MLP feature, concatenated and passed through a linear 24-logit classifier.
- **Evaluation:** clean input and five corruption families (blur, darkening, rotation, lower-face occlusion, JPEG compression), four severity levels each; 21 conditions per run.
- **Metrics:** mAP and macro-F1 with per-attribute thresholds fitted on clean validation and frozen for testing. F1 at a fixed 0.5 threshold is separately reported.
- **Research extension (RQ4):** tests whether corruption-aware training or RGB feature masking changes the incremental utility of **accurate** geometry.
- **Research extension (RQ5):** compares a nonlinear RGB-only head (N0) against predicted-landmark nonlinear fusion (N1), plus RGB-only attribute learning against auxiliary landmark supervision (MTL). Nine new runs use seeds 0–2; RQ4-A is reused as the linear RGB baseline.

## Key findings

**Original four-model study.** Explicit five-point fusion did **not** demonstrate a stable, universal robustness advantage over RGB-only models. Effects depend on the attribute, corruption, metric and run. The GT Oracle is a *separately trained diagnostic*, not a causal replacement of M3's detector output. See the [archived research release](docs/releases/research_v1/README.md) and [four-seed comparison](docs/results/benchmark/four_seed_and_remake_comparison/README.md). The research release describes its **three-seed snapshot**; later four-seed results must not be conflated with it.

Original corruption averages across 20 conditions and original nominal seeds 0–3 (mean ± sample SD; %):

| Model | Frozen macro-F1 | mAP |
|---|---:|---:|
| R18 | 61.94 ± 0.13 | 76.02 ± 0.32 |
| R50 | 62.37 ± 0.20 | 76.31 ± 0.15 |
| M3 | 61.28 ± 0.21 | 75.76 ± 0.32 |
| Oracle | 61.31 ± 0.40 | 75.90 ± 0.33 |

The separate seed-0 reproduction is excluded. Historical environment and loader differences remain documented limitations.

**RQ4 extension, three paired seeds (0–2):** corrupted-test averages across 20 conditions:

| Training group | Model | mAP | Frozen macro-F1 |
|---|---|---:|---:|
| A — clean-only | RGB-only R18 | 76.05% | 61.37% |
| B — clean-only | R18 + GT geometry | 75.98% | 61.91% |
| C — corruption-aware | RGB-only R18 | 79.78% | 73.42% |
| D — corruption-aware | R18 + GT geometry | 79.82% | 73.55% |
| E — clean-only + RGB masking | RGB-only R18 | 75.80% | 61.28% |
| F — clean-only + RGB masking | R18 + GT geometry | 75.97% | 61.13% |

Corruption-aware training improves **both** RGB and Oracle robustness markedly, without a stable geometry-specific mAP advantage across seeds. The masking intervention does not establish a consistent geometry-specific benefit across both metrics. These are empirical observations, not proof that landmarks contain no useful information or a unique failure mechanism. Clean frozen-F1 remains approximately **75.45%–75.65%** across A–F; it must not be confused with the corrupted average.

**[Read the complete RQ4 summary and download its five source CSVs](docs/results/rq4/README.md).**

## RQ5 nonlinear fusion and auxiliary supervision

The nine new RQ5 runs are **N0, N1 and MTL × seeds 0–2**. The linear RGB baseline **A** consists of three *reused* RQ4 runs, not additional RQ5 training. The primary paired comparisons are **N1−N0** (increment from predicted landmarks with matched nonlinear heads) and **MTL−A** (increment from auxiliary landmark supervision over the historical linear RGB baseline).

Corrupted-test averages across 20 non-clean conditions (three-seed mean ± sample SD, %):

| Model | Frozen macro-F1 | mAP |
|---|---:|---:|
| A — linear RGB control, reused RQ4 | 61.369 ± 0.047 | 76.048 ± 0.226 |
| N0 — nonlinear RGB control | 61.375 ± 0.446 | 76.016 ± 0.122 |
| N1 — predicted geometry, nonlinear fusion | 61.090 ± 0.092 | 76.022 ± 0.252 |
| MTL — auxiliary landmark regression | 61.836 ± 0.648 | 75.565 ± 0.545 |

Paired effects: **N1−N0 = +0.006 ± 0.226 pp mAP / −0.286 ± 0.442 pp frozen F1**; **MTL−A = −0.484 ± 0.358 pp mAP / +0.467 ± 0.627 pp frozen F1**. Neither intervention establishes an improvement consistent across both metrics and all seeds. MTL−A uses an explicitly identified historical control, and later-stage results are exploratory follow-ups on a previously inspected test benchmark.

**[RQ5 results, scripts, figure source records and verification limits](docs/results/rq5/README.md)** · **[Figure-to-paper index](docs/paper_figures.md)**

## Repository guide

| Directory | Purpose |
|---|---|
| [`models/`](models/) | R18, R50, R18-geometry PyTorch model definitions |
| [`datasets/`](datasets/) | CelebA loading, shared preprocessing, corruption transforms and landmark cache interfaces |
| [`detectors/`](detectors/) | MTCNN detector wrapper |
| [`evaluation/`](evaluation/) | Attribute and benchmark metrics |
| [`configs/`](configs/) | Attribute ordering, benchmark protocols and per-run thresholds |
| [`notebooks/`](notebooks/) | Development records, multi-seed experiments and RQ4 runner; RQ5 evidence has its [own archive](docs/results/rq5/README.md); [start at notebook index](notebooks/README.md) |
| [`scripts/`](scripts/) | Evaluation, reproduction, comparison and release utilities |
| [`docs/results/`](docs/results/) | Frozen research evidence, experimental reports, figures and audited outputs |
| [`docs/releases/research_v1/`](docs/releases/research_v1/) | Preserved prior research snapshot and evidence manifest |
| [`docs/hci/`](docs/hci/) | Research-to-HCI handoff notes; no public hosted demo is asserted here |


## Reproducibility and setup

The original studies rely on CelebA images, archived trained checkpoints and prediction shards **stored separately** (including Google Drive). They are not committed to GitHub. This repository contains source code and a selected auditable research record; cloning it alone does **not** reproduce every archived result.

For a local environment (check Python/PyTorch compatibility with your machine):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

The command above installs the base local environment. For attribute metrics, use `python -m pip install -r requirements-metrics.txt`; for MTCNN plus metrics, use `python -m pip install -r requirements-detector.txt`. See [local dependency scope](docs/local_dependencies.md). These are local convenience profiles, not replacements for archived training environments.

The research input interface uses centered 224×224 letterboxing, ImageNet normalization and five GT/predicted facial points mapped to ten normalized coordinates. Attribute ordering lives in [`configs/attributes.yaml`](configs/attributes.yaml). Use the archived [training](docs/training_protocol_v1.md) and [benchmark](docs/benchmark_protocol_v1.md) protocols before claiming an exact reproduction. The local requirements file is **not** a complete frozen Colab training environment.

For historical, raw experiment evidence and provenance, see [experiment log](docs/experiment_log.md), [prior release](docs/releases/research_v1/README.md), [four-seed analyses](docs/results/benchmark/four_seed_and_remake_comparison/README.md) [RQ4](docs/results/rq4/README.md) and [RQ5](docs/results/rq5/README.md).

## Interpretation and scope

- **Clean vs corrupted:** aggregate corrupted metrics exclude the clean test condition.
- **Oracle:** aligned GT landmarks are privileged input; they are not available for arbitrary uploaded portraits.
- **NME associations:** low localization error alone does not imply better attribute F1 or identify a causal failure mechanism.
- **Uncertainty:** seed sample standard deviations are not confidence intervals; historical execution environments are not identical.
- **RQ4:** interventions test training distribution and RGB masking, but do not isolate the information content of all possible geometry representations.
- **RQ5:** N1−N0 tests adding predicted geometry under matched nonlinear classifier dimensions; MTL−A uses a reused RQ4 control. These experiments do not isolate error-free fusion capacity or establish equivalence.
- **Artifacts:** preserve source and hash-linked experiment snapshots; avoid casually deleting apparent duplicates under `docs/results/`.

This README is a navigation entrypoint to research evidence, not a replacement for the frozen protocols and recorded experiment limitations.
