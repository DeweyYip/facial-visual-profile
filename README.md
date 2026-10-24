# Facial Visual Profile

**Research project:** Landmark-Guided Deep Learning for Robust Facial Attribute Analysis

Facial Visual Profile investigates whether explicit **five-point facial geometry** provides consistent value beyond RGB features for multi-label facial attribute recognition, particularly when images are corrupted. It is an experimental research repository, **not** a claim that geometry fusion universally improves robustness.

## Research at a glance

- **Task:** 24-attribute multi-label prediction on the official [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned-image splits: 162,770 train / 19,867 validation / 19,962 test.
- **Models:** ImageNet-initialized ResNet18, ResNet50, ResNet18 + MTCNN-predicted five-point geometry (**M3**), and a separately trained ResNet18 + ground-truth geometry (**Oracle**).
- **Fusion:** 512-dimensional RGB feature + 32-dimensional landmark MLP feature, concatenated and passed through a linear 24-logit classifier.
- **Evaluation:** clean input and five corruption families (blur, darkening, rotation, lower-face occlusion, JPEG compression), four severity levels each; 21 conditions per run.
- **Metrics:** mAP and macro-F1 with per-attribute thresholds fitted on clean validation and frozen for testing. F1 at a fixed 0.5 threshold is separately reported.
- **Research extension (RQ4):** tests whether corruption-aware training or RGB feature masking changes the incremental utility of **accurate** geometry.

## Key findings

**Original four-model study.** Explicit five-point fusion did **not** demonstrate a stable, universal robustness advantage over RGB-only models. Effects depend on the attribute, corruption, metric and run. The GT Oracle is a *separately trained diagnostic*, not a causal replacement of M3's detector output. See the [archived research release](docs/releases/research_v1/README.md) and [four-seed comparison](docs/results/benchmark/four_seed_and_remake_comparison/README.md). The research release describes its **three-seed snapshot**; later four-seed results must not be conflated with it.

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

## Repository guide

| Directory | Purpose |
|---|---|
| [`models/`](models/) | R18, R50, R18-geometry PyTorch model definitions |
| [`datasets/`](datasets/) | CelebA loading, shared preprocessing, corruption transforms and landmark cache interfaces |
| [`detectors/`](detectors/) | MTCNN detector wrapper |
| [`evaluation/`](evaluation/) | Attribute and benchmark metrics |
| [`configs/`](configs/) | Attribute ordering, benchmark protocols and per-run thresholds |
| [`notebooks/`](notebooks/) | Development records, multi-seed experiments, RQ4 runner; [start at notebook index](notebooks/README.md) |
| [`scripts/`](scripts/) | Evaluation, reproduction, comparison and release utilities |
| [`docs/results/`](docs/results/) | Frozen research evidence, experimental reports, figures and audited outputs |
| [`docs/releases/research_v1/`](docs/releases/research_v1/) | Preserved prior research snapshot and evidence manifest |
| [`docs/hci/`](docs/hci/) | Research-to-HCI handoff notes; no public hosted demo is asserted here |
| [`demo/`](demo/) | Placeholder for future checked-in demonstration code |

## Reproducibility and setup

The original studies rely on CelebA images, archived trained checkpoints and prediction shards **stored separately** (including Google Drive). They are not committed to GitHub. This repository contains source code and a selected auditable research record; cloning it alone does **not** reproduce every archived result.

For a local environment (check Python/PyTorch compatibility with your machine):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

The research input interface uses centered 224×224 letterboxing, ImageNet normalization and five GT/predicted facial points mapped to ten normalized coordinates. Attribute ordering lives in [`configs/attributes.yaml`](configs/attributes.yaml). Use the archived [training](docs/training_protocol_v1.md) and [benchmark](docs/benchmark_protocol_v1.md) protocols before claiming an exact reproduction. The local requirements file is **not** a complete frozen Colab training environment.

For historical, raw experiment evidence and provenance, see [experiment log](docs/experiment_log.md), [prior release](docs/releases/research_v1/README.md), [four-seed analyses](docs/results/benchmark/four_seed_and_remake_comparison/README.md) and [RQ4](docs/results/rq4/README.md).

## Interpretation and scope

- **Clean vs corrupted:** aggregate corrupted metrics exclude the clean test condition.
- **Oracle:** aligned GT landmarks are privileged input; they are not available for arbitrary uploaded portraits.
- **NME associations:** low localization error alone does not imply better attribute F1 or identify a causal failure mechanism.
- **Uncertainty:** seed sample standard deviations are not confidence intervals; historical execution environments are not identical.
- **RQ4:** interventions test training distribution and RGB masking, but do not isolate the information content of all possible geometry representations.
- **Artifacts:** preserve source and hash-linked experiment snapshots; avoid casually deleting apparent duplicates under `docs/results/`.

This README is a navigation entrypoint to research evidence, not a replacement for the frozen protocols and recorded experiment limitations.
