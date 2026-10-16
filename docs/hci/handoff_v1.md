# Facial Visual Profile: research-to-HCI handoff v1

## 1. Read this first

This is a self-contained handoff for a NEW working conversation. Respond to the user in Simplified Chinese; use English for code, comments, runtime messages and repository technical documentation.

The user has finished the planned research-experiment phase and explicitly wants to move to HCI/demo development. Do not expand training, seeds, datasets, ablations or statistical analysis by default. Future research ideas are optional, not blockers. Research release v1 was committed and pushed to main as `6980d5e` (confirmed by the user's terminal output). This handoff is newer than older current_state_v1.md copies.

Do not claim the paper manuscript is already written or that a user study has been completed. Experiment/evidence preparation is complete at the agreed scope; manuscript polishing can happen later. Minor narrative/documentation items do not block the demo.

Next task: calendar-plan October 26, build a Gradio single-image attribute-inference prototype. Dates in the plan are milestone labels, not evidence that those calendar dates have already elapsed. Do not confuse them with historical development names such as Day18 or Day21.

## 2. Project and product goal

Project: Facial Visual Profile. Research title: Landmark-Guided Deep Learning for Robust Facial Attribute Analysis.

Research question: does adding explicit five-point geometry improve facial-attribute prediction and robustness relative to RGB-only depth? The application/portfolio demonstrates model inference, comparison and an understandable interface. Later, structured attribute evidence may be converted into grounded natural-language descriptions. First implement reliable image-to-attribute inference; an LLM is not required for October 26.

The intended demo comparison is R18, R50 and M3. Oracle needs privileged ground-truth points and is a research reference, not an ordinary user-upload deployment model. Never invent GT points for an uploaded image.

## 3. What is already finished

- Four models x three training seeds (0/1/2), ten epochs each; clean-validation checkpoint selection, prediction export and frozen attribute thresholds.
- Official CelebA test set: 19962 images, 21 frozen conditions, all twelve runs. 252 overall metric rows; full per-attribute evidence exists.
- Separate F1(frozen), F1@0.5 and mAP summaries; mean/sample SD, paired model differences and attribute heatmaps.
- Detector-NME subgroup analysis and 1000-replicate paired image-bootstrap F1 intervals. Intervals are pointwise/unadjusted and conditional on fixed trained models; training-seed uncertainty is excluded. No mAP confidence intervals were calculated.
- Qualitative case review: 24 different original images, four conditions, six strata per condition, 288 selected-attribute prediction records. Images, transformed GT and archived predictions are linked and checked.
- Fresh Colab CPU replay: Seed1 four models x first eight official test images x four conditions = 128 model-sample evaluations. All passed declared numeric tolerances, zero decision changes for frozen and 0.5 rules. Largest printed logit difference approximately 1.31e-5. MTCNN was NOT rerun.
- Research release v1 verified 193 input files and 252/252 overall rows against reports; four release documents plus assembly script pushed as `6980d5e`.

Calendar milestones: Oct21 complete replicated evaluation; Oct22 summarize overall/24-attribute results; Oct23 inspect successes/failures and landmark errors; Oct24 replay small inference in a fresh environment; Oct25 archive evidence as research release v1. Detailed minority-attribute prose, complete case narrative and actual monetary spending remain documentation items, not requirements for additional experiments.

Main supported conclusion: no consistent universal benefit from the current simple predicted-five-point fusion was established. Gains depend on attribute, corruption, threshold rule and training run. Do not describe M3 as proven superior or claim R18 explicitly learned landmark localization. Oracle is independently trained, not a guaranteed upper bound or a causal intervention on M3.

## 4. Model interfaces and decision rules

| Model | Architecture | Archived class | Checkpoint state key for Seed1 |
|---|---|---|---|
| R18 | ImageNet ResNet18, 512 -> 24 | ResNet18Attributes | model_state |
| R50 | ImageNet ResNet50, 2048 -> 24 | ResNet50Attributes | model_state |
| M3 | R18 512 + geometry MLP 10 -> 32 -> 32; concat 544 -> 24 | ResNet18GeometryAttributes | model_state_dict |
| Oracle | Same fusion design, independently trained with GT | ResNet18GeometryAttributes | model_state_dict |

The RGB forward interface is model(images). M3 is model(images, normalized_points, geometry_valid). Geometry points are five (x,y) pairs flattened to ten; order: left eye, right eye, nose, left mouth, right mouth. M3 has 11191000 parameters. All models return logits, not thresholded labels.

Use eval(), torch.inference_mode(), FP32 and strict state loading. Initialize constructors with pretrained=False when loading trained checkpoints; do not download ImageNet weights again. torch.load uses weights_only=True; archived checkpoints may require the scoped TorchVersion safe-global allowlist.

Compute stable float64 sigmoid scores from logits. Call them model scores, not demonstrated calibrated probabilities. Each run has its OWN 24 clean-validation frozen thresholds, with comparison >=. A 0.5 view may be added but must be labeled separately. Never fit thresholds on user uploads or reuse another run's vector. Verify attribute order and checkpoint/source/threshold hashes before use.

Missing M3 geometry was handled during training by zeroing invalid coordinate inputs before the MLP and masking geometry features AFTER the MLP (including biases). Retain this model contract. For the interactive product, the schedule explicitly asks to show M3/Dual as unavailable on landmark failure; do not silently present that as a normal geometry-backed result. RGB results may still be shown with a clear detection status. Keep the experiment masking policy and UI availability policy distinct.

## 5. Exact 24-attribute order

```text
Smiling
Mouth_Slightly_Open
Eyeglasses
Bangs
Wearing_Hat
Mustache
No_Beard
Black_Hair
Bushy_Eyebrows
Arched_Eyebrows
High_Cheekbones
Bags_Under_Eyes
Big_Lips
Big_Nose
Double_Chin
Oval_Face
Receding_Hairline
Blond_Hair
Brown_Hair
Gray_Hair
Bald
Straight_Hair
Wavy_Hair
Sideburns
```

CelebA official split counts: train 162770, validation 19867, test 19962. Original -1/+1 attribute labels were converted to 0/1. Human-readable result text should mean predicted attributes, not an unquestionable factual description. Case review found potential label ambiguity, not a measured dataset label-error rate. Original labels and formal metrics were not changed.

## 6. Shared image and detector pipeline: preserve it

- Convert to RGB; preserve aspect ratio; resize longest side to 224 using Pillow BILINEAR with Python round for dimensions; center on a black 224x224 letterbox canvas.
- Pixel-center coordinate mapping: (coord + 0.5) * resized_dimension / original_dimension - 0.5 + padding.
- Normalize point coordinates as 2*coord/223 - 1 and flatten to ten values.
- Normalize image RGB using ImageNet mean (0.485,0.456,0.406), std (0.229,0.224,0.225).
- Frozen functions are datasets/benchmark_preprocessing.py (prepare_canvas, normalize_canvas, normalize_points), datasets/preprocessing.py and datasets/corruptions.py inside the archived frozen-input ZIP.
- Formal benchmark detection runs on the FINAL 224x224 canvas, AFTER the selected corruption and BEFORE RGB normalization. A new clean canvas must also be detected in that pipeline.
- Historical training/validation detected the original aligned image then mapped points. Do not confuse that historical cache pipeline with demo/final-canvas detection.
- Uploaded/changed images require actual detector execution for M3. Do not pull a random CelebA cached detection, reuse stale points after an image/control change, or substitute GT.
- Do not introduce automatic face cropping or alignment silently: it changes the shared pipeline. CelebA used aligned faces, so arbitrary uploads can be outside the evaluated distribution. Any new crop/EXIF/orientation behavior must be explicit and reproducible.

Frozen perturbations: blur sigma 1/2/3/4; brightness factors 0.8/0.6/0.4/0.2; rotation magnitudes 5/10/15/20 degrees with deterministic signed directions; lower-face square occlusion side fractions 0.1/0.2/0.3/0.4; JPEG quality 75/50/25/10. Each starts from the clean canvas. Preserve corruption VERSION=day14_draft_v1: it participates in the rotation-sign hash. Rotate GT only in research reproduction, not because real user uploads have GT.

## 7. Locations: GitHub, Mac, Drive and temporary runtimes

GitHub remote: DeweyYip/facial-visual-profile, branch main. User-confirmed latest release commit: 6980d5e. SSH remote uses ssh.github.com port 443; the user previously unlocked their key using ssh-add --apple-use-keychain ~/.ssh/id_ed25519. Never request, upload or package the private key.

Mac project root:
`/Users/deweye/Desktop/Facial Visual Profile`

Mac download directory:
`/Users/deweye/Downloads`

Colab Drive project root:
`/content/drive/MyDrive/FacialVisualProfile`

These are DIFFERENT filesystems. A new assistant does not automatically see the Mac checkout, browser account, Drive session, checkpoints or previous scratch files. A path in this document is a locator, not proof the assistant can read it. The user runs Mac terminal and Colab code and shares outputs/files.

| Storage | Contents | Role |
|---|---|---|
| GitHub / Mac checkout | Python source, notebooks, configs, result JSON/CSV/figures, provenance, release index | Versioned code and compact evidence |
| Drive data/ | img_align_celeba.zip, list_attr_celeba.txt, list_eval_partition.txt, list_landmarks_align_celeba.txt | Original dataset and official annotations |
| Drive runs/{model}_seed{0,1,2}/ | config.json, best.json, source/, checkpoint .pt files, history and validation predictions | Authoritative trained model runs |
| Drive runs/benchmark_preparation/seed1/ and seed2/ | {run_name}.json | Frozen per-run thresholds |
| Drive cache/mtcnn_clean.sqlite | Historical clean detections for CelebA | Not a detector for arbitrary user uploads |
| Drive runs/benchmark/day19/shards/ | Seed0 RGB/M3 clean, blur, brightness | Prediction batches |
| Drive runs/benchmark/day20/shards/ | Seed0 RGB/M3 rotation, occlusion, JPEG | Prediction batches |
| Drive runs/benchmark/day21/shards/ | Seed0 Oracle, all 21 conditions | Oracle prediction batches |
| Drive runs/benchmark/multiseed_seed1_seed2/shards/ | Eight Seed1/2 runs, all 21 conditions | Formal later-seed predictions |
| Drive runs/benchmark/prediction_case_review_v1/ | 24-case review, figures, source, archive | Qualitative evidence |
| Drive runs/benchmark/small_scale_reproduction/20261016T155816211904Z/ | Completed CPU reproduction and ZIP | Verified replay evidence |
| Colab /content/ outside drive | Copied source, extracted images, caches, runtime variables | Temporary; can vanish on reset |
| Chat workspace scratch | Copies of uploads/generated code in the OLD chat | Not a reliable transfer mechanism; upload the new handoff bundle |

Mac may contain a small runs/ directory or archive ZIPs; do not assume it contains the Drive checkpoints. Earlier Mac inventory did not find the twelve canonical trained-run directories. HCI model assets still need an explicit Drive-to-demo export/verification step.

## 8. Recommended starting model assets (proposal, not a completed export)

Use a fixed Seed1 deployment set to reuse the already verified CPU loader. Start October26 with R18; add R50 and M3 in October27. This is a reproducibility/engineering choice, not test-based winner selection. Confirm the deployment choice with current project context if needed; do not silently change run/seed.

All three Seed1 selected checkpoints are epoch_03.pt:

| Run | Drive checkpoint relative to project root | Frozen thresholds |
|---|---|---|
| r18_seed1 | runs/r18_seed1/epoch_03.pt | runs/benchmark_preparation/seed1/r18_seed1.json |
| r50_seed1 | runs/r50_seed1/epoch_03.pt | runs/benchmark_preparation/seed1/r50_seed1.json |
| m3_seed1 | runs/m3_seed1/epoch_03.pt | runs/benchmark_preparation/seed1/m3_seed1.json |

Checkpoint SHA256:
```text
r18_seed1 c5168e8c5b50146e25148804d75ff77096ec2613f3ee5ccc26aac7526182feb5
r50_seed1 361664364df2bd44a12a550b71a5821f89c7fe68f824caf0b5583389ab1e4d4f
m3_seed1  6737a5d74f07e395b9d98da24ff7932c2540c401ef611506f6d628a63f709ed8
```

Use runs/{run_name}/config.json, best.json and source/models/*.py as the corresponding source of truth. Complete config/source/checkpoint/threshold hashes and state-key mappings are in:
`docs/results/benchmark/multiseed_seed1_seed2/execution_binding.json`.

Do not bundle the whole original dataset or all shards for the demo. A minimal deployment transfer needs the selected checkpoint(s), config, best record, source snapshots, threshold JSONs, frozen shared preprocessing, detector wrapper/weights as required, and a small permitted smoke-test fixture. Actual runtime behavior on the Mac has not been tested yet. The checkpoint files include optimizer state and are larger than inference-only weights; any later inference-only export must retain a manifest linking it to the original verified checkpoint.

## 9. Verified environments and current unknowns

Training Seed1: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130, numpy 2.1.3, Pillow 11.3.0, scikit-learn 1.6.1, Tesla T4. Seed0 has recorded cu128 differences.

Completed fresh replay: Python 3.13.16, torch 2.11.0+cpu, torchvision 0.26.0+cpu, numpy 2.1.3, Pillow 11.3.0, scikit-learn 1.6.1. Replay batch 8 versus original test batch16. No changed decisions on the fixed subset. This does not validate every device, input or model condition.

The latest Mac runtime, CPU architecture, free memory, Gradio version, installed detector dependencies and local model-file availability are UNKNOWN. Existing terminal traceback once showed Python3.10; do not assume that is still the chosen environment. Inspect first. Do not upgrade the existing research virtual environment blindly. A separate demo environment can be created if needed.

No Gradio demo has been implemented or launched in this conversation. No service, API budget, public hosting, user study or LLM has been selected. Do not claim otherwise.

## 10. Evidence and reusable entry points

Research entry: docs/releases/research_v1/README.md; summary_metrics.csv; evidence_manifest.json; release_audit.json.

Results under docs/results/benchmark/:
- day19/, day20/, day21/: original Seed0 evidence.
- multiseed_seed1_seed2/: later-seed report, execution binding, source/frozen inputs.
- three_seed_comparison/: 252-row overall results and curves/differences.
- three_seed_nme_and_f1_intervals/: subgroups, image intervals, paired audit.
- three_seed_attribute_comparison/: attribute comparisons and heatmaps.
- prediction_case_review/: 24 selected cases and prediction records.
- small_scale_reproduction/: completed CPU replay.

Notebooks:
- multiseed_training_and_validation.ipynb: historical Seed1 execution, includes recovery/history; not the preferred clean starting point.
- multiseed_training_and_validation_seed2.ipynb: reusable training/validation/threshold workflow; no need to run it for HCI.
- multiseed_frozen_test_benchmark.ipynb: formal test with saved-shard recovery.
- three_seed_nme_and_f1_intervals.ipynb: archived-prediction analysis only.
- three_seed_prediction_case_review.ipynb: qualitative rendering; not model inference.
- small_scale_inference_reproduction.ipynb: verified model loading and real inference; most useful demo-backend reference.

CPU replay runner: docs/results/benchmark/small_scale_reproduction/source/reproduce_small_inference.py.
Frozen preprocessing/detector source ZIP: docs/results/benchmark/multiseed_seed1_seed2/source/day19_frozen_inputs.zip.

## 11. Known issues and preservation rules

- Original case-review analysis_binding.json field conditions accidentally contains all 19962 official image IDs, due to a variable-name collision. Actual four conditions and results are correct and recorded in completion_audit.json and caches. The original archive and hashes are preserved, with a local-manifest note. Fix only a new code version/output if reusing it; no need to rerun cases to start HCI.
- best.json schemas differ: some contain checkpoint; Oracle may contain only epoch and validation metrics. Resolve checkpoint with best.get('checkpoint', f"epoch_{best['epoch']:02d}.pt") and verify against the execution binding. Do not assume all schemas are identical.
- Original ZIP schemas differ: Seed0 Oracle has only manifest.json and predictions.npz; other prediction shards have records.json too. Do not conflate them. Demo inference does not need the full shard collection.
- Some CelebA labels are potentially ambiguous. Example Black_Hair 199481.jpg has target=0 despite visually dark hair; correct/wrong in case tables means agreement/disagreement with dataset labels. Do not change labels or formal results based on a few selected cases.
- User's old tracked edits must be preserved. Earlier paths include models/resnet18_geometry.py, models/resnet50.py, notebooks/day18_oracle_seed0.ipynb, notebooks/day19_clean_blur_brightness_benchmark.ipynb, scripts/day17_freeze_benchmark.py, scripts/day17_metrics_smoke.py, scripts/day19_evaluate_benchmark.py. Release assembly recorded EIGHT dirty tracked paths; the complete current list is in evidence_manifest.json and should be refreshed, not guessed.
- Do not use git add . or git commit -a. Stage/commit explicit new demo paths only. No cleanup/reset/checkout of unrelated edits. A dirty working tree does not invalidate verified historical run snapshots.
- Do not alter the frozen research release to make demo results look better. Add new HCI code/config/logs separately.
- Monetary compute cost was not measured automatically. Unmeasured does not mean zero; it does not block HCI.

## 12. HCI implementation plan and acceptance

| Plan milestone | Work | Acceptance |
|---|---|---|
| Oct26 | Gradio upload, verified model load, shared preprocessing, 24 labels | Upload one image and receive real attribute scores/decisions |
| Oct27 | R18/R50/M3 comparison, per-attribute scores, five-point overlay | Compare three models; no Oracle/GT requirement |
| Oct28 | Five corruption controls with frozen transforms | Recompute final image and detector when input/settings change; no stale result reuse |
| Oct29 | No face, multiple faces, invalid keypoints, model load errors | Explicit status; M3 unavailable when geometry cannot be supplied |
| Oct30 | Define structured language input and permissible factual descriptions; templates; 60 development cases | Basic explanation works without an LLM |
| Oct31 | Trial planned local Qwen3-1.7B/0.6B options if appropriate | Measure quality, latency, memory and Intel compatibility; not yet done |
| Nov1 | Choose template/local/API language layer and output checks | Usable first product version |
| Nov2 | Check grounding on 20 reserved cases, separate from development examples | Language-faithfulness record; not a completed human-user study |

Start with an engineering MVP, not an unrequested redesign or expanded research program. Product UI should explain relevant scores/status without flooding users with source hashes and implementation details. Keep provenance in developer logs/config, not the main user flow. User images should not be persisted or made public by default; local access is the starting point. Public sharing must be an explicit later choice.

First new-chat actions:
1. Read this handoff and the actual bundled release README/binding/replay runner.
2. Inspect Mac runtime and local model availability with one small read-only command; determine local versus Colab execution from actual constraints.
3. Reuse verified archived-source loaders/preprocessing. If local checkpoints are absent, supply a minimal verified Drive export step.
4. Implement Oct26 end-to-end upload -> actual inference -> 24 scores/decisions, with a reproducible launch command and simple smoke check.
5. Save new code and run instructions to explicit repository paths; guide a selective Git commit once working. Do not stop at a plan if inputs are sufficient.

## 13. What the new chat should receive

Preferred initial upload: the small hci_handoff_bundle ZIP produced by build_hci_handoff_bundle.py from the current Mac checkout. It contains this handoff, actual release files, execution binding, frozen preprocessing ZIP, replay runner/notebook, relevant concise result tables, and clearly labeled current source candidates. It has a file-hash manifest and local/remote Git state metadata. It does NOT contain checkpoints, full data/shards, virtual environments, SSH keys or user-private files.

The new assistant can understand the project without all old chat messages or the old long proposal. The original proposal is optional background only; if it conflicts with completed evidence, the evidence and this handoff take precedence. Upload model assets only when the implementation step needs them; a new chat cannot access Drive merely because a locator is written here.
