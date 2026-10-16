# Small-scale inference reproduction

Seed1, four models, first eight official test images and four conditions: 128 model-sample evaluations.
Model checkpoints, archived source, frozen thresholds, JPEG bytes, final RGB and geometry inputs are verified.
MTCNN outputs are reused; this does not validate a fresh detector execution or the full 21-condition benchmark.
FP32 reference logits were generated on GPU with a different batch size; replay hardware is recorded.
Tolerances are declared in execution_binding.json before model inference. Every threshold decision is compared.
An execution-complete report may still require review if numeric tolerances fail or any decision changes.
No performance metrics or model ranking should be inferred from this small subset.
Runtime is measured; monetary cost must come from actual billing records and is not estimated here.
