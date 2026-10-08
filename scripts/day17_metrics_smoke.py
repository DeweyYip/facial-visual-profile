"""Check metrics on synthetic data only; no CelebA access."""
from pathlib import Path
import importlib.util
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load("evaluation/benchmark_attributes.py", "benchmark_metrics")
rng = np.random.default_rng(17)
logits = rng.normal(size=(80, 24))
targets = rng.integers(0, 2, size=(80, 24))
targets[0], targets[1] = 0, 1
probabilities, _ = new.probabilities_and_targets(logits, targets)
thresholds = np.linspace(.1, .9, 24)
assert not hasattr(new, "select_validation_thresholds")

report = new.evaluate_benchmark_attributes(logits, targets, thresholds)
expected_f1 = f1_score(targets, probabilities >= thresholds,
                       average=None, zero_division=0)
expected_ap = [average_precision_score(targets[:, j], probabilities[:, j])
               for j in range(24)]
np.testing.assert_allclose(report["per_attribute_f1"], expected_f1)
np.testing.assert_allclose(report["per_attribute_ap"], expected_ap)

perfect_targets = np.tile([[0], [1]], (2, 24))
perfect_logits = np.where(perfect_targets == 1, 1000.0, -1000.0)
perfect = new.evaluate_benchmark_attributes(
    perfect_logits, perfect_targets, [0.5] * 24)
assert perfect["macro_f1"] == perfect["map"] == 1.0
print("Independent F1/AP and extreme-logit sigmoid: PASS")

print("No threshold fitting performed")

historical = ROOT / "evaluation/attributes.py"
if historical.exists():
    old = load("evaluation/attributes.py", "historical_metrics")
    previous = old.evaluate_attributes(logits, targets, threshold=0.5)
    current = new.evaluate_benchmark_attributes(logits, targets, [0.5] * 24)
    for key in ("macro_f1", "map", "per_attribute_f1", "per_attribute_ap"):
        np.testing.assert_allclose(previous[key], current[key], rtol=0, atol=0)
    print("Historical threshold-0.5 metric equivalence: PASS")
print("Day 17 synthetic metrics smoke: PASS")
