"""Frozen benchmark metrics; leave historical training metrics unchanged."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def probabilities_and_targets(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    if logits.ndim != 2 or logits.shape != targets.shape:
        raise ValueError("Logits and targets must have matching [N,24] shapes")
    if logits.shape[0] == 0 or logits.shape[1] != 24:
        raise ValueError("Require at least one sample and exactly 24 attributes")
    if not np.isfinite(logits).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Require finite logits and binary targets")
    if any(np.unique(targets[:, j]).size != 2 for j in range(24)):
        raise ValueError("Every attribute must contain both target classes")
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result, targets.astype(np.int64)


def evaluate_benchmark_attributes(logits, targets, thresholds):
    probabilities, targets = probabilities_and_targets(logits, targets)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (24,) or not np.isfinite(thresholds).all():
        raise ValueError("Require exactly 24 finite thresholds")
    if not ((thresholds > 0) & (thresholds < 1)).all():
        raise ValueError("Thresholds must lie strictly between zero and one")
    f1 = f1_score(targets, probabilities >= thresholds, average=None,
                  zero_division=0)
    ap = [float(average_precision_score(targets[:, j], probabilities[:, j]))
          for j in range(24)]
    return {"thresholds": thresholds.tolist(), "sample_count": len(targets),
            "macro_f1": float(np.mean(f1)), "map": float(np.mean(ap)),
            "per_attribute_f1": f1.tolist(), "per_attribute_ap": ap,
            "positive_counts": targets.sum(axis=0).tolist()}
