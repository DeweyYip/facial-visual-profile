"""Validation metrics for multi-label facial attributes."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def evaluate_attributes(logits, targets, threshold=0.5):
    """Compute macro F1 and mAP from all validation samples.

    Args:
        logits: Array with shape (samples, attributes), before sigmoid.
        targets: Binary array with the same shape.
        threshold: Probability threshold used for F1 only.

    Returns:
        A dictionary containing aggregate and per-attribute metrics.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)

    if logits.ndim != 2 or logits.shape != targets.shape:
        raise ValueError("Logits and targets must have the same 2D shape")
    if logits.shape[0] == 0:
        raise ValueError("At least one sample is required")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain non-finite values")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between zero and one")

    # The stable sigmoid form avoids overflow for large logits.
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (
        1.0 + np.exp(-logits[positive])
    )
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)

    predictions = probabilities >= threshold
    per_attribute_f1 = f1_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )

    per_attribute_ap = []
    for index in range(targets.shape[1]):
        labels = targets[:, index]
        if np.unique(labels).size < 2:
            raise ValueError(
                f"Attribute {index} has only one class in validation"
            )
        per_attribute_ap.append(
            float(average_precision_score(
                labels, probabilities[:, index]
            ))
        )

    return {
        "threshold": float(threshold),
        "macro_f1": float(np.mean(per_attribute_f1)),
        "map": float(np.mean(per_attribute_ap)),
        "per_attribute_f1": per_attribute_f1.tolist(),
        "per_attribute_ap": per_attribute_ap,
        "positive_counts": targets.sum(axis=0).astype(int).tolist(),
    }
