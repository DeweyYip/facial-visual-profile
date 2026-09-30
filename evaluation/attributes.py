"""Validation metrics for multi-label facial attributes."""
"""Applies to R18, R50, Predicted Geometry, and Oracle Geometry to make sure everything's fair"""

"""
validation mAP, macro-F1 are all calculated here
we input raw logits and real labels here to compute the scores needed for evaluation
logits -> sigmoid -> probability -> F1, AP -> macro-F1, mAP
"""


"""
Notice no pyTorch is used here
"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

"""
input: 
1. raw logits before sigmoid (shape: [19867, 24] (there're 19867 images in the validation set))
2. targets: the correcot answer of each image 
3. threshold used for calculate the final answer: eg.  0.89 > 0.5 => 1; 0.36 < 0.5 => 0
"""
def evaluate_attributes(logits, targets, threshold=0.5):
    
    # Transform the input into NumPy array
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)

    """ To prevent the shape of our input goes wrong we need to give them a check
        ideal shape: logits.shape = (19867, 24); targets.shape = (19867, 23)
        and target should only contain 0 and 1 
    """
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

    # we manually define a sigmoid function here
    """
    logits: [2.0, -1.0, 0.3]
        ↓ sigmoid
    probabilities: [0.88, 0.27, 0.57]
    """
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (
        1.0 + np.exp(-logits[positive])
    )
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)

    # From probability to 1 and 0
    predictions = probabilities >= threshold

    # F1
    """
    There will be 24 F1 in total:
    Bushy_Eyebrows     F1 = 0.76
    Arched_Eyebrows    F1 = 0.71
    ...
    Oval_Face          F1 = 0.70
    """
    per_attribute_f1 = f1_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )

    # AP
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
    
    """
    there is this attribute and you predict that there is：4 个      TP = 4
    there isn't this attribute but you predict that there is：1 个      FP = 1
    there is this attribute but you didn't predict that there is：2 个      FN = 2

    Precision = TP / (TP + FP)
        = 4 / (4 + 1)
        = 0.80
    
    Recall = TP / (TP + FN)
       = 4 / (4 + 2)
       = 0.667

    F1 = 2 × Precision × Recall
     ----------------------
       Precision + Recall
    
    this is how we get a F1 for an attribute

    The way we get AP is:
    order the image by probability
    predicted value for Big_Nose     real value
        0.95                             1
        0.80                             0
        0.70                             1
        0.60                             1
        0.20                             0
    among the top 1 values：1 predicted: Precision = 1/1 = 1.00
    among the top 4 values：3 predicted: = 3/4 = 0.75
    AP ≈ (1.00 + 0.667 + 0.75) / 3 ≈ 0.806
    """

    return {
        "threshold": float(threshold),
        "macro_f1": float(np.mean(per_attribute_f1)),
        "map": float(np.mean(per_attribute_ap)),
        "per_attribute_f1": per_attribute_f1.tolist(),
        "per_attribute_ap": per_attribute_ap,
        "positive_counts": targets.sum(axis=0).astype(int).tolist(),
    }
