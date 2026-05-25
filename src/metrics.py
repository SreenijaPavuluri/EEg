"""
Evaluation metrics for MI EEG classification.
Includes accuracy, balanced accuracy, F1, kappa, Brier score, ECE.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, brier_score_loss, roc_auc_score
)


def expected_calibration_error(y_true, probs, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence and measures mean |accuracy - confidence|.
    """
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = (y_pred[mask] == y_true[mask]).mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)

    return float(ece)


def compute_all_metrics(y_true, y_pred, probs, classes=None):
    """
    Compute all evaluation metrics for a single fold.
    Returns a dict.
    """
    if classes is None:
        classes = np.unique(y_true)

    acc = accuracy_score(y_true, y_pred)
    balacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    ece = expected_calibration_error(y_true, probs)

    # Brier score (only well-defined for binary)
    if len(classes) == 2 and probs.shape[1] == 2:
        # Convert y_true to binary
        label_map = {c: i for i, c in enumerate(classes)}
        y_bin = np.array([label_map[c] for c in y_true])
        brier = brier_score_loss(y_bin, probs[:, 1])
    else:
        brier = np.nan

    # AUROC (binary only)
    if len(classes) == 2 and probs.shape[1] == 2:
        label_map = {c: i for i, c in enumerate(classes)}
        y_bin = np.array([label_map[c] for c in y_true])
        auroc = roc_auc_score(y_bin, probs[:, 1])
    else:
        auroc = np.nan

    return {
        'accuracy': acc,
        'balanced_accuracy': balacc,
        'f1_weighted': f1,
        'cohen_kappa': kappa,
        'ece': ece,
        'brier_score': brier,
        'auroc': auroc,
        'n_test': len(y_true),
    }


def reliability_diagram_data(y_true, probs, n_bins=10):
    """
    Return data for a reliability diagram (calibration curve).
    Returns (mean_confidence_per_bin, fraction_correct_per_bin, bin_counts).
    """
    confidences = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    mean_conf = []
    frac_pos = []
    counts = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            mean_conf.append(np.nan)
            frac_pos.append(np.nan)
            counts.append(0)
        else:
            mean_conf.append(confidences[mask].mean())
            frac_pos.append((y_pred[mask] == y_true[mask]).mean())
            counts.append(int(mask.sum()))

    return np.array(mean_conf), np.array(frac_pos), np.array(counts)
