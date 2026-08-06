import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path

def compute_metrics(true_labels: np.ndarray, pred_scores: np.ndarray) -> dict:
    """Calculate common regression and classification metrics.

    Args:
        true_labels: Ground‑truth binary or continuous values.
        pred_scores: Predicted probability or continuous scores.
    Returns:
        Dictionary with MAE, RMSE, ROC‑AUC (if binary), and calibration data.
    """
    metrics = {}
    # Regression metrics (useful for continuous risk scores)
    metrics["mae"] = float(mean_absolute_error(true_labels, pred_scores))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(true_labels, pred_scores)))

    # Classification metrics – only compute if labels are binary (0/1)
    if set(np.unique(true_labels)).issubset({0, 1}):
        try:
            metrics["roc_auc"] = float(roc_auc_score(true_labels, pred_scores))
            precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
            metrics["pr_auc"] = float(auc(recall, precision))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics

def calibration_curve(true_labels: np.ndarray, pred_scores: np.ndarray, n_bins: int = 10) -> dict:
    """Generate calibration data for plotting.

    Returns a dict with ``bin_centers`` and ``fraction_of_positives`` which can be
    directly fed to a plotting routine.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(pred_scores, bins) - 1
    bin_sums = np.bincount(binids, weights=pred_scores, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_pred = np.where(bin_total > 0, bin_sums / bin_total, 0)
    # Actual fraction of positives per bin
    pos_sums = np.bincount(binids, weights=true_labels, minlength=n_bins)
    frac_pos = np.where(bin_total > 0, pos_sums / bin_total, 0)
    centers = (bins[:-1] + bins[1:]) / 2
    return {"bin_centers": centers.tolist(), "fraction_of_positives": frac_pos.tolist(), "avg_pred": avg_pred.tolist()}

def save_calibration_plot(calibration_data: dict, output_path: Path) -> None:
    """Create and save a calibration (reliability) plot.

    The function expects the dict produced by :func:`calibration_curve`.
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(
        calibration_data["bin_centers"],
        calibration_data["fraction_of_positives"],
        "s-",
        label="Model",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
