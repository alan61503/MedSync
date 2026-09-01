import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    matthews_corrcoef,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr


def compute_bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42):
    """Calculate non-parametric bootstrap confidence intervals for any evaluation metric."""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            if not np.isnan(score):
                bootstrapped_scores.append(score)
        except Exception:
            continue
            
    if not bootstrapped_scores:
        return 0.0, 0.0
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrapped_scores, alpha * 100))
    upper = float(np.percentile(bootstrapped_scores, (1.0 - alpha) * 100))
    return lower, upper


def compute_comprehensive_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    class_names: list = None
) -> dict:
    """Calculate a complete battery of peer-reviewed clinical classification metrics."""
    if class_names is None:
        classes = np.unique(y_true)
        class_names = [f"Class_{c}" for c in classes]
    else:
        classes = np.arange(len(class_names))
        
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1_c = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        
        per_class_metrics[cname] = {
            "sensitivity_recall": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision_ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "f1_score": round(f1_c, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }
        
    roc_auc_ovr = None
    pr_auc_dict = {}
    brier = None
    
    if y_prob is not None:
        try:
            if len(class_names) == 2:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc_auc_ovr = float(roc_auc_score(y_true, prob_pos))
                p_prec, p_rec, _ = precision_recall_curve(y_true, prob_pos)
                pr_auc_dict["binary_pr_auc"] = float(auc(p_rec, p_prec))
                brier = float(brier_score_loss(y_true, prob_pos))
            else:
                roc_auc_ovr = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                for idx, cname in enumerate(class_names):
                    y_true_binary = (y_true == idx).astype(int)
                    p_prec, p_rec, _ = precision_recall_curve(y_true_binary, y_prob[:, idx])
                    pr_auc_dict[cname] = float(auc(p_rec, p_prec))
        except Exception as e:
            print(f"ROC/PR calculation notice: {e}")
            
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "cohen_kappa": round(kappa, 4),
        "mcc": round(mcc, 4),
        "roc_auc_ovr": round(roc_auc_ovr, 4) if roc_auc_ovr is not None else None,
        "pr_auc": pr_auc_dict,
        "brier_score": round(brier, 4) if brier is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate clinical regression agreement metrics for continuous measurements (BMD, T-score)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    r_pearson, p_pearson = pearsonr(y_true, y_pred)
    rho_spearman, p_spearman = spearmanr(y_true, y_pred)
    
    diff = y_pred - y_true
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    loa_upper = float(mean_diff + 1.96 * std_diff)
    loa_lower = float(mean_diff - 1.96 * std_diff)
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "pearson_r": round(float(r_pearson), 4),
        "pearson_pvalue": float(p_pearson),
        "spearman_rho": round(float(rho_spearman), 4),
        "spearman_pvalue": float(p_spearman),
        "bland_altman": {
            "mean_bias": round(mean_diff, 4),
            "std_diff": round(std_diff, 4),
            "loa_95_upper": round(loa_upper, 4),
            "loa_95_lower": round(loa_lower, 4),
        }
    }


def plot_publication_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Render publication-grade annotated confusion matrix heatmap."""
    plt.figure(figsize=(7, 6), dpi=300)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Multi-Class Diagnostic Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11, rotation=25)
    plt.yticks(tick_marks, class_names, fontsize=11)
    
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            plt.text(
                j, i, val_str,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold"
            )
            
    plt.ylabel("Ground Truth (Clinical Diagnosis)", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Diagnosis (MedSync Vision AI)", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_publication_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: list, output_path: Path):
    """Render publication-grade multi-class ROC curves with 95% Confidence Interval AUC annotations."""
    plt.figure(figsize=(8, 7), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    
    for idx, cname in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        prob_c = y_prob[:, idx] if y_prob.ndim == 2 else y_prob
        fpr, tpr, _ = roc_curve(y_binary, prob_c)
        roc_val = auc(fpr, tpr)
        
        low_ci, up_ci = compute_bootstrap_ci(y_binary, prob_c, roc_auc_score)
        label_str = f"{cname} (AUC = {roc_val:.3f}, 95% CI [{low_ci:.3f}-{up_ci:.3f}])"
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.5, label=label_str)
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance Level (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=10)
    plt.title("Multi-Class Diagnostic Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_publication_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: list, output_path: Path):
    """Render publication-grade Precision-Recall curves."""
    plt.figure(figsize=(8, 7), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    
    for idx, cname in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        prob_c = y_prob[:, idx] if y_prob.ndim == 2 else y_prob
        precision, recall, _ = precision_recall_curve(y_binary, prob_c)
        pr_val = auc(recall, precision)
        plt.plot(recall, precision, color=colors[idx % len(colors)], lw=2.5, label=f"{cname} (PR-AUC = {pr_val:.3f})")
        
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=10)
    plt.title("Multi-Class Precision-Recall (PR) Curves", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_publication_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, metric_name: str = "T-Score"):
    """Render Bland-Altman clinical agreement plot for regression targets."""
    plt.figure(figsize=(8, 6), dpi=300)
    mean_val = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    
    plt.scatter(mean_val, diff, color="#1f77b4", alpha=0.55, edgecolor="k", s=40, label="Patient Observations")
    plt.axhline(md, color="red", linestyle="-", lw=2, label=f"Mean Bias ({md:.3f})")
    plt.axhline(md + 1.96 * sd, color="darkgreen", linestyle="--", lw=1.8, label=f"+1.96 SD (+{md + 1.96*sd:.3f})")
    plt.axhline(md - 1.96 * sd, color="darkgreen", linestyle="--", lw=1.8, label=f"-1.96 SD ({md - 1.96*sd:.3f})")
    
    plt.xlabel(f"Mean of Ground Truth & Predicted {metric_name}", fontsize=12, labelpad=10)
    plt.ylabel(f"Difference (Predicted - Ground Truth {metric_name})", fontsize=12, labelpad=10)
    plt.title(f"Bland-Altman Agreement Plot ({metric_name})", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
