# MedSync Model Accuracy & Benchmark Evaluation Report

**Independent Quantitative Validation (No Filename Leakage / Unbiased Feature Extraction)**

---

## 1. Executive Summary

This report documents the quantitative validation of the AI diagnostic models within **MedSync**, evaluated against both the **Primary Clinical Knee X-ray Cohort** ($N=239$) and the **Expanded Multi-Cohort Radiographic Benchmark** ($N=1198$).

### Key Performance Highlights:
- **Primary Clinical Knee Cohort Accuracy**: **46.86%** (Balanced Accuracy: **36.06%**)
- **Expanded Multi-Cohort Benchmark Accuracy**: **35.31%** (Balanced Accuracy: **38.31%**)
- **Expanded Multi-Class ROC-AUC (One-vs-Rest)**: **0.7113**
- **Expanded Macro F1-Score**: **0.2676**
- **Expanded Cohen's Kappa ($\kappa$)**: **0.0758** (Slight agreement)
- **T-Score Continuous Correlation**: **$r = 0.2242$** ($p = 4.03e-15$ on Expanded Cohort; $r = 0.0809$ on Clinical Cohort).

---

## 2. Multi-Class Diagnostic Confusion Matrices & Per-Class Metrics

### A. Primary Clinical Knee Cohort ($N=239$):

| Diagnostic Class | Sensitivity (Recall) | Specificity | Precision (PPV) | NPV | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal BMD** | 44.44% | 62.56% | 17.39% | 86.39% | 0.2500 |
| **Osteopenia** | 61.69% | 41.18% | 65.52% | 37.23% | 0.6355 |
| **Osteoporosis** | 2.04% | 99.47% | 50.00% | 79.75% | 0.0392 |

#### Confusion Matrix (Primary Clinical Cohort):
| Ground Truth \ Predicted | Normal BMD | Osteopenia | Osteoporosis | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Normal BMD** | 16 | 20 | 0 | 36 |
| **Osteopenia** | 58 | 95 | 1 | 154 |
| **Osteoporosis** | 18 | 30 | 1 | 49 |

---

### B. Expanded Multi-Cohort Benchmark ($N=1198$):

| Diagnostic Class | Sensitivity (Recall) | Specificity | Precision (PPV) | NPV | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal BMD** | 90.74% | 19.91% | 29.58% | 85.29% | 0.4461 |
| **Osteopenia** | 23.61% | 89.24% | 64.14% | 58.90% | 0.3451 |
| **Osteoporosis** | 0.60% | 99.54% | 33.33% | 71.98% | 0.0117 |

#### Confusion Matrix (Expanded Benchmark):
| Ground Truth \ Predicted | Normal BMD | Osteopenia | Osteoporosis | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Normal BMD** | 294 | 29 | 1 | 324 |
| **Osteopenia** | 408 | 127 | 3 | 538 |
| **Osteoporosis** | 292 | 42 | 2 | 336 |

---

## 3. Comparison Across Cohorts

| Metric | Primary Clinical Knee Cohort ($N=239$) | Expanded Multi-Cohort Benchmark ($N=1198$) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 46.86% | 35.31% |
| **Balanced Accuracy** | 36.06% | 38.31% |
| **ROC-AUC (OvR)** | 0.5140 | 0.7113 |
| **Macro F1** | 0.3082 | 0.2676 |
| **Weighted F1** | 0.4552 | 0.2789 |
| **Cohen's Kappa ($\kappa$)** | 0.0328 (Slight agreement) | 0.0758 (Slight agreement) |
| **T-Score MAE** | 0.987 | 1.529 |
| **T-Score RMSE** | 1.212 | 1.862 |
| **T-Score Pearson $r$** | 0.0809 | 0.2242 |

---

## 4. Multi-Modal Verification

- **DXA BMD Estimator**: Evaluated across 100 scans ($\mu=0.880$ g/cm$^2$, mean T-score = -1.80).
- **3D CT Volumetric BMD (DeepBone)**: Evaluated across 0 volumetric datasets.

---

## 5. Artifacts & Visualizations Generated

- **Confusion Matrices**: `outputs/evaluation_results/confusion_matrix_primary_clinical_cohort.png`, `confusion_matrix_expanded_benchmark.png`
- **ROC Curves with 95% CIs**: `outputs/evaluation_results/roc_curves_primary_clinical_cohort.png`, `roc_curves_expanded_benchmark.png`
- **Precision-Recall Curves**: `outputs/evaluation_results/pr_curves_primary_clinical_cohort.png`, `pr_curves_expanded_benchmark.png`
- **Bland-Altman Agreement**: `outputs/evaluation_results/bland_altman_tscore.png`
- **LaTeX Research Tables**: `outputs/evaluation_results/paper_tables.latex`
- **JSON Metrics**: `outputs/evaluation_results/metrics_summary.json`
