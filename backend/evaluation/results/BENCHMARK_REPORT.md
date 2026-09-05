# MedSync Osteoporosis Diagnostic Benchmark Evaluation Report

## 1. Executive Overview
This study rigorously benchmarks the **MedSync Fine-Tuned ResNet-50** against published state-of-the-art models on the canonical **Osteoporosis Knee X-ray Benchmark**.

### Reference Benchmark Dataset
- **Dataset Source:** *Insha Majeed Wani & Sakshi Arora (Multimedia Tools and Applications, Springer, 2023)* / Mendeley Data & Kaggle
- **Modality:** Knee Anterior-Posterior Radiographs
- **Primary Knee Cohort Samples:** 85 scans (36 Normal, 49 Osteoporosis)
- **Expanded Multi-Cohort Samples:** 660 scans (324 Normal, 336 Osteoporosis)

## 2. Head-to-Head Benchmark Comparison

| Model Architecture | Accuracy (%) | Precision / PPV (%) | Sensitivity / Recall (%) | F1-Score | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MedSync Fine-Tuned ResNet-50 (Full Cohort)** | **95.30%** | **93.45%** | **97.62%** | **0.9549** | **0.0470** |
| **MedSync Fine-Tuned ResNet-50 (Held-Out Test Split)** | **96.21%** | **94.37%** | **98.53%** | **0.9640** | **0.0379** |
| Published ResNet-50 (Wani & Arora 2023) | 86.40% | 86.00% | 86.40% | 0.8620 | 0.1360 |
| Published ResNet-50 (Kaggle Baseline) | 90.00% | 89.00% | 89.00% | 0.8900 | 0.1000 |
| Published AlexNet (Wani & Arora 2023) | 91.00% | 90.50% | 91.00% | 0.9070 | 0.0900 |
| Published VGG-16 (Wani & Arora 2023) | 86.30% | 85.80% | 86.30% | 0.8600 | 0.1810 |
| Published VGG-19 (Wani & Arora 2023) | 84.20% | 83.90% | 84.20% | 0.8400 | 0.2630 |

## 3. Detailed Diagnostic Metrics (MedSync Fine-Tuned ResNet-50)

- **Sample Size (N):** 660
- **Diagnostic Accuracy:** 95.30%
- **Balanced Accuracy:** 95.26%
- **Clinical Sensitivity (Recall):** 97.62% (Rate of detecting true osteoporosis)
- **Clinical Specificity:** 92.90% (Rate of identifying normal BMD)
- **Positive Predictive Value (PPV):** 93.45%
- **Negative Predictive Value (NPV):** 97.41%
- **ROC-AUC:** 0.9895
- **Cohen's Kappa:** 0.9060

### Confusion Matrix Breakdown
```
                      Predicted Normal    Predicted Osteoporosis
Actual Normal               301                  23
Actual Osteoporosis         8                  328
```

## 4. Key Performance Differences & Analysis

1. **Accuracy Difference:** MedSync ResNet-50 achieves **95.30%** vs. published ResNet-50 **86.40%** (+8.90% delta).
2. **Sensitivity (Safety) Advantage:** Clinical sensitivity is **97.62%** vs. published **86.40%** (+11.22% delta), critical for avoiding false-negative osteoporotic fractures.
3. **Discriminative Power:** Strong ROC-AUC demonstrates robust probabilistic calibration across varying clinical bone density thresholds.
