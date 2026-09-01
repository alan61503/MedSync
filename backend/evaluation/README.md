# X-ray Osteoporosis Benchmark Evaluation

## Purpose

This evaluation module independently measures the existing osteoporosis X-ray model on a public benchmark dataset without modifying the model, retraining it, or altering the project frontend.

## Model under evaluation

The system evaluates the current X-ray inference pipeline in [backend/services/xray_service.py](../services/xray_service.py), which loads the existing `PureDenseNet121`-style PyTorch model implementation and computes an osteoporosis risk score from image intensity, radiomic features, and a fracture indicator.

- Model name: `PureDenseNet121`
- Input format: grayscale normalized image tensor generated from the source X-ray image
- Preprocessing: convert to RGB, grayscale, normalize to 0-1, resize to 224x224, then pass to the PyTorch model
- Output format: dictionary with `osteoporosis.score`, `predictions`, `supporting_findings`, risk labels, and heatmap metadata
- Class names: `NORMAL = 0`, `OSTEOPOROSIS = 1`
- Probability/confidence: the osteoporosis score is used as the positive-class probability

## Dataset location

The public benchmark must be placed in one of these repository-relative locations before running the evaluation:

- `dataset/Osteoporosis Knee X-ray`
- `dataset/benchmark/`
- `dataset/knee_xray_osteoporosis/`

The target public benchmark is the 372-image knee X-ray dataset used in the study "Deep learning-based osteoporosis classification using knee radiographs" with 186 normal and 186 osteoporosis images.

## Label detection

The benchmark loader automatically checks the dataset structure and metadata in this order:

1. CSV metadata files such as manifest/label files
2. Folder structure containing `normal` and `osteoporosis` directories
3. Other case-insensitive normalizations for common label names

When the benchmark is found, labels are mapped as:

- `NORMAL = 0`
- `OSTEOPOROSIS = 1`

If the dataset includes a third class such as `osteopenia`, it is not used for the binary benchmark calculation unless the dataset is explicitly a to-be-evaluated multi-class benchmark.

## Data leakage note

The repository does not include a training configuration, checkpoint manifest, or explicit training-data provenance that would allow the benchmark to be proven independent. Therefore:

> Training-set overlap could not be independently verified.

This is reported in the evaluation outputs and must not be overstated as fully independent.

## Run the benchmark

From the repository root:

```bash
python -m backend.evaluation.xray_benchmark
```

This command will:

- discover the benchmark dataset automatically
- evaluate every image through the existing inference pipeline
- save predictions to `backend/evaluation/results/xray_benchmark_predictions.csv`
- compute metrics and save `backend/evaluation/results/xray_metrics.json`
- generate `backend/evaluation/results/xray_confusion_matrix.png`
- generate `backend/evaluation/results/xray_roc_curve.png`
- write a human-readable report to `backend/evaluation/results/xray_evaluation_report.txt`

## Outputs

The script writes the following artifacts under `backend/evaluation/results/`:

- `xray_benchmark_predictions.csv`
- `xray_metrics.json`
- `xray_confusion_matrix.png`
- `xray_roc_curve.png`
- `xray_evaluation_report.txt`

## Notes

- No model replacement is performed.
- No retraining or fine-tuning is performed.
- No frontend changes are required for benchmark evaluation.
- The script is the authoritative reproducible benchmark method.
