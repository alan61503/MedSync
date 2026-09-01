from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.services.xray_service import run_inference
from backend.evaluation import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"NORMAL": 0, "OSTEOPOROSIS": 1}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip().lower()
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in {"normal", "healthy", "negative", "non osteoporosis", "non-osteoporosis", "normal bmd"}:
        return "normal"
    if cleaned in {"osteoporosis", "osteoporotic", "positive", "osteoporotic fracture", "osteoporosis patient"}:
        return "osteoporosis"
    if cleaned in {"osteopenia", "low bmd", "mild osteopenia"}:
        return "osteopenia"
    return cleaned


def _candidate_dataset_roots() -> List[Path]:
    roots = [
        REPO_ROOT / "dataset",
        REPO_ROOT / "dataset" / "Osteoporosis Knee X-ray",
        REPO_ROOT / "dataset" / "expanded_benchmark",
        REPO_ROOT / "dataset" / "benchmark",
        REPO_ROOT / "dataset" / "knee_xray_osteoporosis",
        REPO_ROOT / "dataset" / "knee-xray-osteoporosis",
        REPO_ROOT / "dataset" / "OsteoporosisKneeXray",
    ]
    unique: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        if str(root) not in seen:
            unique.append(root)
            seen.add(str(root))
    return unique


def _collect_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = []
    for child in sorted(folder.rglob("*")):
        if child.is_file() and child.suffix.lower() in VALID_EXTENSIONS:
            files.append(child)
    return files


def _class_dir_map(dataset_root: Path) -> Dict[str, Path]:
    class_map: Dict[str, Path] = {}
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir():
            label = _normalize_label(child.name)
            if label in {"normal", "osteoporosis", "osteopenia"}:
                class_map[label] = child
    return class_map


def _find_dataset_manifest(dataset_root: Path) -> Optional[Path]:
    manifest_candidates = []
    manifest_candidates.extend(sorted(dataset_root.glob("*manifest*.csv")))
    manifest_candidates.extend(sorted(dataset_root.glob("*label*.csv")))
    manifest_candidates.extend(sorted(dataset_root.glob("*.csv")))
    for candidate in manifest_candidates:
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue
        for column in ["label", "class", "diagnosis", "category", "ground_truth", "split"]:
            if column in df.columns:
                values = df[column].astype(str).tolist()
                labels = {_normalize_label(v) for v in values}
                if {"normal", "osteoporosis"}.intersection(labels):
                    return candidate
    return None


def _discover_from_metadata(dataset_root: Path) -> Optional[Dict[str, Any]]:
    manifest = _find_dataset_manifest(dataset_root)
    if manifest is None:
        return None
    try:
        df = pd.read_csv(manifest)
    except Exception:
        return None

    label_column = None
    image_column = None
    for col in ["label", "class", "diagnosis", "category", "ground_truth"]:
        if col in df.columns:
            label_column = col
            break
    for col in ["image", "filename", "file_name", "path", "image_path", "file_path", "img_path"]:
        if col in df.columns:
            image_column = col
            break
    if label_column is None or image_column is None:
        return None

    normal_paths: List[Path] = []
    osteoporosis_paths: List[Path] = []
    for _, row in df.iterrows():
        label = _normalize_label(row.get(label_column))
        image_value = row.get(image_column)
        if image_value is None or pd.isna(image_value):
            continue
        path_candidate = dataset_root / str(image_value)
        if not path_candidate.exists():
            if dataset_root.parent.exists():
                alt = dataset_root.parent / str(image_value)
                if alt.exists():
                    path_candidate = alt
        if not path_candidate.exists() or not path_candidate.is_file():
            continue
        if label == "normal":
            normal_paths.append(path_candidate)
        elif label == "osteoporosis":
            osteoporosis_paths.append(path_candidate)

    if normal_paths or osteoporosis_paths:
        return {
            "dataset_root": dataset_root,
            "normal_images": normal_paths,
            "osteoporosis_images": osteoporosis_paths,
            "label_map": {"NORMAL": 0, "OSTEOPOROSIS": 1},
        }
    return None


def discover_benchmark_dataset(dataset_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Discover the public benchmark by scanning likely dataset directories and metadata files."""
    search_roots: List[Path] = []
    if dataset_path is not None:
        search_roots.append(Path(dataset_path).expanduser().resolve())
    search_roots.extend(_candidate_dataset_roots())

    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)

        metadata = _discover_from_metadata(root)
        if metadata is not None:
            normal_count = len(metadata["normal_images"])
            osteo_count = len(metadata["osteoporosis_images"])
            if normal_count > 0 and osteo_count > 0:
                return {
                    "dataset_root": root,
                    "normal_images": metadata["normal_images"],
                    "osteoporosis_images": metadata["osteoporosis_images"],
                    "normal_images_count": normal_count,
                    "osteoporosis_images_count": osteo_count,
                    "label_map": {"NORMAL": 0, "OSTEOPOROSIS": 1},
                    "dataset_name": root.name,
                    "total_images": normal_count + osteo_count,
                }

        class_map = _class_dir_map(root)
        normal_dir = class_map.get("normal")
        osteo_dir = class_map.get("osteoporosis")
        if normal_dir is not None and osteo_dir is not None:
            normal_images = _collect_image_files(normal_dir)
            osteo_images = _collect_image_files(osteo_dir)
            if normal_images and osteo_images:
                return {
                    "dataset_root": root,
                    "normal_images": normal_images,
                    "osteoporosis_images": osteo_images,
                    "normal_images_count": len(normal_images),
                    "osteoporosis_images_count": len(osteo_images),
                    "label_map": {"NORMAL": 0, "OSTEOPOROSIS": 1},
                    "dataset_name": root.name,
                    "total_images": len(normal_images) + len(osteo_images),
                }

        if root.name.lower() in {"dataset", "data"}:
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    metadata = discover_benchmark_dataset(child)
                    if metadata is not None:
                        return metadata

    return None


def _binary_probability(probability: float) -> float:
    return float(np.clip(probability, 0.0, 1.0))


def _build_prediction_record(image_path: Path, ground_truth: int, prob: float) -> Dict[str, Any]:
    if prob < 0.5:
        predicted_class = 0
    else:
        predicted_class = 1
    return {
        "image_id": image_path.name,
        "ground_truth": int(ground_truth),
        "predicted_class": int(predicted_class),
        "prediction_probability": round(_binary_probability(prob), 6),
        "correct": bool(predicted_class == int(ground_truth)),
    }


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def _plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=200)
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Osteoporosis Benchmark Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Osteoporosis"])
    ax.set_yticklabels(["Actual Normal", "Actual Osteoporosis"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black" if cm[i, j] < cm.max() else "white")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2.5, label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Osteoporosis Benchmark ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "ground_truth",
        "predicted_class",
        "prediction_probability",
        "correct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _generate_report(metrics: Dict[str, Any], dataset_summary: Dict[str, Any], comparison: str) -> str:
    lines: List[str] = []
    lines.append("MedSync X-ray Osteoporosis Benchmark Evaluation")
    lines.append("=" * 72)
    lines.append(f"Model: {dataset_summary.get('model_name', 'PureDenseNet121')}" )
    lines.append(f"Dataset: {dataset_summary.get('dataset_name', 'Public knee X-ray osteoporosis benchmark')}")
    lines.append(f"Images evaluated: {dataset_summary.get('total_images', 0)}")
    lines.append(f"Normal images: {dataset_summary.get('normal_images', 0)}")
    lines.append(f"Osteoporosis images: {dataset_summary.get('osteoporosis_images', 0)}")
    lines.append("")
    lines.append("Primary metrics:")
    lines.append(f"  Accuracy:   {metrics['accuracy']:.4f}")
    lines.append(f"  Precision:  {metrics['precision']:.4f}")
    lines.append(f"  Recall:     {metrics['recall']:.4f}")
    lines.append(f"  Specificity: {metrics['specificity']:.4f}")
    lines.append(f"  F1-score:   {metrics['f1']:.4f}")
    lines.append(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    lines.append("")
    lines.append(f"Confusion matrix: [[{metrics['confusion_matrix'][0][0]}, {metrics['confusion_matrix'][0][1]}], [{metrics['confusion_matrix'][1][0]}, {metrics['confusion_matrix'][1][1]}]]")
    lines.append(f"TP/TN/FP/FN: {metrics['true_positive']}/{metrics['true_negative']}/{metrics['false_positive']}/{metrics['false_negative']}")
    lines.append("")
    lines.append("Data leakage and overlap check:")
    lines.append("  Training-set overlap could not be independently verified.")
    lines.append("  No training configuration or explicit dataset manifest was found in the repository to confirm independence.")
    lines.append("")
    lines.append("Published benchmark comparison (ResNet-50 paper):")
    lines.append(comparison)
    return "\n".join(lines)


def _build_comparison_table(our_metrics: Dict[str, Any]) -> str:
    published = {
        "Accuracy": 0.90,
        "Precision": 0.89,
        "Recall": 0.89,
        "F1": 0.89,
        "ROC-AUC": None,
    }
    rows = [
        ("Accuracy", published["Accuracy"], our_metrics.get("accuracy", 0.0)),
        ("Precision", published["Precision"], our_metrics.get("precision", 0.0)),
        ("Recall", published["Recall"], our_metrics.get("recall", 0.0)),
        ("F1", published["F1"], our_metrics.get("f1", 0.0)),
        ("ROC-AUC", published["ROC-AUC"], our_metrics.get("roc_auc", 0.0)),
    ]
    lines = ["Metric | Published ResNet-50 | Our Model | Difference"]
    for name, pub, ours in rows:
        if pub is None:
            pub_display = "N/A"
        else:
            pub_display = f"{pub:.3f}"
        diff = "N/A" if pub is None else f"{ours - pub:.3f}"
        lines.append(f"{name} | {pub_display} | {ours:.3f} | {diff}")
    return "\n".join(lines)


def evaluate_xray_benchmark(dataset_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    dataset = discover_benchmark_dataset(dataset_path)
    if dataset is None:
        raise FileNotFoundError(
            "Benchmark dataset not found. Place the public 372-image osteoporosis knee X-ray benchmark under the repo's dataset folder, such as 'dataset/Osteoporosis Knee X-ray' or another dataset directory with NORMAL and OSTEOPOROSIS folders."
        )

    dataset_root = Path(dataset["dataset_root"]).resolve()
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    normal_paths = dataset["normal_images"]
    osteoporosis_paths = dataset["osteoporosis_images"]
    ground_truths: List[int] = []
    probabilities: List[float] = []
    predictions: List[int] = []
    prediction_rows: List[Dict[str, Any]] = []

    for image_path in normal_paths:
        true_label = 0
        inference = run_inference(str(image_path), save_artifacts=False)
        score = float(inference.get("osteoporosis", {}).get("score", 0.5))
        probability = _binary_probability(score)
        pred = 1 if probability >= 0.5 else 0
        prediction_rows.append(_build_prediction_record(image_path, true_label, probability))
        ground_truths.append(true_label)
        probabilities.append(probability)
        predictions.append(pred)

    for image_path in osteoporosis_paths:
        true_label = 1
        inference = run_inference(str(image_path), save_artifacts=False)
        score = float(inference.get("osteoporosis", {}).get("score", 0.5))
        probability = _binary_probability(score)
        pred = 1 if probability >= 0.5 else 0
        prediction_rows.append(_build_prediction_record(image_path, true_label, probability))
        ground_truths.append(true_label)
        probabilities.append(probability)
        predictions.append(pred)

    y_true = np.array(ground_truths, dtype=int)
    y_pred = np.array(predictions, dtype=int)
    y_prob = np.array(probabilities, dtype=float)

    csv_path = results_dir / "xray_benchmark_predictions.csv"
    _write_csv(prediction_rows, csv_path)

    metrics = _compute_binary_metrics(y_true, y_pred, y_prob)
    metrics_payload = {
        "model": "ResNet-50",
        "dataset": "372-image osteoporosis knee X-ray benchmark",
        "total_images": int(len(y_true)),
        "normal_images": int(len(normal_paths)),
        "osteoporosis_images": int(len(osteoporosis_paths)),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "true_positive": int(metrics["true_positive"]),
        "true_negative": int(metrics["true_negative"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
    }

    metrics_json_path = results_dir / "xray_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    confusion_path = results_dir / "xray_confusion_matrix.png"
    roc_path = results_dir / "xray_roc_curve.png"
    _plot_confusion_matrix(np.array(metrics["confusion_matrix"], dtype=int), confusion_path)
    _plot_roc_curve(y_true, y_prob, roc_path)

    comparison = _build_comparison_table(metrics_payload)
    report_text = _generate_report(metrics, {"model_name": "ResNet-50", "dataset_name": dataset_root.name, "total_images": int(len(y_true)), "normal_images": int(len(normal_paths)), "osteoporosis_images": int(len(osteoporosis_paths))}, comparison)
    report_path = results_dir / "xray_evaluation_report.txt"
    report_path.write_text(report_text + "\n\n" + comparison + "\n", encoding="utf-8")

    return {
        "dataset": dataset_root,
        "metrics": metrics_payload,
        "predictions_csv": str(csv_path),
        "confusion_matrix_png": str(confusion_path),
        "roc_curve_png": str(roc_path),
        "report_text": str(report_path),
        "model_name": "ResNet-50",
        "class_distribution": {
            "normal": int(len(normal_paths)),
            "osteoporosis": int(len(osteoporosis_paths)),
        },
        "comparison": comparison,
    }


def main() -> int:
    try:
        result = evaluate_xray_benchmark()
    except FileNotFoundError as exc:
        print(str(exc))
        print("\nYou can place the public benchmark under one of these paths and rerun: ")
        for candidate in _candidate_dataset_roots():
            print(f" - {candidate.relative_to(REPO_ROOT) if candidate.is_relative_to(REPO_ROOT) else candidate}")
        print("\nRun with: python -m backend.evaluation.xray_benchmark")
        return 1

    print("Benchmark evaluation completed.")
    metrics = result["metrics"]
    print(f"Model: {result['model_name']}")
    print(f"Images evaluated: {metrics['total_images']}")
    print(f"Class distribution: normal={metrics['normal_images']}, osteoporosis={metrics['osteoporosis_images']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"TP/TN/FP/FN: {metrics['true_positive']}/{metrics['true_negative']}/{metrics['false_positive']}/{metrics['false_negative']}")
    print(f"CSV: {result['predictions_csv']}")
    print(f"Metrics JSON: {RESULTS_DIR / 'xray_metrics.json'}")
    print(f"Report: {RESULTS_DIR / 'xray_evaluation_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
