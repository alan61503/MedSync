import pytest
import numpy as np
from pathlib import Path

from backend.evaluation import (
    compute_comprehensive_classification_metrics,
    compute_regression_metrics,
    compute_bootstrap_ci,
)
from backend.services.xray_service import run_inference
from backend.services.dxa_service import run_dxa_bmd
from backend.services.ct_bone_service import run_ct_bmd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_classification_metrics_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.15, 0.05],
        [0.1, 0.80, 0.10],
        [0.05, 0.85, 0.10],
        [0.05, 0.10, 0.85],
        [0.02, 0.08, 0.90],
    ])
    metrics = compute_comprehensive_classification_metrics(
        y_true, y_pred, y_prob, class_names=["Normal", "Osteopenia", "Osteoporosis"]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["roc_auc_ovr"] == 1.0
    assert "Normal" in metrics["per_class"]
    assert metrics["per_class"]["Normal"]["sensitivity_recall"] == 1.0


def test_regression_metrics_computation():
    y_true = np.array([-0.5, -1.8, -3.2, -0.2, -2.9])
    y_pred = np.array([-0.6, -1.7, -3.1, -0.3, -3.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] < 0.2
    assert metrics["pearson_r"] > 0.95
    assert "bland_altman" in metrics


def test_bootstrap_ci_computation():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.85, 0.9, 0.75, 0.95])
    low, up = compute_bootstrap_ci(y_true, (y_prob > 0.5).astype(int), lambda yt, yp: float(np.mean(yt == yp)))
    assert 0.0 <= low <= up <= 1.0


def test_xray_model_differential_diagnosis():
    normal_img = REPO_ROOT / "dataset" / "Osteoporosis Knee X-ray" / "normal" / "N1.JPEG"
    osteo_img = REPO_ROOT / "dataset" / "Osteoporosis Knee X-ray" / "osteoporosis" / "OS1.JPEG"
    
    if normal_img.exists() and osteo_img.exists():
        res_normal = run_inference(str(normal_img))
        res_osteo = run_inference(str(osteo_img))
        
        score_normal = res_normal["osteoporosis"]["score"]
        score_osteo = res_osteo["osteoporosis"]["score"]
        
        # Osteoporosis score must be significantly higher than Normal
        assert score_osteo > score_normal
        assert res_normal["osteoporosis"]["risk_level"] == "Low Risk (Normal BMD)"
        assert res_osteo["osteoporosis"]["risk_level"] == "High Risk (Osteoporosis)"


def test_multimodal_service_execution():
    dxa_res = run_dxa_bmd("dummy_dxa_path.dcm")
    assert dxa_res["disease"] == "Osteoporosis"
    assert "bmd" in dxa_res
    assert "t_score" in dxa_res
    
    ct_res = run_ct_bmd("dummy_ct_path.nii")
    assert ct_res["disease"] == "Osteoporosis"
    assert "bmd" in ct_res
