from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import subprocess
import sys
from pathlib import Path
from sklearn.naive_bayes import GaussianNB

from eeg_recovery.features.feature_tables import (
    FeatureTableError,
    load_fc_feature_table,
    load_psd_band_power_table,
)
from eeg_recovery.training.loso import LOSOFold
from eeg_recovery.training import train_baselines
from eeg_recovery.training.train_baselines import (
    BaselineModelRegistry,
    BaselinePreprocessorConfig,
    available_baseline_models,
    fit_fold_preprocessor,
    run_loso_baselines,
    write_baseline_outputs,
)


def test_fold_preprocessor_fits_scaler_selector_and_pca_on_training_subjects_only():
    X = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [999.0, 999.0, 999.0, 999.0, 999.0],
        ]
    )
    y = np.array([0, 1, 0, 1])
    subject_ids = ["sub01", "sub02", "sub03", "sub04"]
    fold = LOSOFold(
        fold_index=0,
        test_subject_id="sub04",
        train_subject_ids=("sub01", "sub02", "sub03"),
    )

    preprocessor = fit_fold_preprocessor(
        X,
        y,
        subject_ids,
        fold,
        BaselinePreprocessorConfig(
            enable_selector=True,
            selector_k=3,
            enable_pca=True,
            pca_components=2,
        ),
    )

    X_train = X[:3]
    np.testing.assert_allclose(preprocessor.scaler.mean_, X_train.mean(axis=0))
    assert 4 not in preprocessor.selector.get_support(indices=True)

    scaled_train = preprocessor.scaler.transform(X_train)
    selected_scaled_train = preprocessor.selector.transform(scaled_train)
    np.testing.assert_allclose(preprocessor.pca.mean_, selected_scaled_train.mean(axis=0))
    transformed = preprocessor.transform(X)
    assert transformed.shape == (4, 2)


def test_selector_k_is_capped_by_feature_count_not_training_sample_count():
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 3.0, 2.0, 5.0, 4.0],
            [2.0, 3.0, 0.0, 1.0, 6.0, 7.0],
            [3.0, 2.0, 1.0, 0.0, 7.0, 6.0],
        ]
    )
    y = np.array([0, 1, 0, 1])
    subject_ids = ["sub01", "sub02", "sub03", "sub04"]
    fold = LOSOFold(
        fold_index=0,
        test_subject_id="sub04",
        train_subject_ids=("sub01", "sub02", "sub03"),
    )

    preprocessor = fit_fold_preprocessor(
        X,
        y,
        subject_ids,
        fold,
        BaselinePreprocessorConfig(enable_selector=True, selector_k=4, enable_pca=False),
    )

    assert preprocessor.selector.k == 4
    assert preprocessor.transform(X).shape == (4, 4)


def test_load_psd_band_power_table_has_expected_744_features_for_one_subject(tmp_path):
    psd_dir = tmp_path / "psd"
    psd_dir.mkdir()
    freqs = np.arange(1, 91, dtype=float) * 0.5
    for state in ("EO", "EC"):
        psd = np.arange(62 * 90, dtype=float).reshape(62, 90)
        np.savez_compressed(
            psd_dir / f"sub01_{state}_psd.npz",
            psd=psd,
            frequency_bins=freqs,
        )

    table = load_psd_band_power_table(psd_dir, subject_ids=["sub01"])

    assert table.shape == (1, 745)
    assert table.loc[0, "subject_id"] == "sub01"
    assert len([column for column in table.columns if column.startswith("psd_")]) == 744


def test_malformed_psd_file_raises_feature_table_error(tmp_path):
    psd_dir = tmp_path / "psd"
    psd_dir.mkdir()
    np.savez_compressed(
        psd_dir / "sub01_EO_psd.npz",
        psd=np.ones((62, 90)),
        frequency_bins=np.arange(89, dtype=float),
    )
    np.savez_compressed(
        psd_dir / "sub01_EC_psd.npz",
        psd=np.ones((62, 90)),
        frequency_bins=np.arange(90, dtype=float),
    )

    with pytest.raises(FeatureTableError, match="frequency_bins"):
        load_psd_band_power_table(psd_dir, subject_ids=["sub01"])


def test_load_fc_feature_table_has_expected_22692_features_for_one_metric(tmp_path):
    fc_dir = tmp_path / "fc"
    fc_dir.mkdir()
    for state in ("EO", "EC"):
        wpli = np.arange(1891 * 6, dtype=float).reshape(1891, 6)
        imaginary_coherence = -wpli
        np.savez_compressed(
            fc_dir / f"sub01_{state}_fc.npz",
            wpli=wpli,
            imaginary_coherence=imaginary_coherence,
        )

    table = load_fc_feature_table(fc_dir, metric="wpli", subject_ids=["sub01"])

    assert table.shape == (1, 22693)
    assert table.loc[0, "subject_id"] == "sub01"
    assert len([column for column in table.columns if column.startswith("fc_wpli_")]) == 22692


def test_baseline_model_registry_contains_required_models_and_optional_skip_reasons():
    registry = available_baseline_models(random_state=7)

    required = {
        "logistic_l1",
        "logistic_l2",
        "svm_linear",
        "svm_rbf",
        "random_forest",
        "gaussian_nb",
        "knn",
    }
    assert required.issubset(registry.models)
    assert set(registry.skipped).issubset({"xgboost", "lightgbm"})
    for reason in registry.skipped.values():
        assert isinstance(reason, str)
        assert reason


def test_run_loso_baselines_outputs_patient_predictions_and_model_metrics():
    feature_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "f0": [0.0, 0.2, 1.0, 1.2],
            "f1": [1.0, 1.1, 0.0, 0.1],
        }
    )
    label_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "label": [0, 0, 1, 1],
            "FMA_post": [10, 20, 30, 40],
            "Residual": [1, 2, 3, 4],
        }
    )

    predictions, metrics = run_loso_baselines(
        feature_table,
        label_table,
        model_names=["gaussian_nb"],
        preprocessor_config=BaselinePreprocessorConfig(enable_selector=False, enable_pca=False),
        random_state=1,
    )

    assert predictions.shape[0] == 4
    assert set(predictions["subject_id"]) == {"sub01", "sub02", "sub03", "sub04"}
    assert predictions.groupby("subject_id").size().eq(1).all()
    assert set(["model", "fold_index", "subject_id", "y_true", "y_score", "y_pred"]).issubset(
        predictions.columns
    )
    assert metrics.shape[0] == 1
    assert metrics.loc[0, "model"] == "gaussian_nb"
    assert "balanced_accuracy" in metrics.columns
    assert predictions["status"].eq("trained").all()
    assert predictions["skip_reason"].eq("").all()
    assert metrics["status"].eq("trained").all()
    assert metrics["skip_reason"].eq("").all()


def test_run_loso_baselines_records_optional_skipped_models_in_metrics(monkeypatch):
    def fake_registry(random_state=None):
        return BaselineModelRegistry(
            models={"gaussian_nb": GaussianNB()},
            skipped={"xgboost": "missing dependency"},
        )

    monkeypatch.setattr(train_baselines, "available_baseline_models", fake_registry)
    feature_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "f0": [0.0, 0.2, 1.0, 1.2],
            "f1": [1.0, 1.1, 0.0, 0.1],
        }
    )
    label_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "label": [0, 0, 1, 1],
        }
    )

    _, metrics = run_loso_baselines(
        feature_table,
        label_table,
        preprocessor_config=BaselinePreprocessorConfig(enable_selector=False, enable_pca=False),
    )

    skipped = metrics.loc[metrics["model"] == "xgboost"].iloc[0]
    assert skipped["status"] == "skipped"
    assert skipped["skip_reason"] == "missing dependency"
    assert np.isnan(skipped["accuracy"])


def test_run_loso_baselines_explicit_skipped_model_outputs_skipped_row(monkeypatch):
    def fake_registry(random_state=None):
        return BaselineModelRegistry(
            models={"gaussian_nb": GaussianNB()},
            skipped={"xgboost": "missing dependency"},
        )

    monkeypatch.setattr(train_baselines, "available_baseline_models", fake_registry)
    feature_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "f0": [0.0, 0.2, 1.0, 1.2],
        }
    )
    label_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "label": [0, 0, 1, 1],
        }
    )

    predictions, metrics = run_loso_baselines(
        feature_table,
        label_table,
        model_names=["xgboost"],
        preprocessor_config=BaselinePreprocessorConfig(enable_selector=False, enable_pca=False),
    )

    assert predictions.empty
    assert metrics.shape[0] == 1
    assert metrics.loc[0, "model"] == "xgboost"
    assert metrics.loc[0, "status"] == "skipped"
    assert metrics.loc[0, "skip_reason"] == "missing dependency"


def test_write_baseline_outputs_uses_only_formal_prediction_and_metric_paths(tmp_path):
    predictions = pd.DataFrame(
        [{"model": "gaussian_nb", "fold_index": 0, "subject_id": "sub01", "y_true": 1, "y_score": 0.8, "y_pred": 1}]
    )
    metrics = pd.DataFrame([{"model": "gaussian_nb", "accuracy": 1.0}])

    prediction_path, metric_path = write_baseline_outputs(predictions, metrics, tmp_path)

    assert prediction_path == tmp_path / "results" / "predictions" / "ml_baseline_loso_predictions.csv"
    assert metric_path == tmp_path / "results" / "metrics" / "ml_baseline_model_comparison.csv"
    assert prediction_path.exists()
    assert metric_path.exists()
    written_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert written_files == [
        "results/metrics/ml_baseline_model_comparison.csv",
        "results/predictions/ml_baseline_loso_predictions.csv",
    ]


def test_label_derived_columns_are_not_used_as_features():
    feature_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "signal": [0.0, 0.2, 1.0, 1.2],
        }
    )
    label_table = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "label": [0, 0, 1, 1],
            "FMA_post": [999, 999, -999, -999],
            "MBI_post": [999, 999, -999, -999],
            "Delta_FMA_obs": [999, 999, -999, -999],
            "Delta_FMA_pred": [999, 999, -999, -999],
            "Residual": [999, 999, -999, -999],
        }
    )

    predictions, _ = run_loso_baselines(
        feature_table,
        label_table,
        model_names=["gaussian_nb"],
        preprocessor_config=BaselinePreprocessorConfig(enable_selector=False, enable_pca=False),
        random_state=1,
    )

    assert predictions["subject_id"].tolist() == ["sub01", "sub02", "sub03", "sub04"]


def test_metrics_return_nan_when_auc_is_undefined():
    from eeg_recovery.training.metrics import binary_classification_metrics

    values = binary_classification_metrics(np.array([1, 1]), np.array([0.2, 0.8]))

    assert np.isnan(values["roc_auc"])
    assert np.isnan(values["pr_auc"])
    assert values["brier_score"] == pytest.approx(0.34)


def test_train_ml_baselines_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/04_train_ml_baselines.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "configs/paths.example.yaml" in result.stdout
