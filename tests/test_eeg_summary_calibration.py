from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_recovery.training.eeg_summary_calibration import (
    EEGSummaryCalibrationConfig,
    _fuse_base_summary_scores,
    _transform_probability_scores,
    nested_eeg_summary_residual_calibration,
    nested_univariate_eeg_summary_residual_calibration,
    select_summary_feature_columns,
)


def test_nested_eeg_summary_calibration_returns_oof_predictions_and_importance():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 7)],
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_score": [0.40, 0.55, 0.45, 0.50, 0.60, 0.65],
        }
    )
    summary = np.asarray(
        [
            [-2.0, 0.0],
            [-1.5, 0.2],
            [-1.0, 0.1],
            [1.0, 0.4],
            [1.5, 0.2],
            [2.0, 0.1],
        ],
        dtype=np.float32,
    )

    predictions, metrics, choices, importance = nested_eeg_summary_residual_calibration(
        base,
        summary,
        ("bsi_like", "noise_like"),
        EEGSummaryCalibrationConfig(
            candidate_cs=(0.01, 0.1),
            candidate_weights=(0.0, 0.5, 0.9),
            selection_objective="aucpr",
        ),
    )

    assert predictions["subject_id"].tolist() == base["subject_id"].tolist()
    assert predictions["calibrated_y_score"].between(0.0, 1.0).all()
    assert set(choices.columns) >= {"subject_id", "selected_c", "selected_weight", "selection_objective"}
    assert metrics.loc[0, "model"] == "eeg_summary_nested_calibrator"
    assert metrics.loc[0, "n_patients"] == 6
    assert importance.iloc[0]["feature_name"] == "bsi_like"
    assert importance["mean_abs_coefficient"].ge(0.0).all()


def test_nested_eeg_summary_calibration_validates_feature_alignment():
    base = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03"],
            "y_true": [0, 1, 0],
            "y_score": [0.2, 0.8, 0.4],
        }
    )
    summary = np.zeros((2, 2), dtype=np.float32)

    try:
        nested_eeg_summary_residual_calibration(
            base,
            summary,
            ("a", "b"),
            EEGSummaryCalibrationConfig(),
        )
    except ValueError as exc:
        assert "same number of rows" in str(exc)
    else:
        raise AssertionError("Expected a row-count validation error.")


def test_nested_eeg_summary_calibration_selects_threshold_inside_outer_train_only():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.20, 0.30, 0.40, 0.45, 0.48, 0.55, 0.65, 0.75],
        }
    )
    summary = np.asarray(
        [
            [-2.0],
            [-1.7],
            [-1.2],
            [-0.8],
            [0.8],
            [1.2],
            [1.7],
            [2.0],
        ],
        dtype=np.float32,
    )

    predictions, metrics, choices, _ = nested_eeg_summary_residual_calibration(
        base,
        summary,
        ("bsi_like",),
        EEGSummaryCalibrationConfig(
            candidate_cs=(0.01,),
            candidate_weights=(0.0, 0.5),
            candidate_thresholds=(0.40, 0.60),
            selection_objective="balanced_accuracy",
        ),
    )

    assert "selected_threshold" in choices.columns
    assert "selected_threshold" in predictions.columns
    assert set(choices["selected_threshold"]).issubset({0.40, 0.60})
    expected_pred = (predictions["calibrated_y_score"] >= predictions["selected_threshold"]).astype(int)
    assert predictions["y_pred"].tolist() == expected_pred.tolist()
    assert metrics.loc[0, "threshold_strategy"] == "inner_oof"


def test_nested_eeg_summary_calibration_can_fallback_when_constraints_fail():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90],
        }
    )
    misleading_summary = np.asarray(
        [
            [2.0],
            [1.7],
            [1.2],
            [0.8],
            [-0.8],
            [-1.2],
            [-1.7],
            [-2.0],
        ],
        dtype=np.float32,
    )

    predictions, metrics, choices, _ = nested_eeg_summary_residual_calibration(
        base,
        misleading_summary,
        ("misleading_bsi",),
        EEGSummaryCalibrationConfig(
            candidate_cs=(0.1,),
            candidate_weights=(0.9,),
            candidate_thresholds=(0.95,),
            selection_objective="aucpr",
            preserve_base_classification=True,
        ),
    )

    assert predictions["calibrated_y_score"].tolist() == base["y_score"].tolist()
    assert predictions["selected_threshold"].eq(0.5).all()
    assert choices["fallback_to_base"].all()
    assert metrics.loc[0, "accuracy"] == 1.0
    assert metrics.loc[0, "balanced_accuracy"] == 1.0


def test_nested_eeg_summary_calibration_can_require_preserved_sensitivity():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90],
        }
    )
    misleading_summary = np.asarray(
        [
            [-1.0],
            [-0.8],
            [-0.6],
            [-0.4],
            [-0.2],
            [-0.1],
            [0.1],
            [0.2],
        ],
        dtype=np.float32,
    )

    predictions, metrics, choices, _ = nested_univariate_eeg_summary_residual_calibration(
        base,
        misleading_summary,
        ("low_positive_support",),
        EEGSummaryCalibrationConfig(
            candidate_weights=(1.0,),
            candidate_thresholds=(0.95,),
            selection_objective="balanced_accuracy",
            preserve_base_classification=True,
            min_accuracy_delta=-1.0,
            min_balanced_accuracy_delta=-1.0,
            min_sensitivity_delta=0.0,
        ),
    )

    assert predictions["calibrated_y_score"].tolist() == base["y_score"].tolist()
    assert choices["fallback_to_base"].all()
    assert metrics.loc[0, "sensitivity"] == 1.0


def test_select_summary_feature_columns_supports_include_and_exclude_regex():
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    names = (
        "psd_eo_delta_mean",
        "psd_eo_delta_alpha_ratio",
        "psd_eo_frontal_beta_bsi",
        "wpli_eo_alpha_interhemispheric_mean",
    )

    selected, selected_names = select_summary_feature_columns(
        matrix,
        names,
        include_regex="psd|wpli",
        exclude_regex="ratio|interhemispheric",
    )

    assert selected.shape == (3, 2)
    assert selected_names == ("psd_eo_delta_mean", "psd_eo_frontal_beta_bsi")
    assert selected.tolist() == matrix[:, [0, 2]].tolist()


def test_fuse_base_summary_scores_supports_low_capacity_modes():
    base = np.asarray([0.90, 0.20], dtype=float)
    summary = np.asarray([0.10, 0.80], dtype=float)

    linear = _fuse_base_summary_scores(base, summary, weight=0.5, fusion_mode="linear")
    geometric = _fuse_base_summary_scores(base, summary, weight=0.5, fusion_mode="geometric")
    veto = _fuse_base_summary_scores(base, summary, weight=0.5, fusion_mode="veto")

    np.testing.assert_allclose(linear, [0.50, 0.50])
    np.testing.assert_allclose(geometric, [0.30, 0.40])
    np.testing.assert_allclose(veto, [0.50, 0.20])


def test_transform_probability_scores_supports_logit_scale_sharpening():
    scores = np.asarray([0.20, 0.40, 0.50, 0.60, 0.80], dtype=float)

    unchanged = _transform_probability_scores(scores, logit_scale=1.0)
    sharpened = _transform_probability_scores(scores, logit_scale=2.0)

    np.testing.assert_allclose(unchanged, scores)
    assert sharpened[0] < scores[0]
    assert sharpened[1] < scores[1]
    assert sharpened[2] == scores[2]
    assert sharpened[3] > scores[3]
    assert sharpened[4] > scores[4]


def test_nested_eeg_summary_calibration_records_selected_fusion_mode():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.20, 0.70, 0.30, 0.75, 0.65, 0.70, 0.80, 0.85],
        }
    )
    summary = np.asarray(
        [
            [-2.0],
            [-1.8],
            [-1.4],
            [-1.1],
            [1.1],
            [1.4],
            [1.8],
            [2.0],
        ],
        dtype=np.float32,
    )

    predictions, _, choices, _ = nested_eeg_summary_residual_calibration(
        base,
        summary,
        ("bsi_like",),
        EEGSummaryCalibrationConfig(
            candidate_cs=(0.1,),
            candidate_weights=(0.5,),
            candidate_fusion_modes=("veto",),
            selection_objective="balanced_accuracy",
        ),
    )

    assert choices["selected_fusion_mode"].eq("veto").all()
    assert predictions["selected_fusion_mode"].eq("veto").all()


def test_nested_univariate_calibration_selects_one_interpretable_feature_per_outer_fold():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.20, 0.75, 0.30, 0.40, 0.55, 0.65, 0.75, 0.85],
        }
    )
    summary = np.asarray(
        [
            [0.0, 0.4],
            [0.1, 0.5],
            [0.2, 0.1],
            [0.3, 0.7],
            [0.7, 0.2],
            [0.8, 0.6],
            [0.9, 0.3],
            [1.0, 0.8],
        ],
        dtype=np.float32,
    )

    predictions, metrics, choices, importance = nested_univariate_eeg_summary_residual_calibration(
        base,
        summary,
        ("informative_bsi", "noise_like"),
        EEGSummaryCalibrationConfig(
            candidate_weights=(0.0, 0.4, 0.7),
            selection_objective="balanced_accuracy",
        ),
    )

    assert "selected_feature_name" in predictions.columns
    assert "selected_feature_direction" in predictions.columns
    assert set(choices.columns) >= {"selected_feature_name", "selected_feature_direction", "selected_weight"}
    assert set(predictions["selected_feature_name"]).issubset({"informative_bsi", "noise_like"})
    assert metrics.loc[0, "model"] == "eeg_summary_univariate_nested_calibrator"
    assert importance["selection_count"].sum() == len(base)


def test_nested_univariate_calibration_records_selected_fusion_mode():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.20, 0.70, 0.30, 0.75, 0.65, 0.70, 0.80, 0.85],
        }
    )
    summary = np.asarray(
        [
            [0.0],
            [0.1],
            [0.2],
            [0.3],
            [0.7],
            [0.8],
            [0.9],
            [1.0],
        ],
        dtype=np.float32,
    )

    predictions, _, choices, importance = nested_univariate_eeg_summary_residual_calibration(
        base,
        summary,
        ("rank_bsi",),
        EEGSummaryCalibrationConfig(
            candidate_weights=(0.5,),
            candidate_fusion_modes=("geometric",),
            selection_objective="balanced_accuracy",
        ),
    )

    assert choices["selected_fusion_mode"].eq("geometric").all()
    assert predictions["selected_fusion_mode"].eq("geometric").all()
    assert importance["fusion_mode"].eq("geometric").all()


def test_nested_univariate_calibration_records_score_transform_scale():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.20, 0.70, 0.30, 0.75, 0.65, 0.70, 0.80, 0.85],
        }
    )
    summary = np.asarray(
        [
            [1.0],
            [0.9],
            [0.8],
            [0.7],
            [0.3],
            [0.2],
            [0.1],
            [0.0],
        ],
        dtype=np.float32,
    )

    predictions, _, choices, _ = nested_univariate_eeg_summary_residual_calibration(
        base,
        summary,
        ("locked_inverse_bsi",),
        EEGSummaryCalibrationConfig(
            candidate_weights=(0.5,),
            candidate_fusion_modes=("geometric",),
            candidate_feature_directions=(-1,),
            candidate_score_transform_scales=(2.0,),
            selection_objective="balanced_accuracy",
        ),
    )

    assert choices["selected_score_transform_scale"].eq(2.0).all()
    assert predictions["selected_score_transform_scale"].eq(2.0).all()
    expected = _transform_probability_scores(
        _fuse_base_summary_scores(
            predictions["base_y_score"].to_numpy(dtype=float),
            predictions["summary_y_score"].to_numpy(dtype=float),
            weight=0.5,
            fusion_mode="geometric",
        ),
        logit_scale=2.0,
    )
    np.testing.assert_allclose(predictions["calibrated_y_score"].to_numpy(dtype=float), expected)


def test_nested_univariate_calibration_rejects_invalid_score_transform_scale():
    base = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "y_true": [0, 0, 1, 1],
            "y_score": [0.2, 0.4, 0.6, 0.8],
        }
    )
    summary = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)

    try:
        nested_univariate_eeg_summary_residual_calibration(
            base,
            summary,
            ("bsi_like",),
            EEGSummaryCalibrationConfig(candidate_score_transform_scales=(0.0,)),
        )
    except ValueError as exc:
        assert "candidate_score_transform_scales" in str(exc)
    else:
        raise AssertionError("Expected invalid score transform scale to be rejected.")


def test_nested_univariate_calibration_can_lock_feature_direction():
    base = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.20, 0.70, 0.30, 0.75, 0.65, 0.70, 0.80, 0.85],
        }
    )
    summary = np.asarray(
        [
            [1.0],
            [0.9],
            [0.8],
            [0.7],
            [0.3],
            [0.2],
            [0.1],
            [0.0],
        ],
        dtype=np.float32,
    )

    predictions, _, choices, importance = nested_univariate_eeg_summary_residual_calibration(
        base,
        summary,
        ("locked_inverse_bsi",),
        EEGSummaryCalibrationConfig(
            candidate_weights=(0.5,),
            candidate_fusion_modes=("geometric",),
            candidate_feature_directions=(-1,),
            selection_objective="balanced_accuracy",
        ),
    )

    assert choices["selected_feature_direction"].eq(-1).all()
    assert predictions["selected_feature_direction"].eq(-1).all()
    assert importance["feature_direction"].eq(-1).all()


def test_nested_univariate_calibration_rejects_invalid_locked_direction():
    base = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "y_true": [0, 0, 1, 1],
            "y_score": [0.2, 0.4, 0.6, 0.8],
        }
    )
    summary = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)

    try:
        nested_univariate_eeg_summary_residual_calibration(
            base,
            summary,
            ("bsi_like",),
            EEGSummaryCalibrationConfig(candidate_feature_directions=(0,)),
        )
    except ValueError as exc:
        assert "candidate_feature_directions" in str(exc)
    else:
        raise AssertionError("Expected invalid feature direction to be rejected.")
