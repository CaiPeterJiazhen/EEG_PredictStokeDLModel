from __future__ import annotations

import numpy as np
import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from eeg_recovery.training.frozen_ssl_heads import (
    build_frozen_head_candidates,
    build_pair_difference_embedding,
    fit_predict_selected_frozen_head,
    select_score_threshold,
)


def test_select_score_threshold_maximizes_training_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.7, 0.6, 0.9])

    threshold = select_score_threshold(y_true, y_score, metric="accuracy")

    assert threshold == 0.6


def test_frozen_head_candidates_include_small_sample_linear_and_kernel_heads():
    candidates = build_frozen_head_candidates(seed=3)

    assert "logistic_l2_C1" in candidates
    assert "linear_svm_C1" in candidates
    assert "rbf_svm_C1_scale" in candidates
    assert "neural_mlp_h8_lr0_01_wd0_01" in candidates
    assert "neural_rff_d64_g0_004_lr0_01_wd0_01" in candidates
    assert "neural_rff_d64_g0_01_lr0_01_wd0_001_rs101" in candidates
    assert "neural_rbf_centers_g0_004_lr0_01_wd0_01" in candidates
    assert "neural_rbf_margin_g0_004_lr0_01_wd0_01" in candidates


def test_build_pair_difference_embedding_keeps_state_specific_and_difference_features():
    eo = np.array([[1.0, 3.0]], dtype=float)
    ec = np.array([[2.0, 1.0]], dtype=float)

    features = build_pair_difference_embedding(eo, ec)

    np.testing.assert_allclose(features, np.array([[1.0, 3.0, 2.0, 1.0, 1.0, -2.0, 1.0, 2.0]]))


def test_fit_predict_selected_frozen_head_uses_inner_loso_threshold_without_test_labels():
    x_train = np.array(
        [
            [-2.0, -1.8],
            [-1.5, -1.4],
            [-0.8, -1.0],
            [0.8, 0.7],
            [1.5, 1.3],
            [2.0, 2.2],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[1.2, 1.1]], dtype=float)

    result = fit_predict_selected_frozen_head(
        x_train,
        y_train,
        x_test,
        seed=5,
        candidate_names=("logistic_l2_C1", "linear_svm_C1"),
        selection_metric="accuracy",
    )

    assert result["selected_head"] in {"logistic_l2_C1", "linear_svm_C1"}
    assert result["y_pred"] == 1
    assert result["y_score"] >= 0.5
    assert result["threshold"] > 0


def test_fit_predict_selected_frozen_head_can_use_fixed_probability_threshold():
    x_train = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[1.5]], dtype=float)

    result = fit_predict_selected_frozen_head(
        x_train,
        y_train,
        x_test,
        seed=5,
        candidate_names=("logistic_l2_C1",),
        threshold_strategy="fixed-0.5",
    )

    assert result["threshold"] == 0.5
    assert result["y_score"] == result["raw_score"]


def test_fit_predict_selected_frozen_head_can_use_neural_network_candidate():
    x_train = np.array(
        [
            [-2.0, -1.8],
            [-1.5, -1.4],
            [-0.8, -1.0],
            [0.8, 0.7],
            [1.5, 1.3],
            [2.0, 2.2],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[1.2, 1.1]], dtype=float)

    result = fit_predict_selected_frozen_head(
        x_train,
        y_train,
        x_test,
        seed=5,
        candidate_names=("neural_mlp_h8_lr0_01_wd0_01",),
        threshold_strategy="fixed-0.5",
    )

    assert result["selected_head"] == "neural_mlp_h8_lr0_01_wd0_01"
    assert result["y_pred"] == 1


def test_fit_predict_selected_frozen_head_can_use_rff_neural_network_candidate():
    x_train = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-0.2, -0.1],
            [0.2, 0.1],
        ],
        dtype=float,
    )
    y_train = np.array([1, 1, 1, 1, 0, 0])
    x_test = np.array([[0.0, 0.0]], dtype=float)

    result = fit_predict_selected_frozen_head(
        x_train,
        y_train,
        x_test,
        seed=5,
        candidate_names=("neural_rff_d64_g0_004_lr0_01_wd0_01",),
        threshold_strategy="fixed-0.5",
    )

    assert result["selected_head"] == "neural_rff_d64_g0_004_lr0_01_wd0_01"
    assert result["y_pred"] == 0


def test_fit_predict_selected_frozen_head_can_use_rbf_center_neural_network_candidate():
    x_train = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-0.2, -0.1],
            [0.2, 0.1],
        ],
        dtype=float,
    )
    y_train = np.array([1, 1, 1, 1, 0, 0])
    x_test = np.array([[0.0, 0.0]], dtype=float)

    result = fit_predict_selected_frozen_head(
        x_train,
        y_train,
        x_test,
        seed=5,
        candidate_names=("neural_rbf_centers_g0_1_lr0_01_wd0_01",),
        threshold_strategy="fixed-0.5",
    )

    assert result["selected_head"] == "neural_rbf_centers_g0_1_lr0_01_wd0_01"
    assert result["y_pred"] == 0


def test_fit_predict_selected_frozen_head_can_use_rbf_margin_neural_network_candidate():
    x_train = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-0.2, -0.1],
            [0.2, 0.1],
        ],
        dtype=float,
    )
    y_train = np.array([1, 1, 1, 1, 0, 0])
    x_test = np.array([[0.0, 0.0]], dtype=float)

    result = fit_predict_selected_frozen_head(
        x_train,
        y_train,
        x_test,
        seed=5,
        candidate_names=("neural_rbf_margin_g0_1_lr0_01_wd0_01",),
        threshold_strategy="fixed-0.5",
    )

    assert result["selected_head"] == "neural_rbf_margin_g0_1_lr0_01_wd0_01"
    assert result["y_pred"] == 0


def test_single_head_run_name_includes_candidate_name_to_avoid_overwriting_outputs():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "11_train_frozen_vicreg_heads.py"
    spec = importlib.util.spec_from_file_location("train_frozen_vicreg_heads", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(
        data_scope="all-patient",
        seed=2,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        contrastive_weight=0.001,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        embedding_view="pair-diff",
        candidate_names="neural_rff_d64_g0_004_lr0_01_wd0_01",
        evaluate_all_candidates=False,
        selection_metric="balanced_accuracy",
        threshold_strategy="inner-loso",
    )

    run_name = module._build_run_name(args)

    assert "neural_rff_d64_g0_004" in run_name
    assert "heads1" not in run_name
    assert len(run_name) < 170


def test_multi_head_run_name_keeps_candidate_token_short_for_windows_paths():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "11_train_frozen_vicreg_heads.py"
    spec = importlib.util.spec_from_file_location("train_frozen_vicreg_heads", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(
        data_scope="all-patient",
        seed=2,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        contrastive_weight=0.001,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        embedding_view="pair-diff",
        candidate_names=",".join(
            [
                "neural_rff_d64_g0_004_lr0_01_wd0_001",
                "neural_rff_d64_g0_004_lr0_01_wd0_01",
                "neural_rff_d128_g0_004_lr0_01_wd0_001",
                "neural_rff_d128_g0_004_lr0_01_wd0_01",
            ]
        ),
        evaluate_all_candidates=True,
        selection_metric="balanced_accuracy",
        threshold_strategy="inner-loso",
    )

    run_name = module._build_run_name(args)

    assert "evalheads4_" in run_name
    assert "neural_rff_d64_g0_004_lr0_01_wd0_001" not in run_name
    assert len(run_name) < 170


def test_frozen_vicreg_head_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/11_train_frozen_vicreg_heads.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--selection-metric" in result.stdout
    assert "--threshold-strategy" in result.stdout
    assert "--embedding-view" in result.stdout
    assert "--candidate-names" in result.stdout
    assert "--evaluate-all-candidates" in result.stdout
