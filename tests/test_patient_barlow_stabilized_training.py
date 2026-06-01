from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    _build_model,
    _configure_finetune_stage,
    _is_encoder_parameter_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_patient_barlow_script():
    script_path = PROJECT_ROOT / "scripts" / "29_train_patient_barlow_stabilized.py"
    spec = spec_from_file_location("patient_barlow_stabilized", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _patient_barlow_model():
    return _build_model(
        SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=4,
            dropout=0.0,
        )
    )


def test_staged_schedule_freezes_encoders_in_stage1() -> None:
    model = _patient_barlow_model()
    config = SupervisedTrainingConfig(finetune_schedule="staged_sam_swa")

    _configure_finetune_stage(model, config, stage="stage1")

    encoder_params = [param for name, param in model.named_parameters() if _is_encoder_parameter_name(name)]
    gate_params = [param for name, param in model.named_parameters() if ".gate." in name]
    classifier_params = [param for name, param in model.named_parameters() if name.startswith("classifier.")]
    assert encoder_params
    assert gate_params
    assert classifier_params
    assert all(not param.requires_grad for param in encoder_params)
    assert all(param.requires_grad for param in gate_params)
    assert all(param.requires_grad for param in classifier_params)


def test_staged_schedule_unfreezes_allowed_layers_in_stage2() -> None:
    model = _patient_barlow_model()
    config = SupervisedTrainingConfig(finetune_schedule="staged_sam_swa", freeze_conv_backbone=False)
    _configure_finetune_stage(model, config, stage="stage1")

    _configure_finetune_stage(model, config, stage="stage2")

    encoder_params = [param for name, param in model.named_parameters() if _is_encoder_parameter_name(name)]
    assert encoder_params
    assert all(param.requires_grad for param in encoder_params)


def test_reuse_only_missing_patient_barlow_checkpoint_raises(tmp_path: Path) -> None:
    module = _load_patient_barlow_script()

    with pytest.raises(FileNotFoundError, match="Missing reusable Patient-level Barlow checkpoint"):
        module._load_patient_barlow_checkpoint(
            tmp_path / "missing.pt",
            expected_metadata={"ssl_objective": "barlow", "ssl_level": "patient_level"},
        )


def test_patient_barlow_stabilized_outputs_do_not_overwrite_baseline(tmp_path: Path) -> None:
    module = _load_patient_barlow_script()

    prediction_path, metric_path = module._patient_barlow_stabilized_output_paths(
        tmp_path,
        output_tag="patient_barlow_staged_sam_swa_seed0",
    )

    assert prediction_path.name == "dl_loso_predictions_patient_barlow_staged_sam_swa_seed0.csv"
    assert metric_path.name == "dl_model_comparison_patient_barlow_staged_sam_swa_seed0.csv"
    assert "staged_sam_swa" in prediction_path.name
    assert prediction_path.name != (
        "dl_loso_predictions_feature_ssl_all-patient_psd-fc-wpli_gated_cnn_"
        "barlow_finetune_seed0_pre50_temp0_2_noise0_02_mask0_01_bs8_sup100.csv"
    )
