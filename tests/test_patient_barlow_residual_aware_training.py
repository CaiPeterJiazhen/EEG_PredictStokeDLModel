from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    script_path = PROJECT_ROOT / "scripts" / "30_train_residual_aware_patient_barlow.py"
    spec = spec_from_file_location("residual_aware_patient_barlow", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed0_output_does_not_overwrite_old_patient_barlow_results(tmp_path: Path) -> None:
    module = _load_script()

    prediction_path, metric_path = module._residual_aware_output_paths(
        tmp_path,
        output_tag="patient_barlow_residualaware_seed0",
    )

    assert prediction_path.name == "dl_loso_predictions_patient_barlow_residualaware_seed0.csv"
    assert metric_path.name == "dl_model_comparison_patient_barlow_residualaware_seed0.csv"
    assert "residualaware" in prediction_path.name
    assert prediction_path.name != (
        "dl_loso_predictions_feature_ssl_all-patient_psd-fc-wpli_gated_cnn_"
        "barlow_finetune_seed0_pre50_temp0_2_noise0_02_mask0_01_bs8_sup100_proj32.csv"
    )


def test_reuse_only_missing_checkpoint_raises(tmp_path: Path) -> None:
    module = _load_script()

    with pytest.raises(FileNotFoundError, match="Missing reusable Patient-level Barlow checkpoint"):
        module._load_patient_barlow_checkpoint(
            tmp_path / "missing.pt",
            expected_metadata={"ssl_objective": "barlow", "ssl_level": "patient_level"},
        )


def test_alpha_selection_uses_validation_scores_only() -> None:
    module = _load_script()

    selected = module._select_alpha_on_validation(
        classification_scores=[0.9, 0.8, 0.7, 0.6],
        residual_scores=[0.1, 0.2, 0.3, 0.4],
        y_true=[1, 1, 0, 0],
        candidates=[0.0, 0.5, 1.0],
    )

    assert selected == 1.0


def test_swa_update_gate_starts_at_configured_epoch() -> None:
    module = _load_script()

    assert not module._should_update_swa(epoch=49, use_swa=True, swa_start_epoch=50)
    assert module._should_update_swa(epoch=50, use_swa=True, swa_start_epoch=50)
    assert not module._should_update_swa(epoch=50, use_swa=False, swa_start_epoch=50)


def test_no_ssl_residual_aware_mode_is_not_marked_pretrained() -> None:
    module = _load_script()

    metadata = module._pretraining_mode_metadata("no_ssl", module.VARIANTS["highrank"])

    assert metadata["model_name"] == "no_ssl_residualaware_highrank"
    assert metadata["pretrained"] is False
    assert metadata["pretrained_transfer_mode"] == "residual_aware_multitask_from_scratch"
    assert metadata["requires_checkpoint"] is False
