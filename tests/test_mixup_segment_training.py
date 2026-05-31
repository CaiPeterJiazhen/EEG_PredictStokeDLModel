from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_mixup_script():
    script_path = PROJECT_ROOT / "scripts" / "25_train_mixup_segment_barlow.py"
    spec = spec_from_file_location("mixup_segment_barlow", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reuse_only_missing_checkpoint_raises(tmp_path: Path) -> None:
    module = _load_mixup_script()

    with pytest.raises(FileNotFoundError, match="Missing reusable Segment Barlow checkpoint"):
        module._load_reusable_branch_checkpoint(
            tmp_path / "missing.pt",
            expected_metadata={"branch": "wpli"},
            branch="wpli",
        )


def test_mixup_output_paths_do_not_overlap_bce_or_hardneg_outputs(tmp_path: Path) -> None:
    module = _load_mixup_script()

    prediction_path, metric_path = module._mixup_output_paths(
        tmp_path,
        segment_feature_kind="fc-wpli",
        seed=0,
    )

    assert prediction_path.name == "dl_loso_predictions_wpli_segbarlow_mixup_seed0.csv"
    assert metric_path.name == "dl_model_comparison_wpli_segbarlow_mixup_seed0.csv"
    assert "mixup" in prediction_path.name
    assert "mixup" in metric_path.name
    assert "hardneg" not in prediction_path.name
    assert prediction_path.name != "dl_loso_predictions_segssl_barlow_all-patient_fc-wpli_finetune_seed0_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv"
