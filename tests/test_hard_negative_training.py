from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_hardneg_script():
    script_path = PROJECT_ROOT / "scripts" / "24_train_hard_negative_segment_barlow.py"
    spec = spec_from_file_location("hard_negative_segment_barlow", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reuse_only_missing_checkpoint_raises(tmp_path: Path) -> None:
    module = _load_hardneg_script()

    with pytest.raises(FileNotFoundError, match="Missing reusable Segment Barlow checkpoint"):
        module._load_reusable_branch_checkpoint(
            tmp_path / "missing.pt",
            expected_metadata={"branch": "wpli"},
            branch="wpli",
        )


def test_hardneg_output_paths_do_not_overlap_bce_outputs(tmp_path: Path) -> None:
    module = _load_hardneg_script()

    prediction_path, metric_path = module._hardneg_output_paths(
        tmp_path,
        segment_feature_kind="fc-wpli",
        seed=0,
    )

    assert prediction_path.name == "dl_loso_predictions_wpli_segbarlow_hardneg_seed0.csv"
    assert metric_path.name == "dl_model_comparison_wpli_segbarlow_hardneg_seed0.csv"
    assert "hardneg" in prediction_path.name
    assert "hardneg" in metric_path.name
    assert prediction_path.name != "dl_loso_predictions_segssl_barlow_all-patient_fc-wpli_finetune_seed0_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv"


def test_wpli_baseline_lookup_ignores_psd_fc_wpli_filename(tmp_path: Path) -> None:
    module = _load_hardneg_script()
    prediction_dir = tmp_path / "results" / "predictions"
    prediction_dir.mkdir(parents=True)
    psd_like = prediction_dir / "dl_loso_predictions_segssl_barlow_all-patient_psd-fc-wpli_seed0_x.csv"
    wpli = prediction_dir / "dl_loso_predictions_segssl_barlow_all-patient_fc-wpli_seed0_x.csv"
    psd_like.write_text("subject_id,y_true,y_score\n", encoding="utf-8")
    wpli.write_text("subject_id,y_true,y_score\n", encoding="utf-8")

    assert module._find_baseline_prediction(tmp_path, model_kind="wpli", seed=0) == wpli
    assert module._find_baseline_prediction(tmp_path, model_kind="psd", seed=0) == psd_like
