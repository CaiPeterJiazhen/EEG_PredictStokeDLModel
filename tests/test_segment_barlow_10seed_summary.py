from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from eeg_recovery.training.segment_barlow_model_selection import validate_locked_seed_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary_frame(n_seeds: int) -> pd.DataFrame:
    groups = [
        "no_ssl_stable_cnn",
        "psd_segbarlow_ssl_cnn",
        "wpli_segbarlow_ssl_cnn",
        "psd_wpli_segbarlow_equal_weight",
    ]
    return pd.DataFrame(
        {
            "model_group": groups,
            "aggregation": ["seed_mean_probability"] * len(groups),
            "threshold_method": ["fixed_0.5"] * len(groups),
            "n_seeds": [n_seeds] * len(groups),
        }
    )


def test_validate_locked_seed_summary_accepts_10_seed_result() -> None:
    validate_locked_seed_summary(_summary_frame(10), expected_n_seeds=10)


def test_validate_locked_seed_summary_rejects_6_seed_result() -> None:
    with pytest.raises(ValueError, match="10 locked seeds"):
        validate_locked_seed_summary(_summary_frame(6), expected_n_seeds=10)


def test_ensemble_script_normalizes_locked_output_names() -> None:
    module = _load_ensemble_script()

    assert module._normalize_threshold_methods(["fixed", "leave_one_seed_out_oof", "oof"]) == [
        "fixed",
        "leave_one_seed_out_oof",
    ]
    assert module._tagged_output_name(
        "segment_barlow_model_selection_summary.csv",
        output_tag="10seed_locked",
    ) == "segment_barlow_10seed_model_selection_summary.csv"
    assert module._tagged_output_name(
        "ensemble_predictions_psd_wpli_segbarlow_equal_weight.csv",
        output_tag="10seed_locked",
    ) == "ensemble_predictions_psd_wpli_segbarlow_equal_weight_10seed.csv"
    assert module._tagged_output_name(
        "sub09_sub14_model_scores_summary.csv",
        output_tag="10seed_locked",
    ) == "sub09_sub14_10seed_model_scores_summary.csv"


def _load_ensemble_script():
    path = PROJECT_ROOT / "scripts" / "22_ensemble_segment_barlow_predictions.py"
    spec = importlib.util.spec_from_file_location("ensemble_segment_barlow_predictions", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
