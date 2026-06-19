from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "46_plot_band_only_leave_band_out_ranking.py"
)


def _load_module():
    spec = spec_from_file_location("figure5c_band_ranking", SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(ablation_name: str, balanced_accuracy: float, roc_auc: float) -> dict[str, object]:
    return {
        "ablation_name": ablation_name,
        "ablation_family": "unit",
        "balanced_accuracy_mean": balanced_accuracy,
        "roc_auc_mean": roc_auc,
        "accuracy_mean": 0.0,
    }


def test_load_and_prepare_plot_data_deduplicates_filters_and_sorts(tmp_path: Path) -> None:
    module = _load_module()
    leave_roc = {
        "full_minus_delta": 0.70,
        "full_minus_theta": 0.80,
        "full_minus_alpha": 0.90,
        "full_minus_beta_low": 0.60,
        "full_minus_beta_medium": 0.85,
        "full_minus_beta_high": 0.55,
        "full_minus_psd_gamma": 0.65,
        "full_minus_beta_medium_beta_high": 0.50,
    }
    band_roc = {
        "delta_only": 0.30,
        "theta_only": 0.40,
        "alpha_only": 0.45,
        "beta_low_only": 0.35,
        "beta_medium_only": 0.20,
        "beta_high_only": 0.47,
        "psd_gamma_only": 0.37,
        "beta_medium_beta_high": 0.46,
    }

    primary = pd.DataFrame(
        [
            _row("full_psd_wpli", 0.99, 0.99),
            _row("full_minus_alpha", 0.81, leave_roc["full_minus_alpha"]),
        ]
    )
    supplementary = pd.DataFrame(
        [_row(name, roc - 0.05, roc) for name, roc in {**leave_roc, **band_roc}.items()]
        + [_row("full_minus_alpha", 0.11, 0.01)]
    )
    primary_path = tmp_path / "primary.csv"
    supplementary_path = tmp_path / "supplementary.csv"
    primary.to_csv(primary_path, index=False)
    supplementary.to_csv(supplementary_path, index=False)

    plot_data = module.load_and_prepare_plot_data(primary_path, supplementary_path)

    assert "full_psd_wpli" not in plot_data["ablation_name"].tolist()
    assert plot_data.shape[0] == 16
    assert plot_data.loc[plot_data["ablation_name"] == "full_minus_alpha", "roc_auc_mean"].item() == 0.90
    assert plot_data["ablation_name"].tolist() == [
        "full_minus_alpha",
        "full_minus_beta_medium",
        "full_minus_theta",
        "full_minus_delta",
        "full_minus_psd_gamma",
        "full_minus_beta_low",
        "full_minus_beta_high",
        "full_minus_beta_medium_beta_high",
        "beta_high_only",
        "beta_medium_beta_high",
        "alpha_only",
        "theta_only",
        "psd_gamma_only",
        "beta_low_only",
        "delta_only",
        "beta_medium_only",
    ]
    assert plot_data["display_label"].tolist()[0] == "Minus Alpha"
    assert plot_data["display_label"].tolist()[8] == "Beta High only"
    assert plot_data["block"].tolist() == ["Leave-band-out"] * 8 + ["Band-only"] * 8
