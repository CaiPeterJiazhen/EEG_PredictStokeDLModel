from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "secondary_ablation_explainability.py"
spec = importlib.util.spec_from_file_location("secondary_ablation_explainability", MODULE_PATH)
secondary = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = secondary
spec.loader.exec_module(secondary)


def test_psd_band_label_maps_low_edge_and_gamma() -> None:
    assert secondary.psd_band_label(0.5) == "Delta"
    assert secondary.psd_band_label(29.5) == "Beta High"
    assert secondary.psd_band_label(30.0) == "Gamma"
    assert secondary.psd_band_label(45.0) == "Gamma"


def test_required_ablation_specs_are_present() -> None:
    names = {spec.name for spec in secondary.make_ablation_specs()}
    expected = {
        "full_psd_wpli",
        "psd_only",
        "wpli_only",
        "psd_gamma_only",
        "full_minus_psd_gamma",
        "motor_wpli_edges_only",
        "non_motor_wpli_edges_only",
    }
    assert expected <= names


def test_select_columns_keeps_psd_gamma_without_wpli_gamma() -> None:
    columns = [
        "subject_id",
        "psd_EO_C3_Gamma",
        "psd_EC_C4_Gamma",
        "psd_EO_C3_Delta",
        "fc_wpli_EO_edge0000_C3-C4_Delta",
        "fc_wpli_EC_edge0001_F3-F4_Beta_High",
    ]

    spec = next(item for item in secondary.make_ablation_specs() if item.name == "psd_gamma_only")
    selected = secondary.select_feature_columns(columns, spec)

    assert selected == ["subject_id", "psd_EO_C3_Gamma", "psd_EC_C4_Gamma"]


def test_samplewise_normalization_precedes_group_summary() -> None:
    frame = pd.DataFrame(
        [
            {"subject_id": "sub01", "seed": 0, "feature_id": "a", "signed_attribution": 2.0},
            {"subject_id": "sub01", "seed": 0, "feature_id": "b", "signed_attribution": -1.0},
            {"subject_id": "sub02", "seed": 0, "feature_id": "a", "signed_attribution": 20.0},
            {"subject_id": "sub02", "seed": 0, "feature_id": "b", "signed_attribution": -10.0},
        ]
    )

    normalized = secondary.normalize_attributions_by_sample(frame, ["subject_id", "seed"])
    summary = secondary.summarize_normalized_attribution(normalized, ["feature_id"])

    row_a = summary.set_index("feature_id").loc["a"]
    row_b = summary.set_index("feature_id").loc["b"]
    assert row_a["mean_signed_attribution"] == 1.0
    assert row_a["mean_abs_attribution"] == 1.0
    assert row_b["mean_signed_attribution"] == -0.5
    assert row_b["mean_abs_attribution"] == 0.5


def test_checkpoint_audit_extracts_swa_alpha_and_inference_flags() -> None:
    payload = {
        "checkpoint_type": "residual_aware_supervised_fold",
        "metadata": {
            "model_group": "residualaware_highrank_swa_clsalpha1",
            "seed": 7,
            "fold_index": 3,
            "test_subject_id": "sub08",
            "use_swa": True,
            "selected_alpha": 1.0,
            "residual_alpha": 1.0,
        },
        "state_dict": {"classification_head.weight": object()},
        "state_scaler": {"psd": {"mean": object(), "std": object()}},
    }

    row = secondary.checkpoint_audit_row(Path("fold.pt"), payload)

    assert row["seed"] == 7
    assert row["fold_index"] == 3
    assert row["uses_swa_confirmed"] is True
    assert row["residual_alpha_is_1"] is True
    assert row["classification_head_available"] is True
