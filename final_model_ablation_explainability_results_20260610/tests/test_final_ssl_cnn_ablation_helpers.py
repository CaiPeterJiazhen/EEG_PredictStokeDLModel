from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "43_run_final_ssl_cnn_feature_state_band_ablation.py"
)


def _load_module():
    spec = spec_from_file_location("final_ssl_cnn_ablation", SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_psd_band_labels_include_gamma_and_no_other() -> None:
    module = _load_module()
    frequencies = np.arange(0.5, 45.5, 0.5)
    labels = module.psd_band_labels(frequencies)
    assert "Gamma" in labels
    assert "Other" not in labels
    assert labels[-1] == "Gamma"


def test_motor_edge_mask_keeps_edges_with_at_least_one_motor_node() -> None:
    module = _load_module()
    edges = [("FP1", "FPZ"), ("FP1", "C3"), ("PZ", "CP6")]
    mask = module.motor_edge_mask(edges)
    assert mask.tolist() == [False, True, True]


def test_ablation_mask_counts_full_and_psd_only() -> None:
    module = _load_module()
    metadata = module.FeatureMetadata(
        channel_names=tuple(f"C{i}" for i in range(62)),
        frequency_bins=np.arange(0.5, 45.5, 0.5),
        edge_list=tuple(("C3", "PZ") for _ in range(1891)),
        wpli_band_names=("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High"),
    )
    specs = {spec.name: spec for spec in module.ablation_specs()}
    full = module.build_ablation_mask(specs["full_psd_wpli"], metadata)
    psd_only = module.build_ablation_mask(specs["psd_only"], metadata)
    assert full.n_active_psd_values == 2 * 62 * 90
    assert full.n_active_wpli_values == 2 * 1891 * 6
    assert psd_only.n_active_psd_values == 2 * 62 * 90
    assert psd_only.n_active_wpli_values == 0


def test_apply_mask_zeroes_standardized_space_without_shape_change() -> None:
    module = _load_module()
    mask = module.AblationMask(
        psd_eo=np.array([[True, False], [False, True]]),
        psd_ec=np.array([[False, False], [True, True]]),
        wpli_eo=np.array([[True, False]]),
        wpli_ec=np.array([[False, True]]),
        n_active_psd_values=4,
        n_active_wpli_values=2,
        description="unit",
    )
    batch = {
        "psd_eo": module.torch.ones((1, 2, 2)),
        "psd_ec": module.torch.ones((1, 2, 2)),
        "wpli_eo": module.torch.ones((1, 1, 2)),
        "wpli_ec": module.torch.ones((1, 1, 2)),
    }
    masked = module.apply_ablation_mask_to_batch(batch, mask)
    assert masked["psd_eo"].tolist() == [[[1.0, 0.0], [0.0, 1.0]]]
    assert masked["psd_ec"].tolist() == [[[0.0, 0.0], [1.0, 1.0]]]
    assert masked["wpli_eo"].tolist() == [[[1.0, 0.0]]]
    assert masked["wpli_ec"].tolist() == [[[0.0, 1.0]]]
