from __future__ import annotations

import numpy as np

from eeg_recovery.features.imaginary_coherence_summary import (
    compute_imaginary_coherence_summary_from_payloads,
)


def test_imaginary_coherence_summary_uses_absolute_lagged_connectivity():
    edge_list = np.asarray(
        [
            ["C3", "C4"],
            ["C3", "P3"],
            ["C4", "P4"],
            ["F3", "F4"],
        ]
    )
    band_names = np.asarray(["Alpha", "Beta Low"])
    eo = np.asarray(
        [
            [-0.2, 0.5],
            [0.1, -0.3],
            [0.4, 0.2],
            [-0.5, -0.1],
        ],
        dtype=np.float32,
    )
    ec = eo * 2.0

    vector = compute_imaginary_coherence_summary_from_payloads(
        imaginary_coherence_eo=eo,
        imaginary_coherence_ec=ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    lookup = dict(zip(vector.feature_names, vector.values, strict=True))
    assert np.isfinite(vector.values).all()
    assert lookup["imagcoh_eo_beta_low_global_abs_mean"] == np.mean(np.abs(eo[:, 1]))
    assert lookup["imagcoh_ec_beta_low_global_abs_mean"] == np.mean(np.abs(ec[:, 1]))
    assert "imagcoh_eo_beta_low_ipsilesional_strength_mean" in lookup
    assert "imagcoh_eo_beta_low_contralesional_strength_mean" in lookup
    assert "imagcoh_abs_delta_beta_low_global_abs_mean" in lookup


def test_imaginary_coherence_summary_validates_shapes():
    try:
        compute_imaginary_coherence_summary_from_payloads(
            imaginary_coherence_eo=np.zeros((2, 2), dtype=np.float32),
            imaginary_coherence_ec=np.zeros((3, 2), dtype=np.float32),
            edge_list=np.asarray([["C3", "C4"], ["F3", "F4"]]),
            band_names=np.asarray(["Alpha", "Beta Low"]),
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to fail.")
