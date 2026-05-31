from __future__ import annotations

import numpy as np

from eeg_recovery.features.qeeg_slowing import compute_qeeg_slowing_from_psd_payloads


def test_qeeg_slowing_features_are_low_dimensional_named_and_finite():
    frequency_bins = np.asarray([2.0, 6.0, 9.0, 10.0, 12.0, 20.0], dtype=np.float32)
    channel_names = np.asarray(["F3", "F4", "C3", "C4", "O1", "O2"], dtype="<U2")
    psd_eo = np.asarray(
        [
            [8, 4, 6, 9, 4, 3],
            [4, 2, 3, 5, 2, 2],
            [5, 3, 4, 8, 3, 2],
            [2, 1, 2, 3, 1, 1],
            [2, 2, 5, 10, 4, 2],
            [2, 2, 4, 8, 3, 2],
        ],
        dtype=np.float32,
    )
    psd_ec = psd_eo + np.asarray([1, 1, 2, 4, 2, 1], dtype=np.float32)

    result = compute_qeeg_slowing_from_psd_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
    )

    assert 20 <= len(result.feature_names) <= 60
    assert result.values.shape == (len(result.feature_names),)
    assert np.isfinite(result.values).all()
    assert "qeeg_eo_global_dar" in result.feature_names
    assert "qeeg_ec_global_dtabr" in result.feature_names
    assert "qeeg_ec_global_alpha_peak_hz" in result.feature_names
    assert "qeeg_ec_global_alpha_peak_prominence" in result.feature_names
    assert "qeeg_reactivity_global_alpha_ec_minus_eo" in result.feature_names
    assert "qeeg_eo_global_slow_fast_bsi" in result.feature_names


def test_qeeg_slowing_peak_frequency_tracks_alpha_peak():
    frequency_bins = np.asarray([2.0, 6.0, 8.0, 9.0, 10.0, 12.0, 20.0], dtype=np.float32)
    channel_names = np.asarray(["P3", "P4", "O1", "O2"], dtype="<U2")
    psd = np.asarray(
        [
            [1, 1, 2, 4, 12, 5, 1],
            [1, 1, 2, 4, 12, 5, 1],
            [1, 1, 1, 3, 10, 4, 1],
            [1, 1, 1, 3, 10, 4, 1],
        ],
        dtype=np.float32,
    )

    result = compute_qeeg_slowing_from_psd_payloads(
        psd_eo=psd,
        psd_ec=psd,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
    )
    lookup = dict(zip(result.feature_names, result.values, strict=True))

    assert lookup["qeeg_eo_global_alpha_peak_hz"] == 10.0
    assert lookup["qeeg_ec_global_alpha_peak_prominence"] > 1.0


def test_qeeg_slowing_validates_required_frequency_bins():
    frequency_bins = np.asarray([2.0, 6.0, 20.0], dtype=np.float32)
    channel_names = np.asarray(["F3", "F4"], dtype="<U2")
    psd = np.ones((2, 3), dtype=np.float32)

    try:
        compute_qeeg_slowing_from_psd_payloads(
            psd_eo=psd,
            psd_ec=psd,
            frequency_bins=frequency_bins,
            channel_names=channel_names,
        )
    except ValueError as exc:
        assert "alpha" in str(exc).lower()
    else:
        raise AssertionError("Expected missing alpha-bin validation to fail.")
