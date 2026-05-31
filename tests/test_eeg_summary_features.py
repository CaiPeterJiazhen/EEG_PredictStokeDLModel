from __future__ import annotations

import numpy as np

from eeg_recovery.features.eeg_summary import compute_eeg_summary_from_feature_payloads


def test_eeg_summary_features_are_named_finite_and_include_psd_wpli_bsi():
    frequency_bins = np.array([2.0, 6.0, 10.0, 20.0, 35.0], dtype=np.float32)
    channel_names = np.array(["F3", "F4", "C3", "C4"], dtype="<U2")
    edge_list = np.array([["F3", "F4"], ["C3", "C4"], ["F3", "C3"]], dtype="<U2")
    band_names = np.array(["delta", "theta", "alpha", "beta", "gamma"], dtype="<U5")
    psd_eo = np.array(
        [
            [1, 1, 1, 2, 1],
            [2, 2, 2, 4, 2],
            [3, 3, 3, 6, 3],
            [4, 4, 4, 8, 4],
        ],
        dtype=np.float32,
    )
    psd_ec = psd_eo + 1.0
    wpli_eo = np.tile(np.linspace(0.1, 0.6, 5, dtype=np.float32), (3, 1))
    wpli_ec = wpli_eo + 0.1

    result = compute_eeg_summary_from_feature_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    assert result.values.shape[0] == len(result.feature_names)
    assert np.isfinite(result.values).all()
    assert "psd_eo_beta_mean" in result.feature_names
    assert "wpli_ec_beta_mean" in result.feature_names
    assert "psd_eo_beta_bsi" in result.feature_names
    beta_bsi = result.values[result.feature_names.index("psd_eo_beta_bsi")]
    assert beta_bsi > 0.0


def test_eeg_summary_features_include_roi_bsi_and_wpli_graph_features():
    frequency_bins = np.array([2.0, 6.0, 10.0, 20.0, 35.0], dtype=np.float32)
    channel_names = np.array(["F3", "F4", "C3", "C4"], dtype="<U2")
    edge_list = np.array(
        [
            ["F3", "F4"],
            ["F3", "C3"],
            ["F4", "C4"],
            ["F3", "C4"],
        ],
        dtype="<U2",
    )
    band_names = np.array(["delta", "theta", "alpha", "beta", "gamma"], dtype="<U5")
    psd_eo = np.array(
        [
            [1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3],
            [5, 5, 5, 5, 5],
            [7, 7, 7, 7, 7],
        ],
        dtype=np.float32,
    )
    psd_ec = psd_eo + 1.0
    wpli_eo = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32), (4, 1))
    wpli_ec = wpli_eo + 0.05

    result = compute_eeg_summary_from_feature_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    assert "psd_eo_frontal_beta_bsi" in result.feature_names
    assert "psd_ec_central_alpha_mean" in result.feature_names
    assert "wpli_eo_alpha_interhemispheric_mean" in result.feature_names
    assert "wpli_eo_alpha_left_intra_mean" in result.feature_names
    assert "wpli_eo_alpha_right_intra_mean" in result.feature_names
    assert "wpli_eo_alpha_intra_asymmetry" in result.feature_names


def test_eeg_summary_features_include_lesion_aligned_directional_features():
    frequency_bins = np.array([2.0, 6.0, 10.0, 20.0, 35.0], dtype=np.float32)
    channel_names = np.array(["F3", "F4", "C3", "C4"], dtype="<U2")
    edge_list = np.array(
        [
            ["F3", "C3"],
            ["F4", "C4"],
            ["F3", "F4"],
        ],
        dtype="<U2",
    )
    band_names = np.array(["delta", "theta", "alpha", "beta", "gamma"], dtype="<U5")
    psd_eo = np.array(
        [
            [10, 10, 10, 10, 10],
            [2, 2, 2, 2, 2],
            [6, 6, 6, 6, 6],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    psd_ec = psd_eo + 1.0
    wpli_eo = np.asarray(
        [
            [0.8, 0.8, 0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    wpli_ec = wpli_eo + 0.05

    result = compute_eeg_summary_from_feature_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    assert "psd_eo_beta_ipsilesional_mean" in result.feature_names
    assert "psd_eo_beta_contralesional_mean" in result.feature_names
    assert "psd_eo_beta_ipsi_contra_signed_asymmetry" in result.feature_names
    assert "psd_eo_frontal_beta_ipsilesional_mean" in result.feature_names
    assert "wpli_eo_beta_ipsilesional_intra_mean" in result.feature_names
    assert "wpli_eo_beta_contralesional_intra_mean" in result.feature_names
    assert "wpli_eo_beta_intra_signed_asymmetry" in result.feature_names
    signed = result.values[result.feature_names.index("psd_eo_beta_ipsi_contra_signed_asymmetry")]
    assert signed > 0.0


def test_eeg_summary_features_include_wpli_graph_theory_metrics():
    frequency_bins = np.array([2.0, 6.0, 10.0, 20.0, 35.0], dtype=np.float32)
    channel_names = np.array(["F3", "F4", "C3", "C4"], dtype="<U2")
    edge_list = np.array(
        [
            ["F3", "F4"],
            ["F3", "C3"],
            ["F3", "C4"],
            ["F4", "C3"],
            ["F4", "C4"],
            ["C3", "C4"],
        ],
        dtype="<U2",
    )
    band_names = np.array(["delta", "theta", "alpha", "beta", "gamma"], dtype="<U5")
    psd_eo = np.ones((4, 5), dtype=np.float32)
    psd_ec = psd_eo + 1.0
    wpli_eo = np.asarray(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.8, 0.8, 0.8, 0.8, 0.8],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    wpli_ec = wpli_eo + 0.05

    result = compute_eeg_summary_from_feature_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    assert "wpli_eo_beta_global_strength_mean" in result.feature_names
    assert "wpli_eo_beta_global_efficiency" in result.feature_names
    assert "wpli_eo_beta_weighted_clustering" in result.feature_names
    assert "wpli_eo_beta_ipsilesional_strength_mean" in result.feature_names
    assert "wpli_eo_beta_contralesional_strength_mean" in result.feature_names
    assert "wpli_eo_beta_strength_signed_asymmetry" in result.feature_names
    strength_asym = result.values[result.feature_names.index("wpli_eo_beta_strength_signed_asymmetry")]
    assert strength_asym > 0.0
    assert np.isfinite(result.values).all()


def test_eeg_summary_features_include_psd_spectral_shape_features():
    frequency_bins = np.array([2.0, 6.0, 10.0, 20.0, 35.0], dtype=np.float32)
    channel_names = np.array(["F3", "F4", "C3", "C4"], dtype="<U2")
    edge_list = np.array([["F3", "F4"], ["C3", "C4"]], dtype="<U2")
    band_names = np.array(["delta", "theta", "alpha", "beta", "gamma"], dtype="<U5")
    psd_eo = np.array(
        [
            [10, 4, 2, 1, 1],
            [8, 4, 2, 1, 1],
            [1, 2, 4, 8, 10],
            [1, 2, 4, 8, 8],
        ],
        dtype=np.float32,
    )
    psd_ec = psd_eo + 1.0
    wpli_eo = np.full((2, 5), 0.2, dtype=np.float32)
    wpli_ec = np.full((2, 5), 0.3, dtype=np.float32)

    result = compute_eeg_summary_from_feature_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    assert "psd_eo_spectral_entropy" in result.feature_names
    assert "psd_eo_spectral_centroid_hz" in result.feature_names
    assert "psd_eo_spectral_spread_hz" in result.feature_names
    assert "psd_eo_spectral_edge_95_hz" in result.feature_names
    assert "psd_eo_frontal_spectral_entropy" in result.feature_names
    entropy = result.values[result.feature_names.index("psd_eo_spectral_entropy")]
    edge = result.values[result.feature_names.index("psd_eo_spectral_edge_95_hz")]
    assert 0.0 <= entropy <= 1.0
    assert frequency_bins.min() <= edge <= frequency_bins.max()
    assert np.isfinite(result.values).all()


def test_eeg_summary_features_include_stroke_qeeg_band_ratios_and_relative_power():
    frequency_bins = np.array([2.0, 6.0, 10.0, 20.0, 35.0], dtype=np.float32)
    channel_names = np.array(["F3", "F4", "C3", "C4"], dtype="<U2")
    edge_list = np.array([["F3", "F4"], ["C3", "C4"]], dtype="<U2")
    band_names = np.array(["delta", "theta", "alpha", "beta", "gamma"], dtype="<U5")
    psd_eo = np.array(
        [
            [8, 4, 2, 1, 1],
            [8, 4, 2, 1, 1],
            [4, 4, 4, 2, 2],
            [4, 4, 4, 2, 2],
        ],
        dtype=np.float32,
    )
    psd_ec = psd_eo + 1.0
    wpli_eo = np.full((2, 5), 0.2, dtype=np.float32)
    wpli_ec = np.full((2, 5), 0.3, dtype=np.float32)

    result = compute_eeg_summary_from_feature_payloads(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        frequency_bins=frequency_bins,
        channel_names=channel_names,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        edge_list=edge_list,
        band_names=band_names,
    )

    assert "psd_eo_delta_alpha_ratio" in result.feature_names
    assert "psd_eo_delta_theta_ratio" in result.feature_names
    assert "psd_eo_delta_theta_alpha_beta_ratio" in result.feature_names
    assert "psd_eo_delta_relative_mean" in result.feature_names
    assert "psd_eo_frontal_delta_alpha_ratio" in result.feature_names
    assert "psd_ec_central_beta_relative_mean" in result.feature_names
    assert np.isfinite(result.values).all()
    dar = result.values[result.feature_names.index("psd_eo_delta_alpha_ratio")]
    assert dar > 1.0
