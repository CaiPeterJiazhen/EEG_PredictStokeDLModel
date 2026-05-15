from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_recovery.channels.hemisphere_flip import flip_channels_for_affected_hand
from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62, ChannelMappingError
from eeg_recovery.features.connectivity import (
    ConnectivityConfig,
    ConnectivityFeatureError,
    build_edge_list,
    compute_single_state_connectivity,
    connectivity_bands,
    write_connectivity_feature,
)


SAMPLING_RATE = 250.0
CHANNELS = CANONICAL_CHANNELS_62
TEST_CONFIG = ConnectivityConfig(nperseg=128, noverlap=64)


def _synthetic_data(samples: int = 512) -> np.ndarray:
    time = np.arange(samples) / SAMPLING_RATE
    data = np.zeros((len(CHANNELS), samples), dtype=float)
    for index in range(len(CHANNELS)):
        phase = index * 0.03
        data[index] = (
            np.sin(2 * np.pi * 10.0 * time + phase)
            + 0.4 * np.sin(2 * np.pi * 20.0 * time + phase / 2.0)
        )
    return data


def test_build_edge_list_uses_deterministic_upper_triangle_canonical_order() -> None:
    edges = build_edge_list()

    assert len(edges) == 1891
    assert edges[:5] == (
        ("FP1", "FPZ"),
        ("FP1", "FP2"),
        ("FP1", "AF3"),
        ("FP1", "AF4"),
        ("FP1", "F7"),
    )
    assert edges[60:64] == (
        ("FP1", "CB2"),
        ("FPZ", "FP2"),
        ("FPZ", "AF3"),
        ("FPZ", "AF4"),
    )
    assert edges[-3:] == (
        ("OZ", "O2"),
        ("OZ", "CB2"),
        ("O2", "CB2"),
    )


def test_connectivity_bands_are_fixed_and_ordered() -> None:
    bands = connectivity_bands()

    assert bands == (
        ("Delta", (1.0, 3.0)),
        ("Theta", (4.0, 7.0)),
        ("Alpha", (8.0, 13.0)),
        ("Beta Low", (13.0, 18.0)),
        ("Beta Medium", (18.0, 21.0)),
        ("Beta High", (21.0, 30.0)),
    )


def test_single_state_connectivity_has_fixed_shapes_and_finite_values() -> None:
    feature = compute_single_state_connectivity(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EO",
        source_set_path=Path("source/sub01_eo1.set"),
        config=TEST_CONFIG,
    )

    assert feature.wpli.shape == (1891, 6)
    assert feature.imaginary_coherence.shape == (1891, 6)
    assert np.isfinite(feature.wpli).all()
    assert np.isfinite(feature.imaginary_coherence).all()
    assert feature.edge_list == build_edge_list()
    assert feature.band_names == tuple(name for name, _ in connectivity_bands())
    assert feature.band_ranges_hz == tuple(ranges for _, ranges in connectivity_bands())


def test_write_connectivity_feature_saves_eo_and_ec_separately_with_metadata(tmp_path: Path) -> None:
    eo = compute_single_state_connectivity(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EO",
        source_set_path=Path("source/sub01_eo1.set"),
        config=TEST_CONFIG,
    )
    ec = compute_single_state_connectivity(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EC",
        source_set_path=Path("source/sub01_ec2.set"),
        config=TEST_CONFIG,
    )

    eo_path = write_connectivity_feature(eo, tmp_path)
    ec_path = write_connectivity_feature(ec, tmp_path)

    assert eo_path == tmp_path / "data" / "features" / "fc" / "sub01_EO_fc.npz"
    assert ec_path == tmp_path / "data" / "features" / "fc" / "sub01_EC_fc.npz"
    assert eo_path != ec_path
    assert not (tmp_path / "sub01_EO_fc.npz").exists()

    with np.load(eo_path, allow_pickle=False) as saved:
        assert saved["wpli"].shape == (1891, 6)
        assert saved["imaginary_coherence"].shape == (1891, 6)
        assert saved["subject_id"].item() == "sub01"
        assert saved["state"].item() == "EO"
        assert saved["channel_names_after_alignment"].tolist() == list(CHANNELS)
        assert saved["band_names"].tolist() == [name for name, _ in connectivity_bands()]
        np.testing.assert_allclose(
            saved["band_ranges_hz"],
            np.array([ranges for _, ranges in connectivity_bands()]),
        )
        assert [tuple(edge) for edge in saved["edge_list"].tolist()] == list(build_edge_list())
        assert saved["sampling_rate"].item() == SAMPLING_RATE
        assert saved["source_set_path"].item() == "source/sub01_eo1.set"
        assert saved["affected_hand"].item() == "右"
        assert saved["hemisphere_aligned"].item()


def test_left_affected_hand_flips_before_connectivity_computation() -> None:
    data = _synthetic_data()
    left_feature = compute_single_state_connectivity(
        data,
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="左",
        subject_id="sub05",
        state="EO",
        source_set_path=Path("source/sub05_eo1.set"),
        config=TEST_CONFIG,
    )
    manually_flipped = flip_channels_for_affected_hand(
        data,
        CHANNELS,
        affected_hand="左",
        channel_axis=0,
    )
    right_feature_after_manual_flip = compute_single_state_connectivity(
        manually_flipped,
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub05",
        state="EO",
        source_set_path=Path("source/sub05_eo1.set"),
        config=TEST_CONFIG,
    )

    np.testing.assert_allclose(left_feature.wpli, right_feature_after_manual_flip.wpli)
    np.testing.assert_allclose(
        left_feature.imaginary_coherence,
        right_feature_after_manual_flip.imaginary_coherence,
    )


def test_compute_single_state_connectivity_rejects_noncanonical_channel_order() -> None:
    shuffled = list(CHANNELS)
    shuffled[25], shuffled[29] = shuffled[29], shuffled[25]

    with pytest.raises(ChannelMappingError, match="fixed 62-channel order"):
        compute_single_state_connectivity(
            _synthetic_data(),
            SAMPLING_RATE,
            shuffled,
            affected_hand="右",
            subject_id="sub01",
            state="EO",
            source_set_path=Path("source/sub01_eo1.set"),
            config=TEST_CONFIG,
        )


def test_compute_single_state_connectivity_rejects_too_few_samples() -> None:
    with pytest.raises(ConnectivityFeatureError, match="samples"):
        compute_single_state_connectivity(
            _synthetic_data(samples=127),
            SAMPLING_RATE,
            CHANNELS,
            affected_hand="右",
            subject_id="sub01",
            state="EO",
            source_set_path=Path("source/sub01_eo1.set"),
            config=TEST_CONFIG,
        )
