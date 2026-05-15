from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62, ChannelMappingError
from eeg_recovery.features.psd import (
    PSDFeature,
    PSDFeatureError,
    compute_single_state_psd,
    target_frequency_bins,
    write_psd_feature,
)


SAMPLING_RATE = 250.0
CHANNELS = CANONICAL_CHANNELS_62


def _sine(freq_hz: float, samples: int = 1000) -> np.ndarray:
    time = np.arange(samples) / SAMPLING_RATE
    return np.sin(2 * np.pi * freq_hz * time)


def _synthetic_data(samples: int = 1000) -> np.ndarray:
    data = np.zeros((len(CHANNELS), samples), dtype=float)
    for index in range(len(CHANNELS)):
        data[index] = 0.01 * _sine(2.0 + (index % 5), samples=samples)
    return data


def _peak_frequency(psd: np.ndarray) -> float:
    bins = target_frequency_bins()
    return float(bins[int(np.argmax(psd))])


def test_single_state_psd_has_fixed_62_by_90_shape() -> None:
    feature = compute_single_state_psd(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EO",
        source_set_path=Path("source/sub01_eo1.set"),
    )

    assert feature.psd.shape == (62, 90)


def test_target_frequency_bins_cover_0_5_to_45_hz_at_0_5_resolution() -> None:
    bins = target_frequency_bins()

    assert bins.shape == (90,)
    np.testing.assert_allclose(bins[0], 0.5)
    np.testing.assert_allclose(bins[-1], 45.0)
    np.testing.assert_allclose(np.diff(bins), 0.5)


def test_write_psd_feature_saves_eo_and_ec_separately_with_metadata(tmp_path: Path) -> None:
    eo = compute_single_state_psd(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EO",
        source_set_path=Path("source/sub01_eo1.set"),
    )
    ec = compute_single_state_psd(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EC",
        source_set_path=Path("source/sub01_ec2.set"),
    )

    eo_path = write_psd_feature(eo, tmp_path)
    ec_path = write_psd_feature(ec, tmp_path)

    assert eo_path == tmp_path / "data" / "features" / "psd" / "sub01_EO_psd.npz"
    assert ec_path == tmp_path / "data" / "features" / "psd" / "sub01_EC_psd.npz"
    assert eo_path != ec_path
    assert not (tmp_path / "sub01_EO_psd.npz").exists()

    with np.load(eo_path, allow_pickle=False) as saved:
        assert saved["psd"].shape == (62, 90)
        assert saved["subject_id"].item() == "sub01"
        assert saved["state"].item() == "EO"
        assert saved["channel_names_after_alignment"].tolist() == list(CHANNELS)
        np.testing.assert_allclose(saved["frequency_bins"], target_frequency_bins())
        assert saved["sampling_rate"].item() == SAMPLING_RATE
        assert saved["source_set_path"].item() == "source/sub01_eo1.set"
        assert saved["affected_hand"].item() == "右"
        assert saved["hemisphere_aligned"].item()
        assert "welch_nperseg" in saved
        assert "welch_noverlap" in saved
        assert "welch_window" in saved
        assert "welch_scaling" in saved


def test_left_affected_hand_flips_channels_before_psd_computation() -> None:
    data = _synthetic_data()
    c3_index = CHANNELS.index("C3")
    c4_index = CHANNELS.index("C4")
    data[c3_index] = _sine(10.0)
    data[c4_index] = _sine(20.0)

    feature = compute_single_state_psd(
        data,
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="左",
        subject_id="sub05",
        state="EO",
        source_set_path=Path("source/sub05_eo1.set"),
    )

    assert _peak_frequency(feature.psd[c3_index]) == 20.0
    assert _peak_frequency(feature.psd[c4_index]) == 10.0


def test_right_affected_hand_preserves_c3_c4_psd_peaks() -> None:
    data = _synthetic_data()
    c3_index = CHANNELS.index("C3")
    c4_index = CHANNELS.index("C4")
    data[c3_index] = _sine(10.0)
    data[c4_index] = _sine(20.0)

    feature = compute_single_state_psd(
        data,
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub07",
        state="EO",
        source_set_path=Path("source/sub07_eo1.set"),
    )

    assert feature.metadata["affected_hand"] == "右"
    assert feature.metadata["hemisphere_aligned"] is True
    assert _peak_frequency(feature.psd[c3_index]) == 10.0
    assert _peak_frequency(feature.psd[c4_index]) == 20.0


def test_write_psd_feature_rejects_invalid_shape_without_creating_output(tmp_path: Path) -> None:
    feature = compute_single_state_psd(
        _synthetic_data(),
        SAMPLING_RATE,
        CHANNELS,
        affected_hand="右",
        subject_id="sub01",
        state="EO",
        source_set_path=Path("source/sub01_eo1.set"),
    )
    malformed = PSDFeature(
        psd=feature.psd[:-1],
        frequency_bins=feature.frequency_bins,
        subject_id=feature.subject_id,
        state=feature.state,
        channel_names_after_alignment=feature.channel_names_after_alignment,
        sampling_rate=feature.sampling_rate,
        source_set_path=feature.source_set_path,
        metadata=feature.metadata,
    )

    with pytest.raises(PSDFeatureError, match="PSD shape"):
        write_psd_feature(malformed, tmp_path)

    assert not (tmp_path / "data" / "features" / "psd" / "sub01_EO_psd.npz").exists()


def test_compute_single_state_psd_rejects_too_few_samples() -> None:
    with pytest.raises(PSDFeatureError, match="samples"):
        compute_single_state_psd(
            _synthetic_data(samples=499),
            SAMPLING_RATE,
            CHANNELS,
            affected_hand="右",
            subject_id="sub01",
            state="EO",
            source_set_path=Path("source/sub01_eo1.set"),
        )


def test_compute_single_state_psd_rejects_noncanonical_channel_order() -> None:
    shuffled = list(CHANNELS)
    shuffled[25], shuffled[29] = shuffled[29], shuffled[25]

    with pytest.raises(ChannelMappingError, match="fixed 62-channel order"):
        compute_single_state_psd(
            _synthetic_data(),
            SAMPLING_RATE,
            shuffled,
            affected_hand="右",
            subject_id="sub01",
            state="EO",
            source_set_path=Path("source/sub01_eo1.set"),
        )
