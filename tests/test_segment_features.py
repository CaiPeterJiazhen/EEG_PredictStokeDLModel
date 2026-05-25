from __future__ import annotations

from pathlib import Path
import os
import time

import numpy as np

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.features.segment_features import (
    SegmentFeatureConfig,
    compute_segment_fc_features,
    compute_segment_psd_features,
    is_segment_cache_fresh,
    segment_cache_path,
    slice_fixed_windows,
    write_segment_fc_feature,
    write_segment_psd_feature,
)


SAMPLING_RATE = 250.0


def _synthetic_eeg(samples: int = 3000) -> np.ndarray:
    time_axis = np.arange(samples) / SAMPLING_RATE
    data = np.zeros((62, samples), dtype=np.float32)
    for channel_index in range(62):
        data[channel_index] = np.sin(2 * np.pi * (2.0 + channel_index % 4) * time_axis)
    return data


def test_slice_fixed_windows_uses_seconds_and_overlap() -> None:
    windows = slice_fixed_windows(
        n_samples=2500,
        sampling_rate=250.0,
        window_seconds=4.0,
        overlap_fraction=0.5,
    )

    assert [(window.segment_index, window.start_sample, window.end_sample) for window in windows] == [
        (0, 0, 1000),
        (1, 500, 1500),
        (2, 1000, 2000),
        (3, 1500, 2500),
    ]


def test_compute_segment_psd_features_preserves_metadata_and_shape(tmp_path: Path) -> None:
    source_path = tmp_path / "sub01_eo1.set"
    source_path.write_text("metadata", encoding="utf-8")
    config = SegmentFeatureConfig(psd_window_seconds=4.0, psd_overlap_fraction=0.5)

    features = compute_segment_psd_features(
        data=_synthetic_eeg(samples=3000),
        sampling_rate=SAMPLING_RATE,
        channel_names=CANONICAL_CHANNELS_62,
        affected_hand="right",
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        source_set_path=source_path,
        config=config,
    )

    assert len(features) == 5
    assert features[0].psd.shape == (62, 90)
    assert features[0].metadata["subject_id"] == "sub01"
    assert features[0].metadata["subject_key"] == "patient:sub01"
    assert features[0].metadata["group"] == "patient"
    assert features[0].metadata["stage"] == "baseline"
    assert features[0].metadata["state"] == "EO"
    assert features[0].metadata["segment_index"] == 0
    assert features[0].metadata["start_sample"] == 0
    assert features[0].metadata["end_sample"] == 1000
    assert features[0].metadata["affected_hand_aligned"] is True
    assert features[0].metadata["source_set_path"] == str(source_path)
    assert features[0].metadata["psd_window_seconds"] == 4.0
    assert features[0].metadata["psd_overlap_fraction"] == 0.5


def test_segment_psd_cache_path_and_source_mtime_invalidation(tmp_path: Path) -> None:
    source_path = tmp_path / "sub01_eo1.set"
    source_path.write_text("first", encoding="utf-8")
    feature = compute_segment_psd_features(
        data=_synthetic_eeg(samples=1000),
        sampling_rate=SAMPLING_RATE,
        channel_names=CANONICAL_CHANNELS_62,
        affected_hand="right",
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        source_set_path=source_path,
        config=SegmentFeatureConfig(psd_window_seconds=4.0, psd_overlap_fraction=0.0),
    )[0]

    output_path = write_segment_psd_feature(feature, tmp_path)

    assert output_path == segment_cache_path(tmp_path, "psd", feature.metadata)
    assert "segment_level" in output_path.parts
    assert output_path.parent == tmp_path / "data" / "features" / "segment_level" / "psd"
    assert is_segment_cache_fresh(output_path)

    time.sleep(0.01)
    source_path.write_text("second", encoding="utf-8")
    os.utime(source_path, None)

    assert not is_segment_cache_fresh(output_path)


def test_compute_segment_fc_features_preserves_metadata_and_shape(tmp_path: Path) -> None:
    source_path = tmp_path / "sub01_eo1.set"
    source_path.write_text("metadata", encoding="utf-8")
    config = SegmentFeatureConfig(fc_window_seconds=8.0, fc_overlap_fraction=0.0)

    features = compute_segment_fc_features(
        data=_synthetic_eeg(samples=2000),
        sampling_rate=SAMPLING_RATE,
        channel_names=CANONICAL_CHANNELS_62,
        affected_hand="right",
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        source_set_path=source_path,
        config=config,
    )

    assert len(features) == 1
    assert features[0].wpli.shape == (1891, 6)
    assert features[0].imaginary_coherence.shape == (1891, 6)
    assert features[0].edge_list.shape == (1891, 2)
    assert features[0].band_names.shape == (6,)
    assert features[0].band_ranges_hz.shape == (6, 2)
    assert features[0].metadata["subject_id"] == "sub01"
    assert features[0].metadata["subject_key"] == "patient:sub01"
    assert features[0].metadata["group"] == "patient"
    assert features[0].metadata["stage"] == "baseline"
    assert features[0].metadata["state"] == "EO"
    assert features[0].metadata["segment_index"] == 0
    assert features[0].metadata["start_sample"] == 0
    assert features[0].metadata["end_sample"] == 2000
    assert features[0].metadata["affected_hand_aligned"] is True
    assert features[0].metadata["source_set_path"] == str(source_path)
    assert features[0].metadata["fc_window_seconds"] == 8.0
    assert features[0].metadata["fc_overlap_fraction"] == 0.0


def test_segment_fc_cache_path_and_source_mtime_invalidation(tmp_path: Path) -> None:
    source_path = tmp_path / "sub01_eo1.set"
    source_path.write_text("first", encoding="utf-8")
    feature = compute_segment_fc_features(
        data=_synthetic_eeg(samples=2000),
        sampling_rate=SAMPLING_RATE,
        channel_names=CANONICAL_CHANNELS_62,
        affected_hand="right",
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        source_set_path=source_path,
        config=SegmentFeatureConfig(fc_window_seconds=8.0, fc_overlap_fraction=0.0),
    )[0]

    output_path = write_segment_fc_feature(feature, tmp_path)

    assert output_path == segment_cache_path(tmp_path, "fc", feature.metadata)
    assert "segment_level" in output_path.parts
    assert output_path.parent == tmp_path / "data" / "features" / "segment_level" / "fc"
    assert is_segment_cache_fresh(output_path)

    time.sleep(0.01)
    source_path.write_text("second", encoding="utf-8")
    os.utime(source_path, None)

    assert not is_segment_cache_fresh(output_path)


def test_segment_cache_path_accepts_non_ascii_stage_names(tmp_path: Path) -> None:
    metadata = {
        "subject_key": "patient:sub01",
        "stage": "基线",
        "state": "EO",
        "segment_index": 3,
    }

    path = segment_cache_path(tmp_path, "psd", metadata)

    assert path.parent == tmp_path / "data" / "features" / "segment_level" / "psd"
    assert path.name.startswith("patient_sub01_")
    assert path.name.endswith("_EO_seg0003_psd.npz")
