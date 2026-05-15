from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from eeg_recovery.config import ConfigError, load_path_config
from eeg_recovery.io.eeglab import (
    EEGLABDataError,
    EEGLABMetadata,
    read_eeglab_fdt,
    read_eeglab_set_metadata,
    resolve_eeglab_fdt_path,
)
from eeg_recovery.io.index import EEGIndexError, build_eeg_file_index
from eeg_recovery.metadata.labels import MetadataError, load_supervised_label_table


def baseline_supervised_records():
    try:
        config = load_path_config(Path("configs/paths.example.yaml"))
        labels = load_supervised_label_table(config)
        records = build_eeg_file_index(
            config.patient_eeg_root,
            config.health_eeg_root,
            supervised_subject_ids=labels["subject_id"].tolist(),
        )
    except (ConfigError, MetadataError, EEGIndexError) as error:
        pytest.skip(f"External EEG paths are unavailable: {error}")

    baseline = [
        record
        for record in records
        if record.group == "patient"
        and record.stage == "基线"
        and record.is_supervised_subject
    ]
    if not baseline:
        pytest.skip("No supervised baseline EEG records found in configured paths")
    return baseline[:4]


def write_synthetic_set(
    set_path: Path,
    *,
    data_file: str = "custom_data.fdt",
    ch_names: tuple[str, ...] = ("FP1", "FPZ", "FP2"),
) -> None:
    chanlocs = np.empty((1, len(ch_names)), dtype=[("labels", "O")])
    for index, name in enumerate(ch_names):
        chanlocs[0, index]["labels"] = name

    savemat(
        set_path,
        {
            "nbchan": len(ch_names),
            "srate": 250,
            "trials": 1,
            "pnts": 4,
            "data": data_file,
            "chanlocs": chanlocs,
            "xmin": 0,
            "xmax": 0.012,
        },
    )


def eeglab_chanlocs(ch_names: tuple[str, ...]) -> np.ndarray:
    chanlocs = np.empty((1, len(ch_names)), dtype=[("labels", "O")])
    for index, name in enumerate(ch_names):
        chanlocs[0, index]["labels"] = name
    return chanlocs


def test_read_eeglab_set_metadata_parses_top_level_fields_and_relative_data_file(
    tmp_path: Path,
) -> None:
    set_path = tmp_path / "sample.set"
    write_synthetic_set(set_path, data_file="nested/not_same_stem.fdt")

    metadata = read_eeglab_set_metadata(set_path)

    assert metadata.set_path == set_path
    assert metadata.fdt_path == tmp_path / "nested" / "not_same_stem.fdt"
    assert metadata.nbchan == 3
    assert metadata.srate == 250
    assert metadata.trials == 1
    assert metadata.pnts == 4
    assert metadata.xmin == 0
    assert metadata.xmax == 0.012
    assert metadata.ch_names == ("FP1", "FPZ", "FP2")
    assert metadata.data_file == "nested/not_same_stem.fdt"


def test_read_eeglab_set_metadata_falls_back_to_existing_companion_for_stale_data_file(
    tmp_path: Path,
) -> None:
    set_path = tmp_path / "sample.set"
    write_synthetic_set(set_path, data_file="stale_name.fdt")
    companion_fdt = tmp_path / "sample.fdt"
    companion_fdt.write_bytes(b"placeholder")

    metadata = read_eeglab_set_metadata(set_path)

    assert metadata.fdt_path == companion_fdt
    assert metadata.data_file == "stale_name.fdt"


def test_read_eeglab_set_metadata_parses_nested_eeg_struct(tmp_path: Path) -> None:
    set_path = tmp_path / "nested_struct.set"
    savemat(
        set_path,
        {
            "EEG": {
                "nbchan": 2,
                "srate": 500,
                "trials": 3,
                "pnts": 4,
                "data": "nested_struct.fdt",
                "chanlocs": eeglab_chanlocs(("FZ", "CZ")),
                "xmin": -0.1,
                "xmax": 0.2,
            }
        },
    )

    metadata = read_eeglab_set_metadata(set_path)

    assert metadata.fdt_path == tmp_path / "nested_struct.fdt"
    assert metadata.nbchan == 2
    assert metadata.srate == 500
    assert metadata.trials == 3
    assert metadata.pnts == 4
    assert metadata.xmin == -0.1
    assert metadata.xmax == 0.2
    assert metadata.ch_names == ("FZ", "CZ")


def test_read_eeglab_set_metadata_raises_domain_error_for_missing_set(
    tmp_path: Path,
) -> None:
    missing_set = tmp_path / "missing.set"

    with pytest.raises(EEGLABDataError, match=r"Could not read EEGLAB \.set.*missing\.set"):
        read_eeglab_set_metadata(missing_set)


def test_read_eeglab_fdt_raises_domain_error_for_missing_fdt(tmp_path: Path) -> None:
    missing_fdt = tmp_path / "missing.fdt"
    metadata = EEGLABMetadata(
        set_path=tmp_path / "sample.set",
        fdt_path=missing_fdt,
        nbchan=2,
        srate=250.0,
        trials=1,
        pnts=3,
        xmin=0.0,
        xmax=1.0,
        ch_names=("C1", "C2"),
        data_file="missing.fdt",
    )

    with pytest.raises(EEGLABDataError, match=r"Could not read EEGLAB \.fdt.*missing\.fdt"):
        read_eeglab_fdt(metadata, subject_id="sub01", state="EC")


def test_resolve_eeglab_fdt_path_uses_companion_when_data_file_is_absent(
    tmp_path: Path,
) -> None:
    set_path = tmp_path / "sample.set"

    assert resolve_eeglab_fdt_path(set_path) == tmp_path / "sample.fdt"


def test_read_eeglab_set_metadata_from_real_baseline_files() -> None:
    metadata = [
        read_eeglab_set_metadata(record.set_path)
        for record in baseline_supervised_records()
    ]

    first_channel_order = metadata[0].ch_names
    for item in metadata:
        assert item.nbchan == 62
        assert item.srate == 250
        assert item.trials == 1
        assert len(item.ch_names) == 62
        assert "M1" not in item.ch_names
        assert "M2" not in item.ch_names
        assert item.ch_names == first_channel_order
        assert item.fdt_path == item.set_path.with_suffix(".fdt")


def test_read_eeglab_fdt_returns_channels_by_samples(tmp_path: Path) -> None:
    fdt_path = tmp_path / "sample.fdt"
    raw = np.arange(12, dtype=np.float32)
    raw.tofile(fdt_path)
    metadata = EEGLABMetadata(
        set_path=tmp_path / "sample.set",
        fdt_path=fdt_path,
        nbchan=2,
        srate=250.0,
        trials=2,
        pnts=3,
        xmin=0.0,
        xmax=1.0,
        ch_names=("C1", "C2"),
        data_file="sample.fdt",
    )

    data = read_eeglab_fdt(metadata)

    assert data.shape == (2, 6)
    np.testing.assert_array_equal(
        data,
        np.array(
            [
                [0, 2, 4, 6, 8, 10],
                [1, 3, 5, 7, 9, 11],
            ],
            dtype=np.float32,
        ),
    )


def test_read_eeglab_fdt_reports_length_mismatch_with_context(tmp_path: Path) -> None:
    fdt_path = tmp_path / "short.fdt"
    np.arange(5, dtype=np.float32).tofile(fdt_path)
    metadata = EEGLABMetadata(
        set_path=tmp_path / "short.set",
        fdt_path=fdt_path,
        nbchan=2,
        srate=250.0,
        trials=2,
        pnts=3,
        xmin=0.0,
        xmax=1.0,
        ch_names=("C1", "C2"),
        data_file="short.fdt",
    )

    with pytest.raises(EEGLABDataError) as error:
        read_eeglab_fdt(metadata, subject_id="sub01", state="EO")

    message = str(error.value)
    assert "sub01" in message
    assert "EO" in message
    assert str(fdt_path) in message
    assert "expected 12 float32 samples" in message
    assert "observed 5" in message
