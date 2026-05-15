from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from eeg_recovery.config import ConfigError, load_path_config
from eeg_recovery.io.index import (
    EEGIndexError,
    build_eeg_file_index,
    validate_supervised_baseline_coverage,
)
from eeg_recovery.metadata.labels import load_supervised_label_table


PATIENT_STAGES = ("基线", "即时", "阶段", "最终")


def write_eeg_pair(
    subject_dir: Path,
    filename_stem: str,
    *,
    data_file: str | None = None,
) -> tuple[Path, Path]:
    subject_dir.mkdir(parents=True, exist_ok=True)
    set_path = subject_dir / f"{filename_stem}.set"
    data_file = data_file or f"{filename_stem}.fdt"
    fdt_path = subject_dir / data_file
    fdt_path.parent.mkdir(parents=True, exist_ok=True)

    chanlocs = np.empty((1, 2), dtype=[("labels", "O")])
    chanlocs[0, 0]["labels"] = "C1"
    chanlocs[0, 1]["labels"] = "C2"
    savemat(
        set_path,
        {
            "nbchan": 2,
            "srate": 250,
            "trials": 1,
            "pnts": 3,
            "data": data_file,
            "chanlocs": chanlocs,
            "xmin": 0,
            "xmax": 0.008,
        },
    )
    fdt_path.write_bytes(b"placeholder")
    return set_path, fdt_path


def write_synthetic_eeg_tree(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    patient_root = tmp_path / "patient_eeg"
    health_root = tmp_path / "health_eeg"
    supervised_ids = ["sub01", "sub05"]

    for stage in PATIENT_STAGES:
        for subject_dir_name in ("sub01", "sub005"):
            subject_dir = patient_root / stage / subject_dir_name
            write_eeg_pair(subject_dir, f"{subject_dir_name}_{stage}_eo1")
            write_eeg_pair(subject_dir, f"{subject_dir_name}_{stage}_ec2")

    write_eeg_pair(health_root / "health01", "health01_rest_1")
    write_eeg_pair(health_root / "health01", "health01_rest_2")

    return patient_root, health_root, supervised_ids


def test_build_eeg_file_index_parses_patient_states_stages_and_health_subjects(
    tmp_path: Path,
) -> None:
    patient_root, health_root, supervised_ids = write_synthetic_eeg_tree(tmp_path)

    records = build_eeg_file_index(patient_root, health_root, supervised_ids)

    patient_records = [record for record in records if record.group == "patient"]
    health_records = [record for record in records if record.group == "health"]

    assert {record.stage for record in patient_records} == set(PATIENT_STAGES)
    assert {record.stage for record in health_records} == {"health"}
    assert {record.subject_id for record in health_records} == {"health01"}
    assert {record.group for record in records} == {"patient", "health"}

    for record in records:
        if record.set_path.stem.endswith("1"):
            assert record.state == "EO"
        elif record.set_path.stem.endswith("2"):
            assert record.state == "EC"
        else:
            raise AssertionError(f"Unexpected synthetic EEG filename: {record.set_path.name}")

    baseline = [
        record
        for record in patient_records
        if record.stage == "基线" and record.is_supervised_subject
    ]
    assert {
        (record.subject_id, record.state)
        for record in baseline
    } == {
        ("sub01", "EO"),
        ("sub01", "EC"),
        ("sub05", "EO"),
        ("sub05", "EC"),
    }
    assert all(record.fdt_path == record.set_path.with_suffix(".fdt") for record in records)
    assert all(record.fdt_path.exists() for record in records)


def test_build_eeg_file_index_uses_fdt_path_declared_in_set_metadata(
    tmp_path: Path,
) -> None:
    patient_root = tmp_path / "patient_eeg"
    health_root = tmp_path / "health_eeg"
    health_root.mkdir()
    subject_dir = patient_root / "基线" / "sub01"
    set_path, fdt_path = write_eeg_pair(
        subject_dir,
        "custom1",
        data_file="nested/not_same_stem.fdt",
    )

    records = build_eeg_file_index(
        patient_root,
        health_root,
        supervised_subject_ids=(),
        validate_supervised_baseline=False,
    )

    assert len(records) == 1
    assert records[0].set_path == set_path
    assert records[0].fdt_path == fdt_path
    assert not (subject_dir / "custom1.fdt").exists()


def test_build_eeg_file_index_wraps_unreadable_set_metadata(tmp_path: Path) -> None:
    patient_root = tmp_path / "patient_eeg"
    health_root = tmp_path / "health_eeg"
    health_root.mkdir()
    subject_dir = patient_root / "基线" / "sub01"
    subject_dir.mkdir(parents=True)
    set_path = subject_dir / "broken1.set"
    set_path.write_text("not a matlab file", encoding="utf-8")
    (subject_dir / "broken1.fdt").write_bytes(b"placeholder")

    with pytest.raises(EEGIndexError, match=rf"Could not read EEGLAB metadata.*{set_path.name}"):
        build_eeg_file_index(
            patient_root,
            health_root,
            supervised_subject_ids=(),
            validate_supervised_baseline=False,
        )


def test_build_eeg_file_index_validates_supervised_baseline_coverage(tmp_path: Path) -> None:
    patient_root, health_root, supervised_ids = write_synthetic_eeg_tree(tmp_path)
    missing_ec = patient_root / "基线" / "sub005" / "sub005_基线_ec2.set"
    missing_ec.unlink()

    with pytest.raises(EEGIndexError, match=r"sub05.*基线.*EO.*EC"):
        build_eeg_file_index(patient_root, health_root, supervised_ids)


def test_validate_supervised_baseline_coverage_rejects_duplicate_state_files(
    tmp_path: Path,
) -> None:
    patient_root, health_root, supervised_ids = write_synthetic_eeg_tree(tmp_path)
    write_eeg_pair(patient_root / "基线" / "sub01", "sub01_duplicate_1")

    records = build_eeg_file_index(
        patient_root,
        health_root,
        supervised_subject_ids=(),
        validate_supervised_baseline=False,
    )

    with pytest.raises(EEGIndexError, match=r"sub01.*基线.*exactly one EO and one EC"):
        validate_supervised_baseline_coverage(records, supervised_ids)


def test_build_eeg_file_index_raises_for_malformed_set_suffix(tmp_path: Path) -> None:
    patient_root, health_root, supervised_ids = write_synthetic_eeg_tree(tmp_path)
    write_eeg_pair(patient_root / "基线" / "sub01", "sub01_malformed_3")

    with pytest.raises(EEGIndexError, match=r"EO/EC.*sub01_malformed_3\.set"):
        build_eeg_file_index(patient_root, health_root, supervised_ids)


def test_build_eeg_file_index_rejects_ambiguous_numeric_state_suffix(tmp_path: Path) -> None:
    patient_root, health_root, supervised_ids = write_synthetic_eeg_tree(tmp_path)
    write_eeg_pair(patient_root / "基线" / "sub01", "session12")

    with pytest.raises(EEGIndexError, match=r"EO/EC.*session12\.set"):
        build_eeg_file_index(patient_root, health_root, supervised_ids)


def test_health_subject_key_does_not_collide_with_patient_subject_id(tmp_path: Path) -> None:
    patient_root, health_root, supervised_ids = write_synthetic_eeg_tree(tmp_path)
    write_eeg_pair(health_root / "sub001健康人", "mxg1")
    write_eeg_pair(health_root / "sub001健康人", "mxg2")

    records = build_eeg_file_index(patient_root, health_root, supervised_ids)

    patient_sub01 = [
        record
        for record in records
        if record.group == "patient" and record.subject_id == "sub01"
    ]
    health_sub001 = [
        record
        for record in records
        if record.group == "health" and record.subject_id == "sub001健康人"
    ]

    assert patient_sub01
    assert health_sub001
    assert {record.subject_key for record in patient_sub01} == {"patient:sub01"}
    assert {record.subject_key for record in health_sub001} == {"health:sub001健康人"}
    assert all(record.is_supervised_subject for record in patient_sub01)
    assert not any(record.is_supervised_subject for record in health_sub001)
    assert {
        record.subject_key
        for record in [*patient_sub01, *health_sub001]
    } == {"patient:sub01", "health:sub001健康人"}


def test_build_eeg_file_index_from_real_project_paths() -> None:
    try:
        config = load_path_config(Path("configs/paths.example.yaml"))
    except ConfigError as error:
        pytest.skip(f"External EEG paths are unavailable: {error}")

    labels = load_supervised_label_table(config)
    supervised_ids = labels["subject_id"].tolist()

    records = build_eeg_file_index(
        config.patient_eeg_root,
        config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
    )

    validate_supervised_baseline_coverage(records, supervised_ids)
    supervised_baseline = [
        record
        for record in records
        if record.group == "patient"
        and record.stage == "基线"
        and record.is_supervised_subject
    ]
    assert len(supervised_baseline) == 2 * len(supervised_ids)
    assert {record.state for record in supervised_baseline} == {"EO", "EC"}
    assert any(record.group == "health" for record in records)
