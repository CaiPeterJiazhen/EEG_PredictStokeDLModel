from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_recovery.training.segment_ssl_dataset import (
    SegmentSSLRecord,
    SegmentSSLScaler,
    SegmentSSLDataset,
    load_segment_ssl_records,
    merge_segment_modalities,
    segment_records_for_scope,
)


def _write_segment(
    root: Path,
    *,
    subject_id: str,
    subject_key: str,
    group: str,
    stage: str,
    state: str,
    segment_index: int,
    value: float,
) -> Path:
    path = root / f"{subject_id}_{stage}_{state}_{segment_index}.npz"
    np.savez_compressed(
        path,
        psd=np.full((62, 90), value, dtype=np.float32),
        subject_id=np.array(subject_id),
        subject_key=np.array(subject_key),
        group=np.array(group),
        stage=np.array(stage),
        state=np.array(state),
        segment_index=np.array(segment_index),
        start_sample=np.array(segment_index * 100),
        end_sample=np.array(segment_index * 100 + 100),
        affected_hand_aligned=np.array(True),
        source_set_path=np.array(f"source/{subject_id}_{state}.set"),
    )
    return path


def _write_fc_segment(
    root: Path,
    *,
    subject_id: str,
    subject_key: str,
    group: str,
    stage: str,
    state: str,
    segment_index: int,
    value: float,
) -> Path:
    path = root / f"{subject_id}_{stage}_{state}_{segment_index}_fc.npz"
    np.savez_compressed(
        path,
        wpli=np.full((1891, 6), value, dtype=np.float32),
        imaginary_coherence=np.full((1891, 6), value + 1.0, dtype=np.float32),
        subject_id=np.array(subject_id),
        subject_key=np.array(subject_key),
        group=np.array(group),
        stage=np.array(stage),
        state=np.array(state),
        segment_index=np.array(segment_index),
        start_sample=np.array(segment_index * 100),
        end_sample=np.array(segment_index * 100 + 100),
        affected_hand_aligned=np.array(True),
        source_set_path=np.array(f"source/{subject_id}_{state}.set"),
    )
    return path


def test_load_segment_ssl_records_reads_npz_metadata(tmp_path: Path) -> None:
    segment_dir = tmp_path / "segments"
    segment_dir.mkdir()
    _write_segment(
        segment_dir,
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        segment_index=0,
        value=1.0,
    )

    records = load_segment_ssl_records(segment_dir)

    assert len(records) == 1
    assert records[0].subject_id == "sub01"
    assert records[0].subject_key == "patient:sub01"
    assert records[0].group == "patient"
    assert records[0].stage == "baseline"
    assert records[0].state == "EO"
    assert records[0].segment_index == 0
    assert records[0].features["psd"].shape == (62, 90)


def test_load_segment_ssl_records_maps_fc_cache_keys_to_wpli_and_icoh(tmp_path: Path) -> None:
    segment_dir = tmp_path / "fc"
    segment_dir.mkdir()
    _write_fc_segment(
        segment_dir,
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EC",
        segment_index=0,
        value=2.0,
    )

    records = load_segment_ssl_records(segment_dir, branches=("wpli", "icoh"))

    assert len(records) == 1
    assert records[0].features["wpli"].shape == (1891, 6)
    assert records[0].features["icoh"].shape == (1891, 6)
    np.testing.assert_allclose(records[0].features["wpli"], np.full((1891, 6), 2.0, dtype=np.float32))
    np.testing.assert_allclose(records[0].features["icoh"], np.full((1891, 6), 3.0, dtype=np.float32))


def test_merge_segment_modalities_keeps_only_segments_present_in_both_caches(tmp_path: Path) -> None:
    psd_dir = tmp_path / "psd"
    fc_dir = tmp_path / "fc"
    psd_dir.mkdir()
    fc_dir.mkdir()
    _write_segment(
        psd_dir,
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        segment_index=0,
        value=1.0,
    )
    _write_segment(
        psd_dir,
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        segment_index=1,
        value=5.0,
    )
    _write_fc_segment(
        fc_dir,
        subject_id="sub01",
        subject_key="patient:sub01",
        group="patient",
        stage="baseline",
        state="EO",
        segment_index=0,
        value=2.0,
    )

    merged = merge_segment_modalities(
        load_segment_ssl_records(psd_dir, branches=("psd",)),
        load_segment_ssl_records(fc_dir, branches=("wpli", "icoh")),
        primary_branch="psd",
        secondary_branch=("wpli", "icoh"),
    )

    assert len(merged) == 1
    assert set(merged[0].features) == {"psd", "wpli", "icoh"}
    assert merged[0].segment_index == 0


def test_segment_records_for_scope_excludes_loso_test_subject() -> None:
    records = [
        SegmentSSLRecord("patient", "sub01", "patient:sub01", "baseline", "EO", 0, {"psd": np.zeros((62, 90), dtype=np.float32)}, Path("a")),
        SegmentSSLRecord("patient", "sub02", "patient:sub02", "baseline", "EO", 0, {"psd": np.zeros((62, 90), dtype=np.float32)}, Path("b")),
        SegmentSSLRecord("patient", "sub03", "patient:sub03", "followup", "EO", 0, {"psd": np.zeros((62, 90), dtype=np.float32)}, Path("c")),
        SegmentSSLRecord("health", "h01", "health:h01", "health", "EC", 0, {"psd": np.zeros((62, 90), dtype=np.float32)}, Path("d")),
    ]

    selected = segment_records_for_scope(
        records,
        data_scope="all-patient-health",
        strict_loso_test_subject_id="sub01",
        supervised_subject_ids={"sub01", "sub02"},
        baseline_stage="baseline",
    )

    assert {record.subject_key for record in selected} == {
        "patient:sub02",
        "patient:sub03",
        "health:h01",
    }
    assert all(record.subject_id != "sub01" for record in selected)


def test_segment_ssl_dataset_fits_scaler_only_on_selected_records() -> None:
    train_records = [
        SegmentSSLRecord("patient", "sub01", "patient:sub01", "baseline", "EO", 0, {"psd": np.full((62, 90), 1.0, dtype=np.float32)}, Path("a")),
        SegmentSSLRecord("patient", "sub02", "patient:sub02", "baseline", "EO", 0, {"psd": np.full((62, 90), 3.0, dtype=np.float32)}, Path("b")),
    ]
    leaked_test_record = SegmentSSLRecord(
        "patient",
        "sub99",
        "patient:sub99",
        "baseline",
        "EO",
        0,
        {"psd": np.full((62, 90), 1000.0, dtype=np.float32)},
        Path("c"),
    )

    scaler = SegmentSSLScaler.fit(train_records, branches=("psd",))
    dataset = SegmentSSLDataset([*train_records, leaked_test_record], branches=("psd",), scaler=scaler)

    first = dataset[0]
    second = dataset[1]

    assert first["psd"].shape == (62, 90)
    np.testing.assert_allclose(first["psd"].numpy(), np.full((62, 90), -1.0, dtype=np.float32))
    np.testing.assert_allclose(second["psd"].numpy(), np.full((62, 90), 1.0, dtype=np.float32))
    assert dataset.records[2].subject_id == "sub99"


def test_segment_ssl_dataset_rejects_missing_branch() -> None:
    record = SegmentSSLRecord("patient", "sub01", "patient:sub01", "baseline", "EO", 0, {}, Path("a"))

    with pytest.raises(ValueError, match="branch"):
        SegmentSSLScaler.fit([record], branches=("psd",))
