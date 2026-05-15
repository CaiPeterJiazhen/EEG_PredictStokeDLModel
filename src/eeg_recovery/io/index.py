from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from eeg_recovery.io.eeglab import (
    EEGLABDataError,
    read_eeglab_set_metadata,
)
from eeg_recovery.metadata.subjects import normalize_subject_id


EEGGroup = Literal["patient", "health"]
EEGState = Literal["EO", "EC"]


class EEGIndexError(ValueError):
    """Raised when EEG file indexing finds malformed or incomplete inputs."""


@dataclass(frozen=True)
class EEGFileRecord:
    group: EEGGroup
    subject_id: str
    subject_key: str
    stage: str
    state: EEGState
    set_path: Path
    fdt_path: Path
    is_supervised_subject: bool


def build_eeg_file_index(
    patient_eeg_root: str | Path,
    health_eeg_root: str | Path,
    supervised_subject_ids: Iterable[str] = (),
    *,
    validate_supervised_baseline: bool = True,
) -> list[EEGFileRecord]:
    """Build an index of patient and health EEGLAB ``.set`` files."""

    patient_root = Path(patient_eeg_root)
    health_root = Path(health_eeg_root)
    supervised_ids = _normalize_supervised_ids(supervised_subject_ids)

    records = [
        *_index_patient_records(patient_root, supervised_ids),
        *_index_health_records(health_root, supervised_ids),
    ]

    if validate_supervised_baseline:
        validate_supervised_baseline_coverage(records, supervised_ids)

    # Keep output deterministic for downstream feature generation and tests.
    return sorted(
        records,
        key=lambda record: (
            record.group,
            record.stage,
            record.subject_key,
            record.state,
            str(record.set_path),
        ),
    )


def validate_supervised_baseline_coverage(
    records: Iterable[EEGFileRecord],
    supervised_subject_ids: Iterable[str],
) -> None:
    """Require exactly one EO and one EC baseline file for each supervised subject."""

    supervised_ids = _normalize_supervised_ids(supervised_subject_ids)
    baseline_counts: dict[str, Counter[str]] = {
        subject_id: Counter()
        for subject_id in supervised_ids
    }

    for record in records:
        if (
            record.group == "patient"
            and record.stage == "基线"
            and record.subject_id in baseline_counts
        ):
            baseline_counts[record.subject_id][record.state] += 1

    invalid: list[str] = []
    for subject_id in sorted(supervised_ids):
        counts = baseline_counts[subject_id]
        if counts["EO"] != 1 or counts["EC"] != 1:
            invalid.append(
                f"{subject_id} 基线 requires exactly one EO and one EC file "
                f"(found EO={counts['EO']}, EC={counts['EC']})"
            )

    if invalid:
        raise EEGIndexError("; ".join(invalid))


def _index_patient_records(
    patient_root: Path,
    supervised_ids: set[str],
) -> list[EEGFileRecord]:
    if not patient_root.exists():
        raise EEGIndexError(f"Patient EEG root does not exist: {patient_root}")
    if not patient_root.is_dir():
        raise EEGIndexError(f"Patient EEG root is not a directory: {patient_root}")

    records: list[EEGFileRecord] = []
    for stage_dir in _sorted_dirs(patient_root):
        stage = stage_dir.name
        for subject_dir in _sorted_dirs(stage_dir):
            subject_id = _normalize_patient_subject_dir(subject_dir)
            for set_path in _sorted_set_files(subject_dir):
                records.append(
                    _make_record(
                        group="patient",
                        subject_id=subject_id,
                        stage=stage,
                        set_path=set_path,
                        supervised_ids=supervised_ids,
                    )
                )
    return records


def _index_health_records(
    health_root: Path,
    supervised_ids: set[str],
) -> list[EEGFileRecord]:
    if not health_root.exists():
        raise EEGIndexError(f"Health EEG root does not exist: {health_root}")
    if not health_root.is_dir():
        raise EEGIndexError(f"Health EEG root is not a directory: {health_root}")

    records: list[EEGFileRecord] = []
    files_by_subject: dict[str, list[Path]] = defaultdict(list)
    for set_path in _sorted_set_files(health_root):
        subject_id = _health_subject_id(health_root, set_path)
        files_by_subject[subject_id].append(set_path)

    for subject_id in sorted(files_by_subject):
        for set_path in files_by_subject[subject_id]:
            records.append(
                _make_record(
                    group="health",
                    subject_id=subject_id,
                    stage="health",
                    set_path=set_path,
                    supervised_ids=supervised_ids,
                )
            )
    return records


def _make_record(
    *,
    group: EEGGroup,
    subject_id: str,
    stage: str,
    set_path: Path,
    supervised_ids: set[str],
) -> EEGFileRecord:
    state = _state_from_set_path(set_path)
    try:
        metadata = read_eeglab_set_metadata(set_path)
    except EEGLABDataError as error:
        raise EEGIndexError(
            f"Could not read EEGLAB metadata for .set file {set_path}: {error}"
        ) from error

    fdt_path = metadata.fdt_path
    if not fdt_path.exists():
        raise EEGIndexError(
            f"Missing .fdt data file resolved from EEG .set file {set_path}: {fdt_path}"
        )

    return EEGFileRecord(
        group=group,
        subject_id=subject_id,
        subject_key=f"{group}:{subject_id}",
        stage=stage,
        state=state,
        set_path=set_path,
        fdt_path=fdt_path,
        is_supervised_subject=subject_id in supervised_ids,
    )


def _state_from_set_path(set_path: Path) -> EEGState:
    stem = set_path.stem
    if len(stem) >= 1 and stem[-1] == "1" and (len(stem) == 1 or not stem[-2].isdigit()):
        return "EO"
    if len(stem) >= 1 and stem[-1] == "2" and (len(stem) == 1 or not stem[-2].isdigit()):
        return "EC"
    raise EEGIndexError(
        f"Could not infer EO/EC state from .set filename suffix: {set_path}"
    )


def _normalize_supervised_ids(supervised_subject_ids: Iterable[str]) -> set[str]:
    return {
        normalize_subject_id(subject_id)
        for subject_id in supervised_subject_ids
    }


def _normalize_patient_subject_dir(subject_dir: Path) -> str:
    try:
        return normalize_subject_id(subject_dir.name)
    except ValueError as error:
        raise EEGIndexError(
            f"Could not parse patient subject ID from directory: {subject_dir}"
        ) from error


def _health_subject_id(health_root: Path, set_path: Path) -> str:
    relative = set_path.relative_to(health_root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return set_path.stem


def _sorted_dirs(path: Path) -> list[Path]:
    return sorted(
        (child for child in path.iterdir() if child.is_dir()),
        key=lambda child: child.name,
    )


def _sorted_set_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.set"), key=lambda child: str(child))
