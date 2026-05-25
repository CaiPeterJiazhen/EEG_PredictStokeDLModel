from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.train_ssl import normalize_ssl_data_scope


BASELINE_STAGE_ALIASES = {"baseline", "\u57fa\u7ebf", "鍩虹嚎"}


@dataclass(frozen=True)
class SegmentSSLRecord:
    group: str
    subject_id: str
    subject_key: str
    stage: str
    state: str
    segment_index: int
    features: dict[str, np.ndarray]
    source_path: Path


@dataclass(frozen=True)
class SegmentSSLScaler:
    parameters: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def fit(
        cls,
        records: Iterable[SegmentSSLRecord],
        *,
        branches: Iterable[str],
    ) -> SegmentSSLScaler:
        records = list(records)
        branch_list = tuple(branches)
        if not records:
            raise ValueError("SegmentSSLScaler requires at least one record.")
        if not branch_list:
            raise ValueError("SegmentSSLScaler requires at least one branch.")

        parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for branch in branch_list:
            arrays = []
            for record in records:
                if branch not in record.features:
                    raise ValueError(
                        f"Segment record {record.subject_key} segment {record.segment_index} is missing branch {branch!r}."
                    )
                arrays.append(np.asarray(record.features[branch], dtype=np.float32))
            stacked = np.stack(arrays).astype(np.float32)
            mean = stacked.mean(axis=0).astype(np.float32)
            std = stacked.std(axis=0).astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
            parameters[branch] = (mean, std)
        return cls(parameters=parameters)

    def transform(self, record: SegmentSSLRecord, branch: str) -> np.ndarray:
        if branch not in self.parameters:
            raise ValueError(f"No scaler fitted for branch {branch!r}.")
        if branch not in record.features:
            raise ValueError(
                f"Segment record {record.subject_key} segment {record.segment_index} is missing branch {branch!r}."
            )
        mean, std = self.parameters[branch]
        return ((record.features[branch].astype(np.float32) - mean) / std).astype(np.float32)


class SegmentSSLDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch dataset for fold-local segment-level SSL pretraining records."""

    def __init__(
        self,
        records: Iterable[SegmentSSLRecord],
        *,
        branches: Iterable[str],
        scaler: SegmentSSLScaler,
    ) -> None:
        self.records = list(records)
        self.branches = tuple(branches)
        self.scaler = scaler
        if not self.records:
            raise ValueError("SegmentSSLDataset requires at least one record.")
        if not self.branches:
            raise ValueError("SegmentSSLDataset requires at least one branch.")
        for record in self.records:
            for branch in self.branches:
                if branch not in record.features:
                    raise ValueError(
                        f"Segment record {record.subject_key} segment {record.segment_index} is missing branch {branch!r}."
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            branch: torch.as_tensor(self.scaler.transform(record, branch), dtype=torch.float32)
            for branch in self.branches
        }


def load_segment_ssl_records(segment_dir: str | Path, *, branches: Iterable[str] = ("psd",)) -> list[SegmentSSLRecord]:
    """Load cached segment-level feature records from one modality directory."""

    root = Path(segment_dir)
    if not root.exists():
        raise FileNotFoundError(f"Segment feature directory does not exist: {root}")
    records: list[SegmentSSLRecord] = []
    for path in sorted(root.rglob("*.npz"), key=lambda item: str(item)):
        with np.load(path, allow_pickle=False) as payload:
            features: dict[str, np.ndarray] = {}
            for branch in branches:
                payload_key = _payload_key_for_branch(branch)
                if payload_key in payload:
                    features[branch] = np.asarray(payload[payload_key], dtype=np.float32)
            if not features:
                continue
            records.append(
                SegmentSSLRecord(
                    group=_np_scalar(payload, "group"),
                    subject_id=normalize_subject_id(_np_scalar(payload, "subject_id")),
                    subject_key=_np_scalar(payload, "subject_key"),
                    stage=_np_scalar(payload, "stage"),
                    state=_np_scalar(payload, "state"),
                    segment_index=int(payload["segment_index"].item()),
                    features=features,
                    source_path=path,
                )
            )
    return sorted(
        records,
        key=lambda record: (
            record.group,
            record.stage,
            record.subject_key,
            record.state,
            record.segment_index,
            str(record.source_path),
        ),
    )


def segment_records_for_scope(
    records: Iterable[SegmentSSLRecord],
    *,
    data_scope: str,
    strict_loso_test_subject_id: str | None = None,
    supervised_subject_ids: Iterable[str] = (),
    baseline_stage: str = "baseline",
) -> list[SegmentSSLRecord]:
    """Select a fold-local SSL segment pool without admitting the LOSO test patient."""

    scope = normalize_ssl_data_scope(data_scope)
    supervised = {normalize_subject_id(subject_id) for subject_id in supervised_subject_ids}
    excluded_subject = (
        normalize_subject_id(strict_loso_test_subject_id)
        if strict_loso_test_subject_id is not None
        else None
    )
    baseline_tokens = {baseline_stage, baseline_stage.strip().lower(), *BASELINE_STAGE_ALIASES}

    selected: list[SegmentSSLRecord] = []
    for record in records:
        if excluded_subject is not None and record.group == "patient" and record.subject_id == excluded_subject:
            continue
        if _record_in_scope(record, scope, supervised, baseline_tokens):
            selected.append(record)
    return sorted(
        selected,
        key=lambda record: (
            record.group,
            record.stage,
            record.subject_key,
            record.state,
            record.segment_index,
        ),
    )


def merge_segment_modalities(
    primary: Iterable[SegmentSSLRecord],
    secondary: Iterable[SegmentSSLRecord],
    *,
    primary_branch: str = "psd",
    secondary_branch: str | Iterable[str] = "wpli",
) -> list[SegmentSSLRecord]:
    """Merge modality-specific segment records by subject/stage/state/segment index."""

    secondary_branches = (secondary_branch,) if isinstance(secondary_branch, str) else tuple(secondary_branch)
    if not secondary_branches:
        raise ValueError("At least one secondary branch is required.")
    merged: dict[tuple[str, str, str, int], SegmentSSLRecord] = {
        _record_key(record): record
        for record in primary
    }
    matched_keys: set[tuple[str, str, str, int]] = set()
    for record in secondary:
        key = _record_key(record)
        if key not in merged:
            continue
        existing = merged[key]
        features = dict(existing.features)
        if primary_branch not in features:
            raise ValueError(f"Primary record missing branch {primary_branch!r}.")
        for branch in secondary_branches:
            if branch not in record.features:
                raise ValueError(f"Secondary record missing branch {branch!r}.")
            features[branch] = record.features[branch]
        merged[key] = SegmentSSLRecord(
            group=existing.group,
            subject_id=existing.subject_id,
            subject_key=existing.subject_key,
            stage=existing.stage,
            state=existing.state,
            segment_index=existing.segment_index,
            features=features,
            source_path=existing.source_path,
        )
        matched_keys.add(key)
    return sorted(
        (record for key, record in merged.items() if key in matched_keys),
        key=lambda record: (record.subject_key, record.stage, record.state, record.segment_index),
    )


def _record_in_scope(
    record: SegmentSSLRecord,
    scope: str,
    supervised_subject_ids: set[str],
    baseline_tokens: set[str],
) -> bool:
    is_baseline = record.stage in baseline_tokens or record.stage.strip().lower() in baseline_tokens
    if scope == "supervised-baseline":
        return record.group == "patient" and is_baseline and record.subject_id in supervised_subject_ids
    if scope == "all-patient-baseline":
        return record.group == "patient" and is_baseline
    if scope == "all-patient":
        return record.group == "patient"
    if scope == "all-patient-health":
        return record.group in {"patient", "health"}
    if scope == "health-only":
        return record.group == "health"
    raise ValueError(f"Unsupported segment SSL data scope: {scope}")


def _record_key(record: SegmentSSLRecord) -> tuple[str, str, str, int]:
    return (record.subject_key, record.stage, record.state, record.segment_index)


def _np_scalar(payload: Mapping[str, np.ndarray], key: str) -> str:
    if key not in payload:
        raise KeyError(f"Segment cache is missing required key {key!r}.")
    return str(payload[key].item())


def _payload_key_for_branch(branch: str) -> str:
    if branch == "icoh":
        return "imaginary_coherence"
    return branch
