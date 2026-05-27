from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from eeg_recovery.config import PathConfig
from eeg_recovery.metadata.subjects import normalize_subject_id


SUPERVISED_ID_COLUMN = "患者ID"

CLINICAL_COLUMN_MAP = {
    "编号": "subject_id",
    "年龄": "age",
    "性别": "sex",
    "病程": "duration",
    "患病侧（手）": "affected_hand",
    "患病侧": "affected_hand",
    "治疗前FMA": "FMA_pre",
    "治疗后FMA": "FMA_post",
    "治疗前MBI": "MBI_pre",
    "治疗后MBI": "MBI_post",
}

REQUIRED_CLINICAL_COLUMNS = tuple(dict.fromkeys(CLINICAL_COLUMN_MAP.values()))

NORMALIZED_CLINICAL_COLUMN_MAP = {
    "".join(source.split()): target
    for source, target in CLINICAL_COLUMN_MAP.items()
}

OUTPUT_COLUMNS = [
    "subject_id",
    "age",
    "sex",
    "duration",
    "affected_hand",
    "FMA_pre",
    "FMA_post",
    "MBI_pre",
    "MBI_post",
    "Delta_FMA_pred",
    "Delta_FMA_obs",
    "Residual",
    "label",
]

MODEL_INPUT_COLUMNS = (
    "age",
    "sex",
    "duration",
    "affected_hand",
    "FMA_pre",
    "MBI_pre",
)


class MetadataError(ValueError):
    """Raised when source metadata cannot be parsed into the expected schema."""


def load_supervised_label_table(config: PathConfig) -> pd.DataFrame:
    """Load the supervised 19-subject cohort and derive recovery labels."""

    supervised_subjects = read_supervised_subject_ids(config.patient_info_integrity_xlsx)
    clinical = read_clinical_metadata(config.patient_info_clinical_xlsx)
    supervised = _select_supervised_clinical_rows(clinical, supervised_subjects)

    table = _add_recovery_labels(supervised)
    return table[OUTPUT_COLUMNS].reset_index(drop=True)


def model_input_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return baseline-only clinical metadata columns allowed as model inputs."""

    columns = ["subject_id", *MODEL_INPUT_COLUMNS]
    _validate_columns(metadata.columns, columns, source="metadata dataframe")
    return metadata.loc[:, columns].copy()


def read_supervised_subject_ids(path: str | Path) -> list[str]:
    integrity = pd.read_excel(path)
    integrity = integrity.rename(columns=_canonical_integrity_column_name)
    if SUPERVISED_ID_COLUMN not in integrity.columns:
        return _derive_supervised_subject_ids_from_clinical_layout(path)

    subject_ids = [
        normalize_subject_id(value)
        for value in integrity[SUPERVISED_ID_COLUMN].dropna()
    ]
    if not subject_ids:
        raise MetadataError(f"Integrity workbook contains no supervised subject IDs: {path}")
    if len(subject_ids) != len(set(subject_ids)):
        raise MetadataError(f"Integrity workbook contains duplicate supervised subject IDs: {path}")
    return subject_ids


def read_clinical_metadata(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_index = _find_header_row(raw, required_columns=REQUIRED_CLINICAL_COLUMNS)
    clinical = pd.read_excel(path, header=header_index)
    clinical = clinical.rename(columns=_canonical_clinical_column_name)
    _validate_columns(clinical.columns, REQUIRED_CLINICAL_COLUMNS, source=path)

    clinical = clinical[list(REQUIRED_CLINICAL_COLUMNS)]
    clinical = clinical.dropna(subset=["subject_id"]).copy()
    is_patient_subject = clinical["subject_id"].astype(str).str.contains(
        r"sub\s*\d+",
        case=False,
        regex=True,
    )
    clinical = clinical.loc[is_patient_subject].copy()
    clinical["subject_id"] = clinical["subject_id"].map(normalize_subject_id)
    clinical["affected_hand"] = clinical["affected_hand"].map(_normalize_affected_hand_label)

    for column in ["age", "duration", "FMA_pre", "FMA_post", "MBI_pre", "MBI_post"]:
        clinical[column] = pd.to_numeric(
            clinical[column].map(_extract_numeric_token),
            errors="coerce",
        )

    is_patient_metadata = (
        clinical["affected_hand"].notna()
        | clinical["FMA_pre"].notna()
        | clinical["MBI_pre"].notna()
    )
    clinical = clinical.loc[is_patient_metadata].copy()

    return clinical


def _derive_supervised_subject_ids_from_clinical_layout(path: str | Path) -> list[str]:
    try:
        clinical = read_clinical_metadata(path)
    except MetadataError as exc:
        raise MetadataError(
            f"Integrity workbook is missing required column {SUPERVISED_ID_COLUMN!r} "
            f"and cannot be interpreted as a clinical workbook: {path}"
        ) from exc

    labeled = clinical.dropna(subset=["FMA_pre", "FMA_post"])
    subject_ids = labeled["subject_id"].tolist()
    if not subject_ids:
        raise MetadataError(
            f"Clinical-layout workbook contains no rows with numeric FMA_pre and FMA_post: {path}"
        )
    if len(subject_ids) != len(set(subject_ids)):
        raise MetadataError(f"Clinical-layout workbook contains duplicate supervised subject IDs: {path}")
    return subject_ids


def _find_header_row(raw: pd.DataFrame, *, required_columns: Any) -> int:
    required = set(required_columns)
    for index, row in raw.iterrows():
        observed = {
            _canonical_clinical_column_name(value)
            for value in row.dropna()
        }
        if required.issubset(observed):
            return int(index)

    joined = ", ".join(str(column) for column in required_columns)
    raise MetadataError(f"Clinical workbook is missing a full required header row: {joined}")


def _validate_columns(columns: Any, required: Any, *, source: str | Path) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        joined = ", ".join(repr(column) for column in missing)
        raise MetadataError(f"Workbook is missing required column(s) {joined}: {source}")


def _canonical_clinical_column_name(column: Any) -> Any:
    normalized = _normalize_header_cell(column)
    return NORMALIZED_CLINICAL_COLUMN_MAP.get(normalized, column)


def _canonical_integrity_column_name(column: Any) -> Any:
    if _normalize_header_cell(column) == _normalize_header_cell(SUPERVISED_ID_COLUMN):
        return SUPERVISED_ID_COLUMN
    return column


def _normalize_header_cell(value: Any) -> str:
    return "".join(str(value).split())


def _normalize_affected_hand_label(value: Any) -> Any:
    if pd.isna(value):
        return value
    raw = str(value).strip()
    normalized = raw.lower()
    if normalized in {"left", "l"} or ("左" in raw and "右" not in raw):
        return "左"
    if normalized in {"right", "r"} or ("右" in raw and "左" not in raw):
        return "右"
    return raw


def _extract_numeric_token(value: Any) -> Any:
    if pd.isna(value) or isinstance(value, (int, float)):
        return value
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return match.group(0) if match else value


def _select_supervised_clinical_rows(
    clinical: pd.DataFrame,
    supervised_subjects: list[str],
) -> pd.DataFrame:
    supervised_set = set(supervised_subjects)
    supervised_matches = clinical[clinical["subject_id"].isin(supervised_set)]
    duplicated_subjects = sorted(
        supervised_matches.loc[
            supervised_matches["subject_id"].duplicated(keep=False),
            "subject_id",
        ].unique()
    )
    if duplicated_subjects:
        joined = ", ".join(duplicated_subjects)
        raise MetadataError(f"Clinical workbook contains duplicate supervised subject_id row(s): {joined}")

    indexed = clinical.set_index("subject_id")
    missing = [subject_id for subject_id in supervised_subjects if subject_id not in indexed.index]
    if missing:
        joined = ", ".join(missing)
        raise MetadataError(f"Clinical workbook has no matching row for supervised subject(s): {joined}")

    return indexed.loc[supervised_subjects].reset_index()


def _add_recovery_labels(supervised: pd.DataFrame) -> pd.DataFrame:
    required = ["FMA_pre", "FMA_post"]
    missing_scores = [
        column
        for column in required
        if supervised[column].isna().any()
    ]
    if missing_scores:
        joined = ", ".join(missing_scores)
        raise MetadataError(f"Supervised labeled records contain missing required score(s): {joined}")

    table = supervised.copy()
    table["Delta_FMA_pred"] = 0.7 * (66 - table["FMA_pre"])
    table["Delta_FMA_obs"] = table["FMA_post"] - table["FMA_pre"]
    table["Residual"] = table["Delta_FMA_pred"] - table["Delta_FMA_obs"]

    for column in ["Delta_FMA_pred", "Delta_FMA_obs", "Residual"]:
        table[column] = table[column].round(10)

    median_residual = table["Residual"].median()
    table["label"] = (table["Residual"] <= median_residual).astype(int)
    return table
