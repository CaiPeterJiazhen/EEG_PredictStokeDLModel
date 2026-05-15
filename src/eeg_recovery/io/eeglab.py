from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

_METADATA_VARIABLES = (
    "EEG",
    "nbchan",
    "srate",
    "trials",
    "pnts",
    "data",
    "datfile",
    "chanlocs",
    "xmin",
    "xmax",
)


class EEGLABDataError(ValueError):
    """Raised when EEGLAB metadata or binary data is malformed."""


@dataclass(frozen=True)
class EEGLABMetadata:
    set_path: Path
    fdt_path: Path
    nbchan: int
    srate: float
    trials: int
    pnts: int
    xmin: float
    xmax: float
    ch_names: tuple[str, ...]
    data_file: str


def read_eeglab_set_metadata(set_path: str | Path) -> EEGLABMetadata:
    """Read EEGLAB ``.set`` metadata without loading companion ``.fdt`` samples."""

    resolved_set_path = Path(set_path)
    try:
        payload = loadmat(
            resolved_set_path,
            squeeze_me=True,
            struct_as_record=False,
            variable_names=_METADATA_VARIABLES,
        )
    except Exception as error:
        raise EEGLABDataError(
            f"Could not read EEGLAB .set metadata from {resolved_set_path}: {error}"
        ) from error

    data_file = _read_data_file(payload)
    fdt_path = _resolve_metadata_fdt_path(resolved_set_path, data_file)
    ch_names = _read_channel_labels(_get_mat_value(payload, "chanlocs"))

    return EEGLABMetadata(
        set_path=resolved_set_path,
        fdt_path=fdt_path,
        nbchan=_coerce_int(_get_mat_value(payload, "nbchan"), "nbchan"),
        srate=_coerce_float(_get_mat_value(payload, "srate"), "srate"),
        trials=_coerce_int(_get_mat_value(payload, "trials"), "trials"),
        pnts=_coerce_int(_get_mat_value(payload, "pnts"), "pnts"),
        xmin=_coerce_float(_get_mat_value(payload, "xmin"), "xmin"),
        xmax=_coerce_float(_get_mat_value(payload, "xmax"), "xmax"),
        ch_names=ch_names,
        data_file=data_file,
    )


def read_eeglab_fdt(
    metadata: EEGLABMetadata,
    subject_id: str | None = None,
    state: str | None = None,
) -> np.ndarray:
    """Read EEGLAB float32 ``.fdt`` samples as ``channels x samples``."""

    expected_samples = metadata.nbchan * metadata.pnts * metadata.trials
    try:
        raw = np.fromfile(metadata.fdt_path, dtype=np.float32)
    except OSError as error:
        subject = subject_id if subject_id is not None else "unknown subject"
        eeg_state = state if state is not None else "unknown state"
        raise EEGLABDataError(
            "Could not read EEGLAB .fdt "
            f"for subject={subject}, state={eeg_state}, path={metadata.fdt_path}: "
            f"{error}"
        ) from error
    observed_samples = int(raw.size)

    if observed_samples != expected_samples:
        subject = subject_id if subject_id is not None else "unknown subject"
        eeg_state = state if state is not None else "unknown state"
        raise EEGLABDataError(
            "EEGLAB .fdt length mismatch "
            f"for subject={subject}, state={eeg_state}, path={metadata.fdt_path}: "
            f"expected {expected_samples} float32 samples, observed {observed_samples}"
        )

    return raw.reshape(metadata.pnts * metadata.trials, metadata.nbchan).T


def resolve_eeglab_fdt_path(
    set_path: str | Path,
    data_file: str | Path | None = None,
) -> Path:
    """Resolve an EEGLAB binary data path relative to its ``.set`` file."""

    resolved_set_path = Path(set_path)
    if data_file is None or str(data_file) == "":
        return resolved_set_path.with_suffix(".fdt")

    fdt_path = Path(data_file)
    if fdt_path.is_absolute():
        return fdt_path
    return resolved_set_path.parent / fdt_path


def _resolve_metadata_fdt_path(set_path: Path, data_file: str) -> Path:
    declared_path = resolve_eeglab_fdt_path(set_path, data_file)
    if declared_path.exists():
        return declared_path

    companion_path = resolve_eeglab_fdt_path(set_path)
    if companion_path.exists():
        return companion_path

    return declared_path


def _get_mat_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]

    eeg = payload.get("EEG")
    if eeg is not None and hasattr(eeg, key):
        return getattr(eeg, key)

    raise EEGLABDataError(f"EEGLAB .set metadata is missing required field: {key}")


def _read_data_file(payload: dict[str, Any]) -> str:
    for field_name in ("data", "datfile"):
        try:
            value = _coerce_str(_get_mat_value(payload, field_name), field_name)
        except EEGLABDataError:
            continue
        if value:
            return value
    raise EEGLABDataError(
        "EEGLAB .set metadata is missing required .fdt filename in data/datfile"
    )


def _coerce_int(value: Any, field_name: str) -> int:
    scalar = _scalar_value(value, field_name)
    try:
        return int(scalar)
    except (TypeError, ValueError) as error:
        raise EEGLABDataError(
            f"EEGLAB .set field {field_name!r} must be an integer, got {scalar!r}"
        ) from error


def _coerce_float(value: Any, field_name: str) -> float:
    scalar = _scalar_value(value, field_name)
    try:
        return float(scalar)
    except (TypeError, ValueError) as error:
        raise EEGLABDataError(
            f"EEGLAB .set field {field_name!r} must be numeric, got {scalar!r}"
        ) from error


def _coerce_str(value: Any, field_name: str) -> str:
    scalar = _scalar_value(value, field_name)
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    raise EEGLABDataError(
        f"EEGLAB .set field {field_name!r} must be a string, got {scalar!r}"
    )


def _scalar_value(value: Any, field_name: str) -> Any:
    if isinstance(value, np.ndarray):
        squeezed = np.squeeze(value)
        if squeezed.shape == ():
            return squeezed.item()
        if squeezed.size == 1:
            return squeezed.reshape(-1)[0]
        if squeezed.dtype.kind in {"U", "S"}:
            return "".join(str(part) for part in squeezed.reshape(-1))
        raise EEGLABDataError(
            f"EEGLAB .set field {field_name!r} must be scalar, got shape {value.shape}"
        )
    return value


def _read_channel_labels(chanlocs: Any) -> tuple[str, ...]:
    if isinstance(chanlocs, np.ndarray):
        entries = chanlocs.ravel(order="F")
    else:
        entries = np.array([chanlocs], dtype=object)

    labels: list[str] = []
    for index, entry in enumerate(entries):
        if not hasattr(entry, "labels"):
            raise EEGLABDataError(
                f"EEGLAB chanlocs entry {index} is missing required labels attribute"
            )
        labels.append(_coerce_str(getattr(entry, "labels"), f"chanlocs[{index}].labels"))

    return tuple(labels)
