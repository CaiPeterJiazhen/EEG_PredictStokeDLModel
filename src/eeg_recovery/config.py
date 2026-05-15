from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eeg_recovery.utils.paths import resolve_path


REQUIRED_PATH_KEYS = (
    "patient_info_integrity_xlsx",
    "patient_info_clinical_xlsx",
    "patient_eeg_root",
    "health_eeg_root",
    "standard_1005_ced",
    "output_root",
)

EXISTING_INPUT_PATH_KEYS = (
    "patient_info_integrity_xlsx",
    "patient_info_clinical_xlsx",
    "patient_eeg_root",
    "health_eeg_root",
    "standard_1005_ced",
)


class ConfigError(ValueError):
    """Raised when a project configuration file is invalid."""


@dataclass(frozen=True)
class PathConfig:
    patient_info_integrity_xlsx: Path
    patient_info_clinical_xlsx: Path
    patient_eeg_root: Path
    health_eeg_root: Path
    standard_1005_ced: Path
    output_root: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            key: getattr(self, key)
            for key in REQUIRED_PATH_KEYS
        }


def load_path_config(config_path: str | Path, *, validate_exists: bool = True) -> PathConfig:
    """Load and validate the external data path configuration."""

    config_file = resolve_path(config_path)
    if not config_file.exists():
        raise ConfigError(f"Path config does not exist: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ConfigError(f"Path config must contain a YAML mapping: {config_file}")

    missing_keys = [key for key in REQUIRED_PATH_KEYS if key not in payload]
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ConfigError(f"Path config is missing required key(s): {joined}")

    base_dir = config_file.parent
    paths = {
        key: _coerce_path(payload[key], key, base_dir)
        for key in REQUIRED_PATH_KEYS
    }

    if validate_exists:
        missing_paths = [
            f"{key}={paths[key]}"
            for key in EXISTING_INPUT_PATH_KEYS
            if not paths[key].exists()
        ]
        if missing_paths:
            joined = "; ".join(missing_paths)
            raise ConfigError(f"Required external path(s) do not exist: {joined}")

    return PathConfig(**paths)


def _coerce_path(value: Any, key: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Path config key {key!r} must be a non-empty string")
    return resolve_path(value, base_dir=base_dir)
