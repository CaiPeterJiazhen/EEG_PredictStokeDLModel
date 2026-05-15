from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eeg_recovery.config import ConfigError, PathConfig, load_path_config


def write_yaml(path: Path, payload: dict[str, str]) -> Path:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def valid_payload(tmp_path: Path) -> dict[str, str]:
    patient_info_integrity_xlsx = tmp_path / "integrity.xlsx"
    patient_info_clinical_xlsx = tmp_path / "clinical.xlsx"
    patient_eeg_root = tmp_path / "patient_eeg"
    health_eeg_root = tmp_path / "health_eeg"
    standard_1005_ced = tmp_path / "standard_1005.ced"

    patient_info_integrity_xlsx.write_text("placeholder", encoding="utf-8")
    patient_info_clinical_xlsx.write_text("placeholder", encoding="utf-8")
    patient_eeg_root.mkdir()
    health_eeg_root.mkdir()
    standard_1005_ced.write_text("placeholder", encoding="utf-8")

    return {
        "patient_info_integrity_xlsx": str(patient_info_integrity_xlsx),
        "patient_info_clinical_xlsx": str(patient_info_clinical_xlsx),
        "patient_eeg_root": str(patient_eeg_root),
        "health_eeg_root": str(health_eeg_root),
        "standard_1005_ced": str(standard_1005_ced),
        "output_root": str(tmp_path / "outputs"),
    }


def test_load_path_config_returns_dataclass_and_does_not_create_outputs(tmp_path: Path) -> None:
    config_path = write_yaml(tmp_path / "paths.yaml", valid_payload(tmp_path))

    config = load_path_config(config_path)

    assert isinstance(config, PathConfig)
    assert config.patient_info_integrity_xlsx.exists()
    assert config.patient_info_clinical_xlsx.exists()
    assert config.patient_eeg_root.is_dir()
    assert config.health_eeg_root.is_dir()
    assert config.standard_1005_ced.exists()
    assert config.output_root == tmp_path / "outputs"
    assert not config.output_root.exists()


def test_load_path_config_reports_missing_required_key(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload.pop("standard_1005_ced")
    config_path = write_yaml(tmp_path / "paths.yaml", payload)

    with pytest.raises(ConfigError, match="standard_1005_ced"):
        load_path_config(config_path)


def test_load_path_config_reports_missing_external_paths_without_creating_outputs(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    missing_file = tmp_path / "missing_integrity.xlsx"
    output_root = tmp_path / "outputs"
    payload["patient_info_integrity_xlsx"] = str(missing_file)
    payload["output_root"] = str(output_root)
    config_path = write_yaml(tmp_path / "paths.yaml", payload)

    with pytest.raises(ConfigError) as error:
        load_path_config(config_path)

    assert "patient_info_integrity_xlsx" in str(error.value)
    assert str(missing_file) in str(error.value)
    assert not output_root.exists()


def test_load_path_config_resolves_relative_values_from_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    source_dir = config_dir / "source"
    source_dir.mkdir(parents=True)
    payload = {
        "patient_info_integrity_xlsx": "source/integrity.xlsx",
        "patient_info_clinical_xlsx": "source/clinical.xlsx",
        "patient_eeg_root": "source/patient_eeg",
        "health_eeg_root": "source/health_eeg",
        "standard_1005_ced": "source/standard_1005.ced",
        "output_root": "outputs",
    }

    (source_dir / "integrity.xlsx").write_text("placeholder", encoding="utf-8")
    (source_dir / "clinical.xlsx").write_text("placeholder", encoding="utf-8")
    (source_dir / "patient_eeg").mkdir()
    (source_dir / "health_eeg").mkdir()
    (source_dir / "standard_1005.ced").write_text("placeholder", encoding="utf-8")
    config_path = write_yaml(config_dir / "paths.yaml", payload)

    config = load_path_config(config_path)

    assert config.patient_info_integrity_xlsx == source_dir / "integrity.xlsx"
    assert config.patient_info_clinical_xlsx == source_dir / "clinical.xlsx"
    assert config.patient_eeg_root == source_dir / "patient_eeg"
    assert config.health_eeg_root == source_dir / "health_eeg"
    assert config.standard_1005_ced == source_dir / "standard_1005.ced"
    assert config.output_root == config_dir / "outputs"
