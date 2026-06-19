from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.connectivity import (
    ConnectivityFeature,
    compute_connectivity_for_eeg_record,
)
from eeg_recovery.features.psd import PSDFeature, compute_psd_for_eeg_record
from eeg_recovery.io.index import EEGFileRecord, build_eeg_file_index
from eeg_recovery.metadata.labels import (
    load_supervised_label_table,
    read_clinical_metadata,
)
from eeg_recovery.metadata.subjects import normalize_subject_id


DEFAULT_FEATURE_ROOT = Path("data") / "features_all_stages"
FEATURE_VERSION = "all_stage_patient_health_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute patient-level PSD and WPLI features for all available patient "
            "stages and healthy controls without overwriting supervised baseline caches."
        )
    )
    parser.add_argument("--config", default="configs/paths.example.yaml")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_FEATURE_ROOT),
        help="Output directory relative to project root unless absolute.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["patient", "health"],
        choices=["patient", "health"],
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Optional patient stage filter, e.g. 基线 即时 阶段 最终. Health records use stage=health.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["psd", "wpli"],
        choices=["psd", "wpli"],
        help="Features to compute. wpli is written under the fc directory with key='wpli'.",
    )
    parser.add_argument("--subject-id", default=None, help="Optional patient subject filter.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned coverage.")
    args = parser.parse_args()

    config = load_path_config(Path(args.config))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    supervised = load_supervised_label_table(config)
    supervised_ids = supervised["subject_id"].tolist()
    clinical = read_clinical_metadata(config.patient_info_clinical_xlsx)
    affected_by_subject = {
        normalize_subject_id(row.subject_id): str(row.affected_hand)
        for row in clinical.itertuples(index=False)
    }

    records = build_eeg_file_index(
        config.patient_eeg_root,
        config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=False,
    )
    selected = _select_records(
        records,
        groups=set(args.groups),
        stages=set(args.stages) if args.stages else None,
        subject_id=args.subject_id,
    )
    if not selected:
        raise SystemExit("No EEG records matched the requested filters.")

    coverage = _coverage_rows(selected)
    print(_format_coverage_summary(coverage))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    total_jobs = len(selected) * len(args.modalities)
    completed_jobs = 0

    for record in selected:
        affected_hand = _affected_hand_for_record(record, affected_by_subject)
        if "psd" in args.modalities:
            completed_jobs += 1
            psd_path = _psd_output_path(output_dir, record)
            status = "skipped_existing"
            if args.overwrite or not psd_path.exists():
                feature = compute_psd_for_eeg_record(record, affected_hand)
                _write_psd_feature(feature, record, affected_hand, psd_path)
                status = "written"
            manifest_rows.append(
                _manifest_row(record, "psd", psd_path, affected_hand, status)
            )
            print(f"[{completed_jobs}/{total_jobs}] {status} PSD {record.group} {record.stage} {record.subject_id} {record.state}", flush=True)

        if "wpli" in args.modalities:
            completed_jobs += 1
            wpli_path = _wpli_output_path(output_dir, record)
            status = "skipped_existing"
            if args.overwrite or not wpli_path.exists():
                feature = compute_connectivity_for_eeg_record(record, affected_hand)
                _write_wpli_feature(feature, record, affected_hand, wpli_path)
                status = "written"
            manifest_rows.append(
                _manifest_row(record, "wpli", wpli_path, affected_hand, status)
            )
            print(f"[{completed_jobs}/{total_jobs}] {status} WPLI {record.group} {record.stage} {record.subject_id} {record.state}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "feature_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    coverage_path = output_dir / "feature_coverage_summary.csv"
    pd.DataFrame(coverage).to_csv(coverage_path, index=False, encoding="utf-8-sig")

    print(f"Manifest: {manifest_path}")
    print(f"Coverage: {coverage_path}")


def _select_records(
    records: Iterable[EEGFileRecord],
    *,
    groups: set[str],
    stages: set[str] | None,
    subject_id: str | None,
) -> list[EEGFileRecord]:
    subject_filter = normalize_subject_id(subject_id) if subject_id else None
    selected: list[EEGFileRecord] = []
    for record in records:
        if record.group not in groups:
            continue
        if stages is not None and record.group == "patient" and record.stage not in stages:
            continue
        if stages is not None and record.group == "health" and "health" not in stages:
            continue
        if subject_filter is not None:
            if record.group != "patient" or record.subject_id != subject_filter:
                continue
        selected.append(record)
    return selected


def _affected_hand_for_record(
    record: EEGFileRecord,
    affected_by_subject: dict[str, str],
) -> str:
    if record.group == "health":
        return "右"
    affected = affected_by_subject.get(record.subject_id)
    if affected not in {"右", "左"}:
        raise ValueError(f"Missing valid affected hand for patient {record.subject_id}.")
    return affected


def _coverage_rows(records: Iterable[EEGFileRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    subjects: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        key = (record.group, record.stage)
        grouped[key][record.state] += 1
        subjects[key].add(record.subject_id)

    rows: list[dict[str, object]] = []
    for (group, stage), counts in sorted(grouped.items()):
        rows.append(
            {
                "group": group,
                "stage": stage,
                "subjects": len(subjects[(group, stage)]),
                "EO_records": counts["EO"],
                "EC_records": counts["EC"],
                "complete_eo_ec_pairs": min(counts["EO"], counts["EC"]),
                "total_records": counts["EO"] + counts["EC"],
            }
        )
    return rows


def _format_coverage_summary(rows: list[dict[str, object]]) -> str:
    lines = ["Planned coverage:"]
    for row in rows:
        lines.append(
            "  "
            f"{row['group']} {row['stage']}: "
            f"subjects={row['subjects']}, EO={row['EO_records']}, "
            f"EC={row['EC_records']}, pairs={row['complete_eo_ec_pairs']}"
        )
    return "\n".join(lines)


def _safe_path_part(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_").strip()


def _feature_dir(output_dir: Path, modality: str, record: EEGFileRecord) -> Path:
    return output_dir / modality / _safe_path_part(record.group) / _safe_path_part(record.stage)


def _psd_output_path(output_dir: Path, record: EEGFileRecord) -> Path:
    return _feature_dir(output_dir, "psd", record) / f"{_safe_path_part(record.subject_id)}_{record.state}_psd.npz"


def _wpli_output_path(output_dir: Path, record: EEGFileRecord) -> Path:
    return _feature_dir(output_dir, "fc", record) / f"{_safe_path_part(record.subject_id)}_{record.state}_fc.npz"


def _write_psd_feature(
    feature: PSDFeature,
    record: EEGFileRecord,
    affected_hand: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        psd=feature.psd.astype(np.float32),
        frequency_bins=feature.frequency_bins,
        group=np.array(record.group),
        subject_id=np.array(record.subject_id),
        subject_key=np.array(record.subject_key),
        stage=np.array(record.stage),
        state=np.array(record.state),
        is_supervised_subject=np.array(record.is_supervised_subject),
        channel_names_after_alignment=np.array(feature.channel_names_after_alignment),
        sampling_rate=np.array(feature.sampling_rate),
        source_set_path=np.array(feature.source_set_path.as_posix()),
        source_fdt_path=np.array(record.fdt_path.as_posix()),
        affected_hand=np.array(affected_hand),
        hemisphere_aligned=np.array(True),
        feature_version=np.array(FEATURE_VERSION),
        **{
            key: np.array(value)
            for key, value in feature.metadata.items()
            if key not in {"affected_hand", "hemisphere_aligned"}
        },
    )


def _write_wpli_feature(
    feature: ConnectivityFeature,
    record: EEGFileRecord,
    affected_hand: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wpli=feature.wpli.astype(np.float32),
        edge_list=np.array(feature.edge_list),
        band_names=np.array(feature.band_names),
        band_ranges_hz=np.array(feature.band_ranges_hz, dtype=float),
        group=np.array(record.group),
        subject_id=np.array(record.subject_id),
        subject_key=np.array(record.subject_key),
        stage=np.array(record.stage),
        state=np.array(record.state),
        is_supervised_subject=np.array(record.is_supervised_subject),
        channel_names_after_alignment=np.array(feature.channel_names_after_alignment),
        sampling_rate=np.array(feature.sampling_rate),
        source_set_path=np.array(feature.source_set_path.as_posix()),
        source_fdt_path=np.array(record.fdt_path.as_posix()),
        affected_hand=np.array(affected_hand),
        hemisphere_aligned=np.array(True),
        feature_version=np.array(FEATURE_VERSION),
        **{
            key: np.array(value)
            for key, value in feature.metadata.items()
            if key not in {"affected_hand", "hemisphere_aligned"}
        },
    )


def _manifest_row(
    record: EEGFileRecord,
    modality: str,
    output_path: Path,
    affected_hand: str,
    status: str,
) -> dict[str, object]:
    return {
        "group": record.group,
        "stage": record.stage,
        "subject_id": record.subject_id,
        "subject_key": record.subject_key,
        "state": record.state,
        "modality": modality,
        "affected_hand": affected_hand,
        "hemisphere_aligned": True,
        "is_supervised_subject": record.is_supervised_subject,
        "source_set_path": record.set_path.as_posix(),
        "source_fdt_path": record.fdt_path.as_posix(),
        "feature_path": output_path.as_posix(),
        "status": status,
    }


if __name__ == "__main__":
    main()
