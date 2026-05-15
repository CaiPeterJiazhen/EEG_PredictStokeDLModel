from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.connectivity import (
    compute_connectivity_for_eeg_record,
    write_connectivity_feature,
)
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute supervised baseline EO/EC functional connectivity features."
    )
    parser.add_argument(
        "--config",
        default="configs/paths.example.yaml",
        help="Path YAML containing external EEG/workbook paths and output_root.",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Optional subject filter such as sub01 or sub001.",
    )
    args = parser.parse_args()

    config = load_path_config(Path(args.config))
    labels = load_supervised_label_table(config)
    labels_by_subject = labels.set_index("subject_id")
    supervised_ids = labels["subject_id"].tolist()
    subject_filter = (
        normalize_subject_id(args.subject_id)
        if args.subject_id is not None
        else None
    )

    records = build_eeg_file_index(
        config.patient_eeg_root,
        config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
    )
    selected_records = [
        record
        for record in records
        if record.group == "patient"
        and record.stage == "基线"
        and record.is_supervised_subject
        and (subject_filter is None or record.subject_id == subject_filter)
    ]

    if subject_filter is not None and not selected_records:
        raise SystemExit(f"No supervised baseline EO/EC records found for {subject_filter}")

    written: list[Path] = []
    for record in selected_records:
        affected_hand = str(labels_by_subject.loc[record.subject_id, "affected_hand"])
        feature = compute_connectivity_for_eeg_record(record, affected_hand)
        written.append(write_connectivity_feature(feature, config.output_root))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
