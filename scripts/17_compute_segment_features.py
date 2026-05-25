from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.segment_features import (
    SegmentFeatureConfig,
    compute_segment_fc_for_eeg_record,
    compute_segment_psd_for_eeg_record,
)
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.metadata.labels import read_clinical_metadata
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES, select_ssl_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute segment-level PSD and optional FC/wPLI feature caches for negative-free SSL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--psd-window-seconds", type=float, default=4.0)
    parser.add_argument("--psd-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--fc-window-seconds", type=float, default=8.0)
    parser.add_argument("--fc-overlap-fraction", type=float, default=0.5)
    parser.add_argument(
        "--feature-kind",
        choices=("psd", "fc-wpli", "psd-fc-wpli"),
        default="psd",
        help="Segment cache modality to generate. fc-wpli writes data/features/segment_level/fc.",
    )
    parser.add_argument(
        "--compute-fc",
        action="store_true",
        help="Backward-compatible alias for --feature-kind psd-fc-wpli.",
    )
    parser.add_argument("--force", action="store_true", help="Rewrite segment caches even when source mtimes match.")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    label_table = load_supervised_label_table(path_config)
    supervised_ids = label_table["subject_id"].tolist()
    clinical = read_clinical_metadata(path_config.patient_info_clinical_xlsx)
    affected_by_subject = {
        normalize_subject_id(row.subject_id): str(row.affected_hand)
        for row in clinical.itertuples(index=False)
    }
    eeg_records = build_eeg_file_index(
        path_config.patient_eeg_root,
        path_config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )
    selected_records = select_ssl_records(eeg_records, data_scope=args.data_scope)
    segment_config = SegmentFeatureConfig(
        psd_window_seconds=args.psd_window_seconds,
        psd_overlap_fraction=args.psd_overlap_fraction,
        fc_window_seconds=args.fc_window_seconds,
        fc_overlap_fraction=args.fc_overlap_fraction,
        compute_fc=args.compute_fc,
    )

    feature_kind = "psd-fc-wpli" if args.compute_fc else args.feature_kind
    compute_psd = feature_kind in {"psd", "psd-fc-wpli"}
    compute_fc = feature_kind in {"fc-wpli", "psd-fc-wpli"}

    total_psd_segments = 0
    total_fc_segments = 0
    for record in selected_records:
        affected_hand = "right" if record.group == "health" else affected_by_subject.get(record.subject_id)
        if affected_hand is None:
            raise SystemExit(f"Missing affected hand metadata for patient {record.subject_id}.")
        if compute_psd:
            psd_features = compute_segment_psd_for_eeg_record(
                record,
                affected_hand,
                output_root=path_config.output_root,
                config=segment_config,
                force=args.force,
            )
            total_psd_segments += len(psd_features)
            print(
                f"[segment-psd] {record.subject_key} stage={record.stage} state={record.state} "
                f"segments={len(psd_features)}"
            )
        if compute_fc:
            fc_features = compute_segment_fc_for_eeg_record(
                record,
                affected_hand,
                output_root=path_config.output_root,
                config=segment_config,
                force=args.force,
            )
            total_fc_segments += len(fc_features)
            print(
                f"[segment-fc] {record.subject_key} stage={record.stage} state={record.state} "
                f"segments={len(fc_features)}"
            )

    if compute_psd:
        print(f"Wrote or validated {total_psd_segments} PSD segment feature(s).")
    if compute_fc:
        print(f"Wrote or validated {total_fc_segments} FC/wPLI segment feature(s).")


if __name__ == "__main__":
    main()
