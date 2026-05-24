from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table, read_clinical_metadata
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.train_feature_ssl import (
    build_feature_ssl_pair_records_from_feature_records,
    compute_feature_records_for_eeg_records,
    feature_ssl_pairs_for_scope,
    widest_feature_ssl_scope,
    write_feature_ssl_transfer_outputs,
)
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES, select_ssl_records
from eeg_recovery.training.train_structured_ssl import (
    StructuredSSLConfig,
    load_time_frequency_images_for_records,
    run_structured_ssl_pretraining,
)
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TFR-image PSD SSL plus graph-structured FC SSL transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--psd-ssl-epochs", type=int, default=10)
    parser.add_argument("--fc-ssl-epochs", type=int, default=10)
    parser.add_argument("--ssl-batch-size", type=int, default=8)
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--tfr-noise-std", type=float, default=0.02)
    parser.add_argument("--tfr-channel-mask-prob", type=float, default=0.05)
    parser.add_argument("--tfr-frequency-mask-prob", type=float, default=0.05)
    parser.add_argument("--fc-noise-std", type=float, default=0.02)
    parser.add_argument("--fc-node-dropout-prob", type=float, default=0.05)
    parser.add_argument("--fc-edge-dropout-prob", type=float, default=0.05)
    parser.add_argument("--max-tfr-windows-per-record", type=int, default=2)
    parser.add_argument("--tfr-nperseg", type=int, default=256)
    parser.add_argument("--tfr-noverlap", type=int, default=128)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--summary-name",
        default="structured_ssl_psd_fc_wpli_transfer_summary.csv",
    )
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(
        path_config,
        labels,
        feature_kind="psd-fc-wpli",
    )
    eeg_records = build_eeg_file_index(
        path_config.patient_eeg_root,
        path_config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )
    eeg_records = select_ssl_records(
        eeg_records,
        data_scope=widest_feature_ssl_scope([args.data_scope]),
    )
    feature_records = compute_feature_records_for_eeg_records(
        path_config,
        eeg_records,
        feature_kind="psd-fc-wpli",
    )
    all_ssl_pairs = build_feature_ssl_pair_records_from_feature_records(eeg_records, feature_records)
    clinical = read_clinical_metadata(path_config.patient_info_clinical_xlsx)
    affected_hand_by_subject = {
        normalize_subject_id(row.subject_id): str(row.affected_hand)
        for row in clinical.itertuples(index=False)
    }

    tfr_images_by_record = {}
    for record in eeg_records:
        key = (record.subject_key, record.stage, record.state)
        tfr_images_by_record[key] = load_time_frequency_images_for_records(
            [record],
            max_windows_per_record=args.max_tfr_windows_per_record,
            nperseg=args.tfr_nperseg,
            noverlap=args.tfr_noverlap,
            affected_hand_by_subject=affected_hand_by_subject,
        )

    folds = make_loso_folds([record.subject_id for record in supervised_records])
    pretrained_state_by_test_subject = {}
    ssl_history_frames: list[pd.DataFrame] = []
    for fold in folds:
        ssl_pairs = feature_ssl_pairs_for_scope(
            all_ssl_pairs,
            data_scope=args.data_scope,
            strict_loso_test_subject_id=fold.test_subject_id,
        )
        selected_keys = {
            (pair.subject_key, pair.stage, state)
            for pair in ssl_pairs
            for state in ("EO", "EC")
        }
        tfr_images = [
            image
            for key in selected_keys
            for image in tfr_images_by_record.get(key, [])
        ]
        ssl_config = StructuredSSLConfig(
            psd_epochs=args.psd_ssl_epochs,
            fc_epochs=args.fc_ssl_epochs,
            batch_size=args.ssl_batch_size,
            embedding_dim=args.embedding_dim,
            projection_dim=args.projection_dim,
            lr=args.ssl_lr,
            temperature=args.temperature,
            tfr_noise_std=args.tfr_noise_std,
            tfr_channel_mask_prob=args.tfr_channel_mask_prob,
            tfr_frequency_mask_prob=args.tfr_frequency_mask_prob,
            fc_noise_std=args.fc_noise_std,
            fc_node_dropout_prob=args.fc_node_dropout_prob,
            fc_edge_dropout_prob=args.fc_edge_dropout_prob,
            device=args.device,
            seed=args.seed + fold.fold_index,
        )
        pretrained_state, ssl_history = run_structured_ssl_pretraining(tfr_images, ssl_pairs, ssl_config)
        ssl_history.insert(0, "fold_index", fold.fold_index)
        ssl_history.insert(1, "test_subject_id", fold.test_subject_id)
        ssl_history_frames.append(ssl_history)
        pretrained_state_by_test_subject[fold.test_subject_id] = pretrained_state
        print(
            f"[structured-ssl] fold={fold.fold_index} test={fold.test_subject_id} "
            f"pairs={len(ssl_pairs)} tfr_images={len(tfr_images)}"
        )

    supervised_config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        device=args.device,
        epochs=args.supervised_epochs,
        patience=args.patience,
        lr=args.supervised_lr,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        seed=args.seed,
        pretrained_transfer_mode="finetune",
    )
    predictions, metrics, supervised_loss = run_loso_supervised_with_history(
        supervised_records,
        supervised_config,
        pretrained_state_by_test_subject=pretrained_state_by_test_subject,
    )
    run_name = (
        f"structured_ssl_{args.data_scope}_psd_tfr_aligned_wpli_graph_gated_cnn"
        f"_seed{args.seed}_psd{args.psd_ssl_epochs}_fc{args.fc_ssl_epochs}"
        f"_bs{args.ssl_batch_size}_sup{args.supervised_epochs}"
    )
    ssl_history_all = pd.concat(ssl_history_frames, ignore_index=True)
    for frame in (predictions, metrics, supervised_loss, ssl_history_all):
        frame["ssl_data_scope"] = args.data_scope
        frame["run_name"] = run_name
        frame["structured_ssl"] = True
        frame["psd_ssl_method"] = "tfr_contrastive"
        frame["fc_ssl_method"] = "graph_contrastive"
        frame["tfr_hemisphere_aligned"] = True
        frame["psd_ssl_epochs"] = args.psd_ssl_epochs
        frame["fc_ssl_epochs"] = args.fc_ssl_epochs
    paths = write_feature_ssl_transfer_outputs(
        output_root=path_config.output_root,
        run_name=run_name,
        predictions=predictions,
        metrics=metrics,
        ssl_history=ssl_history_all,
        supervised_loss_history=supervised_loss,
    )
    summary_path = Path(path_config.output_root) / "results" / "metrics" / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_path, index=False)
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote metrics: {paths['metrics']}")
    print(f"Wrote SSL history: {paths['ssl_history']}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
