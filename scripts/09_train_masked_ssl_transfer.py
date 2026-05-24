from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.train_feature_ssl import (
    build_feature_ssl_pair_records_from_feature_records,
    compute_feature_records_for_eeg_records,
    feature_ssl_pairs_for_scope,
    widest_feature_ssl_scope,
    write_feature_ssl_transfer_outputs,
)
from eeg_recovery.training.train_masked_ssl import MaskedSSLConfig, run_masked_ssl_pretraining
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES, select_ssl_records
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run masked PSD/FC reconstruction SSL and transfer to the PSD+FC-wPLI gated CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--psd-ssl-epochs", type=int, default=20)
    parser.add_argument("--fc-ssl-epochs", type=int, default=20)
    parser.add_argument("--ssl-batch-size", type=int, default=8)
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--psd-channel-mask-prob", type=float, default=0.15)
    parser.add_argument("--psd-frequency-mask-prob", type=float, default=0.15)
    parser.add_argument("--psd-element-mask-prob", type=float, default=0.02)
    parser.add_argument("--fc-node-mask-prob", type=float, default=0.05)
    parser.add_argument("--fc-edge-mask-prob", type=float, default=0.15)
    parser.add_argument("--fc-band-mask-prob", type=float, default=0.05)
    parser.add_argument("--eo-ec-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for feature-space NT-Xent contrastive loss. Set >0 for multi-task SSL.",
    )
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--contrastive-noise-std", type=float, default=0.02)
    parser.add_argument("--contrastive-feature-mask-prob", type=float, default=0.05)
    parser.add_argument(
        "--alignment-method",
        choices=("ntxent", "vicreg", "barlow", "byol"),
        default="ntxent",
        help="Feature-space alignment loss used when --contrastive-weight is greater than zero.",
    )
    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--vicreg-variance-target", type=float, default=1.0)
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    parser.add_argument("--byol-momentum", type=float, default=0.99)
    parser.add_argument("--byol-predictor-hidden-dim", type=int, default=64)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--supervised-weight-decay", type=float, default=0.0)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--negative-class-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--transfer-mode", choices=("finetune", "freeze-encoder"), default="finetune")
    parser.add_argument(
        "--limit-ssl-pairs",
        type=int,
        default=None,
        help="Optional cap after scope selection for smoke tests. Omit for real runs.",
    )
    parser.add_argument(
        "--summary-name",
        default="masked_ssl_psd_fc_wpli_transfer_summary.csv",
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

    folds = make_loso_folds([record.subject_id for record in supervised_records])
    pretrained_state_by_test_subject = {}
    ssl_history_frames: list[pd.DataFrame] = []
    for fold in folds:
        ssl_pairs = feature_ssl_pairs_for_scope(
            all_ssl_pairs,
            data_scope=args.data_scope,
            strict_loso_test_subject_id=fold.test_subject_id,
        )
        if args.limit_ssl_pairs is not None:
            if args.limit_ssl_pairs < 1:
                raise SystemExit("--limit-ssl-pairs must be at least 1 when provided.")
            ssl_pairs = ssl_pairs[: args.limit_ssl_pairs]
        ssl_config = MaskedSSLConfig(
            psd_epochs=args.psd_ssl_epochs,
            fc_epochs=args.fc_ssl_epochs,
            batch_size=args.ssl_batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.decoder_hidden_dim,
            lr=args.ssl_lr,
            dropout=args.dropout,
            psd_channel_mask_prob=args.psd_channel_mask_prob,
            psd_frequency_mask_prob=args.psd_frequency_mask_prob,
            psd_element_mask_prob=args.psd_element_mask_prob,
            fc_node_mask_prob=args.fc_node_mask_prob,
            fc_edge_mask_prob=args.fc_edge_mask_prob,
            fc_band_mask_prob=args.fc_band_mask_prob,
            eo_ec_consistency_weight=args.eo_ec_consistency_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            projection_dim=args.projection_dim,
            contrastive_noise_std=args.contrastive_noise_std,
            contrastive_feature_mask_prob=args.contrastive_feature_mask_prob,
            alignment_method=args.alignment_method,
            vicreg_invariance_weight=args.vicreg_invariance_weight,
            vicreg_variance_weight=args.vicreg_variance_weight,
            vicreg_covariance_weight=args.vicreg_covariance_weight,
            vicreg_variance_target=args.vicreg_variance_target,
            barlow_offdiag_weight=args.barlow_offdiag_weight,
            byol_momentum=args.byol_momentum,
            byol_predictor_hidden_dim=args.byol_predictor_hidden_dim,
            device=args.device,
            seed=args.seed + fold.fold_index,
        )
        pretrained_state, ssl_history = run_masked_ssl_pretraining(ssl_pairs, ssl_config)
        ssl_history.insert(0, "fold_index", fold.fold_index)
        ssl_history.insert(1, "test_subject_id", fold.test_subject_id)
        ssl_history_frames.append(ssl_history)
        pretrained_state_by_test_subject[fold.test_subject_id] = pretrained_state
        print(
            f"[masked-ssl] fold={fold.fold_index} test={fold.test_subject_id} "
            f"pairs={len(ssl_pairs)}"
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
        weight_decay=args.supervised_weight_decay,
        positive_class_weight=args.positive_class_weight,
        negative_class_weight=args.negative_class_weight,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        seed=args.seed,
        pretrained_transfer_mode=args.transfer_mode,
    )
    predictions, metrics, supervised_loss = run_loso_supervised_with_history(
        supervised_records,
        supervised_config,
        pretrained_state_by_test_subject=pretrained_state_by_test_subject,
    )
    run_name = _build_run_name(args)
    ssl_history_all = pd.concat(ssl_history_frames, ignore_index=True)
    for frame in (predictions, metrics, supervised_loss, ssl_history_all):
        frame["ssl_data_scope"] = args.data_scope
        frame["run_name"] = run_name
        frame["masked_ssl"] = True
        frame["masked_ssl_reconstruction"] = "pre_pooling_local"
        frame["feature_contrastive_ssl"] = args.contrastive_weight > 0 and args.alignment_method == "ntxent"
        frame["feature_alignment_ssl"] = args.contrastive_weight > 0
        frame["feature_alignment_method"] = args.alignment_method if args.contrastive_weight > 0 else "none"
        frame["psd_ssl_method"] = (
            f"masked_local_reconstruction_plus_{args.alignment_method}_alignment"
            if args.contrastive_weight > 0 and args.alignment_method == "vicreg"
            else "masked_local_reconstruction_plus_feature_contrastive"
            if args.contrastive_weight > 0
            else "masked_local_reconstruction"
        )
        frame["fc_ssl_method"] = (
            f"graph_edge_masked_local_reconstruction_plus_{args.alignment_method}_alignment"
            if args.contrastive_weight > 0 and args.alignment_method == "vicreg"
            else "graph_edge_masked_local_reconstruction_plus_feature_contrastive"
            if args.contrastive_weight > 0
            else "graph_edge_masked_local_reconstruction"
        )
        frame["psd_ssl_epochs"] = args.psd_ssl_epochs
        frame["fc_ssl_epochs"] = args.fc_ssl_epochs
        frame["psd_channel_mask_prob"] = args.psd_channel_mask_prob
        frame["psd_frequency_mask_prob"] = args.psd_frequency_mask_prob
        frame["psd_element_mask_prob"] = args.psd_element_mask_prob
        frame["fc_node_mask_prob"] = args.fc_node_mask_prob
        frame["fc_edge_mask_prob"] = args.fc_edge_mask_prob
        frame["fc_band_mask_prob"] = args.fc_band_mask_prob
        frame["eo_ec_consistency_weight"] = args.eo_ec_consistency_weight
        frame["contrastive_weight"] = args.contrastive_weight
        frame["contrastive_temperature"] = args.contrastive_temperature
        frame["projection_dim"] = args.projection_dim
        frame["contrastive_noise_std"] = args.contrastive_noise_std
        frame["contrastive_feature_mask_prob"] = args.contrastive_feature_mask_prob
        frame["alignment_method"] = args.alignment_method
        frame["vicreg_invariance_weight"] = args.vicreg_invariance_weight
        frame["vicreg_variance_weight"] = args.vicreg_variance_weight
        frame["vicreg_covariance_weight"] = args.vicreg_covariance_weight
        frame["vicreg_variance_target"] = args.vicreg_variance_target
        frame["barlow_offdiag_weight"] = args.barlow_offdiag_weight
        frame["byol_momentum"] = args.byol_momentum
        frame["byol_predictor_hidden_dim"] = args.byol_predictor_hidden_dim
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


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


def _supervised_run_suffix(args: argparse.Namespace) -> str:
    supervised_lr = getattr(args, "supervised_lr", 1e-3)
    supervised_weight_decay = getattr(args, "supervised_weight_decay", 0.0)
    positive_class_weight = getattr(args, "positive_class_weight", 1.0)
    negative_class_weight = getattr(args, "negative_class_weight", 1.0)
    embedding_dim = getattr(args, "embedding_dim", 32)
    projection_dim = getattr(args, "projection_dim", 16)
    dropout = getattr(args, "dropout", 0.0)
    ssl_lr = getattr(args, "ssl_lr", 1e-3)
    eo_ec_consistency_weight = getattr(args, "eo_ec_consistency_weight", 0.0)
    psd_channel_mask_prob = getattr(args, "psd_channel_mask_prob", 0.15)
    psd_frequency_mask_prob = getattr(args, "psd_frequency_mask_prob", 0.15)
    psd_element_mask_prob = getattr(args, "psd_element_mask_prob", 0.02)
    suffix = ""
    if embedding_dim != 32:
        suffix += f"_ed{embedding_dim}"
    if projection_dim != 16:
        suffix += f"_pd{projection_dim}"
    if dropout != 0.0:
        suffix += f"_dp{_number_token(dropout)}"
    if ssl_lr != 1e-3:
        suffix += f"_ssllr{_number_token(ssl_lr)}"
    if eo_ec_consistency_weight != 0.0:
        suffix += f"_cons{_number_token(eo_ec_consistency_weight)}"
    if psd_channel_mask_prob != 0.15:
        suffix += f"_pcm{_number_token(psd_channel_mask_prob)}"
    if psd_frequency_mask_prob != 0.15:
        suffix += f"_pfm{_number_token(psd_frequency_mask_prob)}"
    if psd_element_mask_prob != 0.02:
        suffix += f"_pem{_number_token(psd_element_mask_prob)}"
    if supervised_lr != 1e-3:
        suffix += f"_slr{_number_token(supervised_lr)}"
    if supervised_weight_decay != 0.0:
        suffix += f"_swd{_number_token(supervised_weight_decay)}"
    if positive_class_weight != 1.0:
        suffix += f"_pcw{_number_token(positive_class_weight)}"
    if negative_class_weight != 1.0:
        suffix += f"_ncw{_number_token(negative_class_weight)}"
    return suffix


def _build_run_name(args: argparse.Namespace) -> str:
    supervised_suffix = _supervised_run_suffix(args)
    if args.contrastive_weight > 0 and args.alignment_method == "vicreg":
        return (
            f"mtvicreg_{args.data_scope}_psdfcwpli_gated_{args.transfer_mode}"
            f"_s{args.seed}_p{args.psd_ssl_epochs}_f{args.fc_ssl_epochs}"
            f"_fn{_number_token(args.fc_node_mask_prob)}_fe{_number_token(args.fc_edge_mask_prob)}"
            f"_fb{_number_token(args.fc_band_mask_prob)}_w{_number_token(args.contrastive_weight)}"
            f"_vi{_number_token(args.vicreg_invariance_weight)}"
            f"_vv{_number_token(args.vicreg_variance_weight)}"
            f"_vc{_number_token(args.vicreg_covariance_weight)}"
            f"{supervised_suffix}_bs{args.ssl_batch_size}_sup{args.supervised_epochs}"
        )
    if args.contrastive_weight > 0 and args.alignment_method == "barlow":
        return (
            f"mtbarlow_{args.data_scope}_psdfcwpli_gated_{args.transfer_mode}"
            f"_s{args.seed}_p{args.psd_ssl_epochs}_f{args.fc_ssl_epochs}"
            f"_fn{_number_token(args.fc_node_mask_prob)}_fe{_number_token(args.fc_edge_mask_prob)}"
            f"_fb{_number_token(args.fc_band_mask_prob)}_w{_number_token(args.contrastive_weight)}"
            f"_bo{_number_token(args.barlow_offdiag_weight)}"
            f"{supervised_suffix}_bs{args.ssl_batch_size}_sup{args.supervised_epochs}"
        )
    if args.contrastive_weight > 0 and args.alignment_method == "byol":
        return (
            f"mtbyol_{args.data_scope}_psdfcwpli_gated_{args.transfer_mode}"
            f"_s{args.seed}_p{args.psd_ssl_epochs}_f{args.fc_ssl_epochs}"
            f"_fn{_number_token(args.fc_node_mask_prob)}_fe{_number_token(args.fc_edge_mask_prob)}"
            f"_fb{_number_token(args.fc_band_mask_prob)}_w{_number_token(args.contrastive_weight)}"
            f"_m{_number_token(args.byol_momentum)}"
            f"{supervised_suffix}_bs{args.ssl_batch_size}_sup{args.supervised_epochs}"
        )
    if args.contrastive_weight > 0 and args.alignment_method == "ntxent":
        return (
            f"mtntxent_{args.data_scope}_psdfcwpli_gated_{args.transfer_mode}"
            f"_s{args.seed}_p{args.psd_ssl_epochs}_f{args.fc_ssl_epochs}"
            f"_fn{_number_token(args.fc_node_mask_prob)}_fe{_number_token(args.fc_edge_mask_prob)}"
            f"_fb{_number_token(args.fc_band_mask_prob)}_w{_number_token(args.contrastive_weight)}"
            f"_t{_number_token(args.contrastive_temperature)}"
            f"{supervised_suffix}_bs{args.ssl_batch_size}_sup{args.supervised_epochs}"
        )
    if args.contrastive_weight <= 0:
        ssl_prefix = "masked_local_ssl"
        return (
            f"{ssl_prefix}_{args.data_scope}_psd_masked_wpli_edge_gated_cnn"
            f"_{args.transfer_mode}_seed{args.seed}_psd{args.psd_ssl_epochs}_fc{args.fc_ssl_epochs}"
            f"_fcnode{_number_token(args.fc_node_mask_prob)}_fcedge{_number_token(args.fc_edge_mask_prob)}"
            f"_fcband{_number_token(args.fc_band_mask_prob)}_cons{_number_token(args.eo_ec_consistency_weight)}"
            f"_ctr{_number_token(args.contrastive_weight)}_temp{_number_token(args.contrastive_temperature)}"
            f"_noise{_number_token(args.contrastive_noise_std)}_mask{_number_token(args.contrastive_feature_mask_prob)}"
            f"{supervised_suffix}_bs{args.ssl_batch_size}_sup{args.supervised_epochs}"
        )
    raise ValueError(f"Unsupported alignment method: {args.alignment_method}")


if __name__ == "__main__":
    main()
