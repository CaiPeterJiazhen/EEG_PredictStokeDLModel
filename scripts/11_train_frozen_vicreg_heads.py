from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.frozen_ssl_heads import (
    build_frozen_head_candidates,
    build_pair_difference_embedding,
    fit_predict_selected_frozen_head,
)
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.train_feature_ssl import (
    build_feature_ssl_pair_records_from_feature_records,
    compute_feature_records_for_eeg_records,
    feature_ssl_pairs_for_scope,
    widest_feature_ssl_scope,
)
from eeg_recovery.training.train_masked_ssl import MaskedSSLConfig, run_masked_ssl_pretraining
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES, select_ssl_records
from eeg_recovery.training.train_supervised import (
    _fit_state_scaler,
    _make_batch,
    load_supervised_feature_records,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VICReg SSL, freeze CNN encoders, and train small-sample supervised heads on frozen embeddings.",
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
    parser.add_argument("--fc-node-mask-prob", type=float, default=0.02)
    parser.add_argument("--fc-edge-mask-prob", type=float, default=0.05)
    parser.add_argument("--fc-band-mask-prob", type=float, default=0.02)
    parser.add_argument("--eo-ec-consistency-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.001)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--contrastive-noise-std", type=float, default=0.02)
    parser.add_argument("--contrastive-feature-mask-prob", type=float, default=0.05)
    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("accuracy", "balanced_accuracy"), default="balanced_accuracy")
    parser.add_argument("--threshold-strategy", choices=("inner-loso", "fixed-0.5"), default="inner-loso")
    parser.add_argument("--embedding-view", choices=("fused", "pair-diff"), default="fused")
    parser.add_argument(
        "--candidate-names",
        default=None,
        help="Optional comma-separated frozen head candidate names. Omit to evaluate the full small-head registry.",
    )
    parser.add_argument(
        "--evaluate-all-candidates",
        action="store_true",
        help="Write predictions and metrics for every candidate head instead of selecting one head per fold.",
    )
    parser.add_argument("--summary-name", default="frozen_vicreg_head_summary.csv")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(path_config, labels, feature_kind="psd-fc-wpli")
    record_by_subject = {record.subject_id: record for record in supervised_records}
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

    device = resolve_device(args.device)
    candidate_names = _parse_candidate_names(args.candidate_names)
    folds = make_loso_folds([record.subject_id for record in supervised_records])
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    ssl_history_frames: list[pd.DataFrame] = []
    run_name = _build_run_name(args)

    for fold in folds:
        ssl_pairs = feature_ssl_pairs_for_scope(
            all_ssl_pairs,
            data_scope=args.data_scope,
            strict_loso_test_subject_id=fold.test_subject_id,
        )
        ssl_config = MaskedSSLConfig(
            psd_epochs=args.psd_ssl_epochs,
            fc_epochs=args.fc_ssl_epochs,
            batch_size=args.ssl_batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.decoder_hidden_dim,
            lr=args.ssl_lr,
            psd_channel_mask_prob=args.psd_channel_mask_prob,
            psd_frequency_mask_prob=args.psd_frequency_mask_prob,
            psd_element_mask_prob=args.psd_element_mask_prob,
            fc_node_mask_prob=args.fc_node_mask_prob,
            fc_edge_mask_prob=args.fc_edge_mask_prob,
            fc_band_mask_prob=args.fc_band_mask_prob,
            eo_ec_consistency_weight=args.eo_ec_consistency_weight,
            contrastive_weight=args.contrastive_weight,
            projection_dim=args.projection_dim,
            contrastive_noise_std=args.contrastive_noise_std,
            contrastive_feature_mask_prob=args.contrastive_feature_mask_prob,
            alignment_method="vicreg",
            vicreg_invariance_weight=args.vicreg_invariance_weight,
            vicreg_variance_weight=args.vicreg_variance_weight,
            vicreg_covariance_weight=args.vicreg_covariance_weight,
            device=args.device,
            seed=args.seed + fold.fold_index,
        )
        pretrained_state, ssl_history = run_masked_ssl_pretraining(ssl_pairs, ssl_config)
        ssl_history.insert(0, "fold_index", fold.fold_index)
        ssl_history.insert(1, "test_subject_id", fold.test_subject_id)
        ssl_history_frames.append(ssl_history)

        train_records = [record_by_subject[subject_id] for subject_id in fold.train_subject_ids]
        test_record = record_by_subject[fold.test_subject_id]
        x_train, x_test = _extract_frozen_embeddings(
            train_records,
            test_record,
            pretrained_state,
            embedding_dim=args.embedding_dim,
            embedding_view=args.embedding_view,
            device=device,
        )
        y_train = pd.Series([record.label for record in train_records], dtype=int).to_numpy()
        names_to_run = _candidate_names_for_fold(args, candidate_names)
        for head_names in names_to_run:
            head_result = fit_predict_selected_frozen_head(
                x_train,
                y_train,
                x_test,
                seed=args.seed + fold.fold_index,
                candidate_names=head_names,
                selection_metric=args.selection_metric,
                threshold_strategy=args.threshold_strategy,
            )
            prediction_rows.append(
                {
                    "model": "frozen_vicreg_cnn_head",
                    "fold_index": fold.fold_index,
                    "subject_id": fold.test_subject_id,
                    "y_true": int(test_record.label),
                    "y_score": float(head_result["y_score"]),
                    "y_score_raw": float(head_result["raw_score"]),
                    "y_pred": int(head_result["y_pred"]),
                    "threshold": float(head_result["threshold"]),
                    "selected_head": str(head_result["selected_head"]),
                    "inner_accuracy": float(head_result["inner_accuracy"]),
                    "inner_balanced_accuracy": float(head_result["inner_balanced_accuracy"]),
                    "architecture": "multimodal",
                    "feature_kind": "psd-fc-wpli",
                    "fusion": "gated",
                    "encoder_kind": "cnn",
                    "pretrained": True,
                    "pretrained_transfer_mode": "freeze-encoder-frozen-head",
                    "ssl_data_scope": args.data_scope,
                    "run_name": run_name,
                    "seed": args.seed,
                    "embedding_view": args.embedding_view,
                }
            )
            selection_rows.append({**prediction_rows[-1], "ssl_pairs": len(ssl_pairs)})
        print(
            f"[frozen-vicreg-head] fold={fold.fold_index} test={fold.test_subject_id} "
            f"heads={len(names_to_run)} true={test_record.label}"
        )

    predictions = pd.DataFrame(prediction_rows).sort_values("subject_id").reset_index(drop=True)
    metric_rows = []
    for selected_head, frame in predictions.groupby("selected_head", sort=True):
        metric_values = binary_classification_metrics(
            frame["y_true"].to_numpy(dtype=int),
            frame["y_score"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "model": "frozen_vicreg_cnn_head",
                "selected_head": selected_head,
                "architecture": "multimodal",
                "feature_kind": "psd-fc-wpli",
                "fusion": "gated",
                "encoder_kind": "cnn",
                "pretrained": True,
                "pretrained_transfer_mode": "freeze-encoder-frozen-head",
                "ssl_data_scope": args.data_scope,
                "run_name": run_name,
                "seed": args.seed,
                "selection_metric": args.selection_metric,
                "threshold_strategy": args.threshold_strategy,
                "contrastive_weight": args.contrastive_weight,
                "vicreg_invariance_weight": args.vicreg_invariance_weight,
                "vicreg_variance_weight": args.vicreg_variance_weight,
                "vicreg_covariance_weight": args.vicreg_covariance_weight,
                "embedding_dim": args.embedding_dim,
                "embedding_view": args.embedding_view,
                **metric_values,
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    selection = pd.DataFrame(selection_rows)
    ssl_history_all = pd.concat(ssl_history_frames, ignore_index=True)
    paths = _write_outputs(path_config.output_root, run_name, predictions, metrics, ssl_history_all, selection)
    summary_path = Path(path_config.output_root) / "results" / "metrics" / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_path, index=False)
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote metrics: {paths['metrics']}")
    print(f"Wrote SSL history: {paths['ssl_history']}")
    print(f"Wrote head selection: {paths['head_selection']}")
    print(f"Wrote summary: {summary_path}")


def _extract_frozen_embeddings(
    train_records: list,
    test_record,
    pretrained_state: dict[str, torch.Tensor],
    *,
    embedding_dim: int,
    embedding_view: str,
    device: torch.device,
) -> tuple:
    scaler = _fit_state_scaler(train_records)
    model = MultimodalEEGModel(
        "psd-fc-wpli",
        fusion="gated",
        embedding_dim=embedding_dim,
        dropout=0.0,
        encoder_kind="cnn",
    ).to(device)
    incompatible = model.load_state_dict(pretrained_state, strict=False)
    if incompatible.unexpected_keys:
        joined = ", ".join(incompatible.unexpected_keys)
        raise ValueError(f"Pretrained state contains unexpected key(s): {joined}")
    model.eval()
    with torch.no_grad():
        train_batch = _make_batch(train_records, scaler, device, "multimodal")
        test_batch = _make_batch([test_record], scaler, device, "multimodal")
        x_train = _extract_embedding_view(model, train_batch, embedding_view)
        x_test = _extract_embedding_view(model, test_batch, embedding_view)
    return x_train, x_test


def _extract_embedding_view(model: MultimodalEEGModel, batch: dict[str, torch.Tensor], embedding_view: str):
    if embedding_view == "fused":
        return model.extract_embedding(batch).detach().cpu().numpy()
    if embedding_view == "pair-diff":
        parts = []
        for branch in model.branches:
            branch_model = model.branch_models[branch]
            eo_embedding = branch_model.encoder(batch[f"{branch}_eo"]).detach().cpu().numpy()
            ec_embedding = branch_model.encoder(batch[f"{branch}_ec"]).detach().cpu().numpy()
            parts.append(build_pair_difference_embedding(eo_embedding, ec_embedding))
        return pd.concat([pd.DataFrame(part) for part in parts], axis=1).to_numpy()
    raise ValueError("embedding_view must be 'fused' or 'pair-diff'.")


def _write_outputs(
    output_root: str | Path,
    run_name: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    ssl_history: pd.DataFrame,
    head_selection: pd.DataFrame,
) -> dict[str, Path]:
    root = Path(output_root)
    paths = {
        "predictions": root / "results" / "predictions" / f"dl_loso_predictions_{run_name}.csv",
        "metrics": root / "results" / "metrics" / f"dl_model_comparison_{run_name}.csv",
        "ssl_history": root / "results" / "ssl" / f"feature_ssl_pretraining_history_{run_name}.csv",
        "head_selection": root / "results" / "training_logs" / f"frozen_head_selection_{run_name}.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    ssl_history.to_csv(paths["ssl_history"], index=False)
    head_selection.to_csv(paths["head_selection"], index=False)
    return paths


def _parse_candidate_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise SystemExit("--candidate-names must contain at least one non-empty name when provided.")
    return names


def _candidate_names_for_fold(args: argparse.Namespace, candidate_names: tuple[str, ...] | None) -> list[tuple[str, ...] | None]:
    if not args.evaluate_all_candidates:
        return [candidate_names]
    names = candidate_names or tuple(build_frozen_head_candidates(args.seed).keys())
    return [(name,) for name in names]


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


def _candidate_token(candidate_names: str | None) -> str:
    names = _parse_candidate_names(candidate_names)
    if names is None:
        return "allheads"
    digest = hashlib.sha1(",".join(names).encode("utf-8")).hexdigest()[:8]
    if len(names) == 1:
        return f"head_{_safe_name_token(names[0], max_length=22)}_{digest}"
    return f"heads{len(names)}_{digest}"


def _safe_name_token(value: str, *, max_length: int = 96) -> str:
    token = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return token[:max_length]


def _build_run_name(args: argparse.Namespace) -> str:
    candidate_token = _candidate_token(args.candidate_names)
    if args.evaluate_all_candidates:
        candidate_token = "eval" + candidate_token
    threshold_token = args.threshold_strategy.replace(".", "").replace("-", "")
    metric_token = "balacc" if args.selection_metric == "balanced_accuracy" else "acc"
    scope_token = args.data_scope.replace("all-patient", "allpat").replace("supervised", "sup").replace("-", "")
    device_token = str(getattr(args, "device", "auto")).replace("-", "")
    return (
        f"fvicreg_{scope_token}_psdfcwpli_gcnn"
        f"_s{args.seed}_{device_token}_p{args.psd_ssl_epochs}_f{args.fc_ssl_epochs}"
        f"_fn{_number_token(args.fc_node_mask_prob)}_fe{_number_token(args.fc_edge_mask_prob)}"
        f"_fb{_number_token(args.fc_band_mask_prob)}_w{_number_token(args.contrastive_weight)}"
        f"_vi{_number_token(args.vicreg_invariance_weight)}"
        f"_vv{_number_token(args.vicreg_variance_weight)}"
        f"_vc{_number_token(args.vicreg_covariance_weight)}"
        f"_{args.embedding_view}_{candidate_token}_{metric_token}_{threshold_token}"
    )


if __name__ == "__main__":
    main()
