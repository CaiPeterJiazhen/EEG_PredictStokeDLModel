from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.loso import LOSOFold, make_loso_folds
from eeg_recovery.training.segment_ssl_dataset import load_segment_ssl_records, segment_records_for_scope
from eeg_recovery.training.ssl_checkpointing import save_ssl_encoder_checkpoint
from eeg_recovery.training.train_segment_mil import (
    FOLDSTRICT_WPLI_CHECKPOINT_TAG,
    SegmentMILTrainingConfig,
    expected_wpli_segment_barlow_metadata,
    load_wpli_segment_bag_records,
    resolve_existing_wpli_segment_barlow_encoder_for_fold,
    run_loso_wpli_segment_mil_with_history,
    wpli_segment_barlow_checkpoint_path,
    write_segment_mil_outputs,
)
from eeg_recovery.training.train_segment_ssl import (
    SegmentSSLTrainingConfig,
    extract_branch_encoder_state,
    run_segment_ssl_pretraining,
    segment_records_manifest_hash,
)


RUN_PREFIX = "wpli_segbarlow_attention_mil"


def main() -> None:
    args = _parse_args()
    if args.reuse_only and not args.reuse_ssl_encoders:
        raise SystemExit("--reuse-only requires --reuse-ssl-encoders.")

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    segment_records = load_segment_ssl_records(
        path_config.output_root / "data" / "features" / "segment_level" / "fc",
        branches=("wpli",),
    )
    psd_segment_records = load_segment_ssl_records(
        path_config.output_root / "data" / "features" / "segment_level" / "psd",
        branches=("psd",),
    )
    bags = load_wpli_segment_bag_records(segment_records, labels)
    folds = make_loso_folds([bag.subject_id for bag in bags])

    checkpoint_dir = Path(args.ssl_checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = path_config.output_root / checkpoint_dir
    dual_checkpoint_dir = checkpoint_dir
    if args.dual_ssl_checkpoint_dir:
        dual_checkpoint_dir = Path(args.dual_ssl_checkpoint_dir)
        if not dual_checkpoint_dir.is_absolute():
            dual_checkpoint_dir = path_config.output_root / dual_checkpoint_dir
    encoder_state_by_test_subject: dict[str, dict[str, torch.Tensor]] = {}
    for fold in folds:
        fold_segments = segment_records_for_scope(
            segment_records,
            data_scope=args.data_scope,
            strict_loso_test_subject_id=fold.test_subject_id,
            supervised_subject_ids=supervised_ids,
        )
        psd_fold_segments = segment_records_for_scope(
            psd_segment_records,
            data_scope=args.data_scope,
            strict_loso_test_subject_id=fold.test_subject_id,
            supervised_subject_ids=supervised_ids,
        )
        manifest_hash = segment_records_manifest_hash(fold_segments)
        dual_manifest_hash = segment_records_manifest_hash([*psd_fold_segments, *fold_segments])
        encoder_state = _run_or_load_wpli_encoder(
            args=args,
            checkpoint_dir=checkpoint_dir,
            fold=fold,
            fold_segments=fold_segments,
            manifest_hash=manifest_hash,
            dual_manifest_hash=dual_manifest_hash,
            dual_checkpoint_dir=dual_checkpoint_dir,
        )
        encoder_state_by_test_subject[fold.test_subject_id] = encoder_state
        print(
            f"[wpli-mil] fold={fold.fold_index} test={fold.test_subject_id} "
            f"ssl_segments={len(fold_segments)} encoder_keys={len(encoder_state)}"
        )

    config = SegmentMILTrainingConfig(
        device=args.device,
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        max_segments_per_state=args.max_segments_per_state,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        stage2_head_lr=args.stage2_head_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        freeze_bn=args.freeze_bn,
        reset_bn_running_stats=args.reset_bn_running_stats,
    )
    predictions, metrics, loss_history, attention_summary = run_loso_wpli_segment_mil_with_history(
        bags,
        config,
        encoder_state_by_test_subject=encoder_state_by_test_subject,
    )
    run_name = f"{RUN_PREFIX}_seed{args.seed}"
    if args.output_tag:
        run_name = f"{run_name}_{args.output_tag}"
    for frame in (predictions, metrics, loss_history, attention_summary):
        frame["run_name"] = run_name
        frame["segment_ssl"] = True
        frame["segment_ssl_objective"] = "barlow"
        frame["segment_ssl_feature_kind"] = "fc-wpli"
        frame["ssl_data_scope"] = args.data_scope
        frame["seed"] = args.seed
    paths = write_segment_mil_outputs(
        output_root=path_config.output_root,
        run_name=run_name,
        predictions=predictions,
        metrics=metrics,
        loss_history=loss_history,
        attention_summary=attention_summary,
    )
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote metrics: {paths['metrics']}")
    print(f"Wrote loss history: {paths['loss_history']}")
    print(f"Wrote loss curve: {paths['loss_curve']}")
    print(f"Wrote attention summary: {paths['attention_summary']}")

    if args.seed == 0 and not args.output_tag:
        comparison = _write_seed0_method_comparison(path_config.output_root, run_name)
        _write_seed0_results_doc(path_config.output_root, comparison)
        _append_task_status(path_config.output_root, comparison)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WPLI Segment Barlow + patient-level attention MIL seed0 pilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-segments-per-state", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--head-lr", type=float, default=0.002)
    parser.add_argument("--encoder-lr", type=float, default=0.0001)
    parser.add_argument("--stage2-head-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--save-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--ssl-checkpoint-dir", default="results/checkpoints/ssl_encoders")
    parser.add_argument("--dual-ssl-checkpoint-dir", default=None)
    parser.add_argument("--freeze-bn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-bn-running-stats", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--data-scope", default="all-patient")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=0.001)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--feature-mask-prob", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    return parser.parse_args()


def _run_or_load_wpli_encoder(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    fold: LOSOFold,
    fold_segments: list,
    manifest_hash: str,
    dual_manifest_hash: str,
    dual_checkpoint_dir: Path,
) -> dict[str, torch.Tensor]:
    if args.reuse_ssl_encoders:
        try:
            resolution = resolve_existing_wpli_segment_barlow_encoder_for_fold(
                checkpoint_dir=checkpoint_dir,
                fold=fold,
                seed=args.seed,
                embedding_dim=args.embedding_dim,
                pretrain_epochs=args.pretrain_epochs,
                ssl_data_scope=args.data_scope,
                source_feature_manifest_hash=manifest_hash,
                dual_source_feature_manifest_hash=dual_manifest_hash,
                dropout=args.dropout,
                pretrain_lr=args.pretrain_lr,
                dual_checkpoint_dir=dual_checkpoint_dir,
            )
            if resolution.source == "dual_extracted":
                print(
                    f"[wpli-mil] extracted WPLI encoder from dual checkpoint for "
                    f"fold={fold.fold_index} test={fold.test_subject_id}: "
                    f"{resolution.source_checkpoint_path} -> {resolution.checkpoint_path}"
                )
            else:
                print(
                    f"[wpli-mil] reused branch-specific WPLI checkpoint for "
                    f"fold={fold.fold_index} test={fold.test_subject_id}: {resolution.checkpoint_path}"
                )
            return resolution.encoder_state_dict
        except FileNotFoundError:
            if args.reuse_only:
                raise
            print(
                f"[wpli-mil] no strict reusable WPLI/dual SSL checkpoint for "
                f"fold={fold.fold_index} test={fold.test_subject_id}; will pretrain WPLI SSL."
            )
    elif args.reuse_only:
        raise FileNotFoundError("--reuse-only requested but --reuse-ssl-encoders was not enabled.")

    ssl_config = SegmentSSLTrainingConfig(
        objective="barlow",
        feature_kind="fc-wpli",
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        lr=args.pretrain_lr,
        feature_mask_prob=args.feature_mask_prob,
        noise_std=args.noise_std,
        barlow_offdiag_weight=args.barlow_offdiag_weight,
        lambda_latent=args.lambda_latent,
        lambda_local=args.lambda_local,
        device=args.device,
        seed=args.seed + fold.fold_index,
    )
    pretrained_state, _ = run_segment_ssl_pretraining(fold_segments, ssl_config)
    encoder_state = extract_branch_encoder_state(pretrained_state, "wpli")
    if args.save_ssl_encoders:
        path = wpli_segment_barlow_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            fold=fold,
            seed=args.seed,
            embedding_dim=args.embedding_dim,
            pretrain_epochs=args.pretrain_epochs,
            ssl_data_scope=args.data_scope,
            checkpoint_tag=FOLDSTRICT_WPLI_CHECKPOINT_TAG,
        )
        metadata = expected_wpli_segment_barlow_metadata(
            fold=fold,
            seed=args.seed,
            embedding_dim=args.embedding_dim,
            pretrain_epochs=args.pretrain_epochs,
            ssl_data_scope=args.data_scope,
            source_feature_manifest_hash=manifest_hash,
            dropout=args.dropout,
            pretrain_lr=args.pretrain_lr,
        )
        metadata.update(
            {
                "effective_seed": ssl_config.seed,
                "projection_dim": args.projection_dim,
                "batch_size": args.pretrain_batch_size,
                "feature_mask_prob": args.feature_mask_prob,
                "noise_std": args.noise_std,
                "lambda_latent": args.lambda_latent,
                "lambda_local": args.lambda_local,
                "barlow_offdiag_weight": args.barlow_offdiag_weight,
                "n_ssl_segments": len(fold_segments),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        save_ssl_encoder_checkpoint(path, metadata, encoder_state)
    return encoder_state


def _write_seed0_method_comparison(output_root: Path, g_run_name: str) -> pd.DataFrame:
    methods = [
        (
            "A_no_ssl_cnn",
            "no-SSL CNN",
            "dl_loso_predictions_no_ssl_patient_psd_fc_wpli_seed0_ep100.csv",
            "dl_model_comparison_no_ssl_patient_psd_fc_wpli_seed0_ep100.csv",
        ),
        (
            "C_wpli_segment_barlow_patient_averaged_cnn",
            "WPLI Segment Barlow patient-averaged CNN",
            "dl_loso_predictions_segssl_barlow_all-patient_fc-wpli_finetune_seed0_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv",
            "dl_model_comparison_segssl_barlow_all-patient_fc-wpli_finetune_seed0_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv",
        ),
        (
            "D_dual_segment_barlow_finetune",
            "Dual PSD+WPLI Segment Barlow finetune",
            "dl_loso_predictions_dual_segbarlow_psd_wpli_finetune_lr0_002_seed0.csv",
            "dl_model_comparison_dual_segbarlow_psd_wpli_finetune_lr0_002_seed0.csv",
        ),
        (
            "G_wpli_segment_barlow_attention_mil",
            "WPLI Segment Barlow attention MIL",
            f"dl_loso_predictions_{g_run_name}.csv",
            f"dl_model_comparison_{g_run_name}.csv",
        ),
    ]
    rows = []
    for method_id, method_name, prediction_name, metric_name in methods:
        prediction_path = output_root / "results" / "predictions" / prediction_name
        metric_path = output_root / "results" / "metrics" / metric_name
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        metric = pd.read_csv(metric_path).iloc[0].to_dict() if metric_path.exists() else {}
        y_true = predictions["y_true"].to_numpy(dtype=int)
        y_score = predictions["y_score"].to_numpy(dtype=float)
        y_pred = predictions["y_pred"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        computed_metrics = binary_classification_metrics(y_true, y_score)
        row = {
            "method_id": method_id,
            "method": method_name,
            "prediction_file": str(prediction_path),
            "metric_file": str(metric_path) if metric_path.exists() else "",
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        for column in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
        ):
            row[column] = computed_metrics.get(column, metric.get(column, np.nan))
        for subject_id in ("sub09", "sub14"):
            subject_row = predictions.loc[predictions["subject_id"] == subject_id]
            if subject_row.empty:
                row[f"{subject_id}_y_true"] = np.nan
                row[f"{subject_id}_y_score"] = np.nan
                row[f"{subject_id}_y_pred"] = np.nan
            else:
                observed = subject_row.iloc[0]
                row[f"{subject_id}_y_true"] = int(observed.y_true)
                row[f"{subject_id}_y_score"] = float(observed.y_score)
                row[f"{subject_id}_y_pred"] = int(observed.y_pred)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    path = output_root / "results" / "metrics" / "seed0_ssl_cnn_method_comparison_after_mil.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    print(f"Wrote comparison: {path}")
    return comparison


def _write_seed0_results_doc(output_root: Path, comparison: pd.DataFrame) -> None:
    recommendation = _continuation_recommendation(comparison)
    qc_text = _qc_doc_fragment(output_root)
    path = output_root / "docs" / "wpli_segment_barlow_attention_mil_seed0_results.md"
    lines = [
        "# WPLI Segment Barlow Attention MIL Seed0 Results",
        "",
        "## 1. Why Dual PSD+WPLI Is Not The Next Priority",
        "",
        "The seed0 pilot showed that dual PSD+WPLI Segment Barlow did not improve the main patient-level accuracy or balanced accuracy, and freezing the encoder degraded performance. The new G experiment therefore focuses on the stronger seed0 signal: WPLI Segment Barlow.",
        "",
        "## 2. Why WPLI Segment Barlow",
        "",
        "The WPLI Segment Barlow seed0 run had the strongest ROC-AUC among the segment SSL pilots, suggesting useful ordering information even when the fixed 0.5 threshold did not improve accuracy. MIL keeps that segment-level information instead of averaging it away before supervised training.",
        "",
        "## 3. Why Attention MIL Fits This EEG Setting",
        "",
        "Each patient has many baseline EO/EC WPLI segments, but the clinical recovery label is patient-level. Attention MIL treats segments as instances inside one patient bag, learns which segments carry signal, pools to one patient embedding, and computes one BCE loss per patient.",
        "",
        "## 4. LOSO Leakage Controls",
        "",
        "- The held-out LOSO subject is excluded from supervised fit and validation patients.",
        "- The fold-local scaler is fit only on fit subjects, not validation or test.",
        "- The WPLI Segment Barlow encoder is fold-specific and its checkpoint metadata must match branch, objective, fold index, test subject, excluded subject, seed, encoder kind, embedding dimension, and feature manifest hash.",
        "- Predictions and reported metrics are patient-level only.",
        "",
        "## 5. sub09/sub14 QC",
        "",
        qc_text,
        "",
        "## 6. Seed0 A/C/D/G Comparison",
        "",
        _comparison_markdown(comparison),
        "",
        "## 7. Continue Six Seeds?",
        "",
        recommendation,
        "",
        "## 8. Next Command If Continuing",
        "",
        "```powershell",
        "& 'E:\\ProgramFiles\\Anaconda\\Anaconda3\\python.exe' -B scripts/21_train_wpli_segbarlow_mil_seed0.py --config configs/paths.example.yaml --device cuda --seed 1 --stage1-epochs 20 --stage2-epochs 80 --max-segments-per-state 64 --head-lr 0.002 --encoder-lr 0.0001 --stage2-head-lr 0.001 --weight-decay 0.00001 --embedding-dim 32 --dropout 0.0 --reuse-ssl-encoders --save-ssl-encoders --freeze-bn",
        "```",
        "",
        "Repeat with `--seed 2`, `--seed 3`, `--seed 7`, and `--seed 13` only if the seed0 criteria justify continuation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote doc: {path}")


def _comparison_markdown(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "No comparison rows were available."
    columns = [
        "method_id",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "tn",
        "fp",
        "fn",
        "tp",
        "sub09_y_score",
        "sub09_y_pred",
        "sub14_y_score",
        "sub14_y_pred",
    ]
    available = [column for column in columns if column in comparison.columns]
    frame = comparison.loc[:, available].copy()
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.4g}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def _continuation_recommendation(comparison: pd.DataFrame) -> str:
    if comparison.empty or "G_wpli_segment_barlow_attention_mil" not in set(comparison["method_id"]):
        return "Do not continue yet; the G result row is missing."
    indexed = comparison.set_index("method_id")
    g = indexed.loc["G_wpli_segment_barlow_attention_mil"]
    metric_criteria = []
    subject_criteria = []
    for baseline_id in ("A_no_ssl_cnn", "C_wpli_segment_barlow_patient_averaged_cnn"):
        if baseline_id not in indexed.index:
            continue
        baseline = indexed.loc[baseline_id]
        if float(g.accuracy) > float(baseline.accuracy):
            metric_criteria.append(f"accuracy improved over {baseline_id}")
        if float(g.balanced_accuracy) > float(baseline.balanced_accuracy):
            metric_criteria.append(f"balanced accuracy improved over {baseline_id}")
        if float(g.roc_auc) > float(baseline.roc_auc) and float(g.pr_auc) > float(baseline.pr_auc):
            metric_criteria.append(f"ROC-AUC and PR-AUC improved over {baseline_id}")
        for subject_id in ("sub09", "sub14"):
            true_value = g.get(f"{subject_id}_y_true")
            if pd.isna(true_value):
                continue
            direction = 1.0 if int(true_value) == 1 else -1.0
            moved = (float(g[f"{subject_id}_y_score"]) - float(baseline[f"{subject_id}_y_score"])) * direction
            if moved > 0:
                subject_criteria.append(f"{subject_id} y_score moved toward the correct direction vs {baseline_id}")
    if metric_criteria:
        unique = sorted(set([*metric_criteria, *subject_criteria]))
        return "Recommend continuing seeds 1,2,3,7,13 because: " + "; ".join(unique) + "."
    if subject_criteria:
        unique = sorted(set(subject_criteria))
        return (
            "Seed0 meets only the subject-score movement criterion, not the aggregate metric criteria. "
            "A cautious continuation to seeds 1,2,3,7,13 is defensible because: "
            + "; ".join(unique)
            + ". Inspect attention concentration and sub09/sub14 QC before treating this as a method improvement."
        )
    return (
        "Do not launch the remaining seeds yet. First inspect whether attention concentrates on abnormal segments, "
        "review the sub09/sub14 QC audit, and decide whether to adjust MIL or return to the WPLI Barlow patient-averaged model."
    )


def _qc_doc_fragment(output_root: Path) -> str:
    summary_path = output_root / "results" / "metrics" / "eeg_qc_subject_summary.csv"
    outlier_path = output_root / "results" / "metrics" / "eeg_qc_feature_outliers.csv"
    if not summary_path.exists() or not outlier_path.exists():
        return "QC outputs were not found. Run `scripts/20_qc_error_subjects.py` before interpreting sub09/sub14."
    summary = pd.read_csv(summary_path)
    outliers = pd.read_csv(outlier_path)
    lines = []
    for subject_id in ("sub09", "sub14"):
        row = summary.loc[summary["subject_id"] == subject_id].iloc[0]
        subject_outliers = outliers.loc[
            (outliers["subject_id"] == subject_id)
            & (outliers["is_outlier"].astype(bool))
            & (~outliers["metric"].isin(["residual", "distance_to_threshold", "FMA_pre", "FMA_post"]))
        ]
        status = "objective QC outlier" if not subject_outliers.empty else "no objective QC outlier"
        lines.append(
            f"- {subject_id}: residual={row.residual:.3f}, distance_to_threshold={row.distance_to_threshold:.3f}, {status}."
        )
    return "\n".join(lines)


def _append_task_status(output_root: Path, comparison: pd.DataFrame) -> None:
    path = output_root / "docs" / "task_status.md"
    if not path.exists():
        return
    marker = "2026-05-27: Added WPLI Segment Barlow attention MIL seed0 pilot"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    recommendation = _continuation_recommendation(comparison)
    addition = (
        f"\n- {marker}. Generated patient-level MIL outputs, attention summary, "
        f"seed0 A/C/D/G comparison, and result documentation. {recommendation}\n"
    )
    path.write_text(text + addition, encoding="utf-8")


if __name__ == "__main__":
    main()
