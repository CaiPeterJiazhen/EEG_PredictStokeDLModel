from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.models.multimodal_model import branches_for_feature_kind
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.segment_barlow_model_selection import (
    classification_metrics_from_scores,
    equal_weight_model_ensemble,
)
from eeg_recovery.training.segment_ssl_dataset import load_segment_ssl_records, segment_records_for_scope
from eeg_recovery.training.ssl_checkpointing import (
    load_reusable_ssl_encoder_checkpoint,
    load_ssl_encoder_checkpoint,
    make_ssl_checkpoint_name,
    prefix_branch_encoder_state_dict,
)
from eeg_recovery.training.train_segment_ssl import segment_records_manifest_hash
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


DEFAULT_SEEDS = [0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hard-negative supervised fine-tuning from reusable Segment Barlow SSL encoder checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--segment-feature-kind", choices=("psd", "fc-wpli"), required=True)
    parser.add_argument("--supervised-feature-kind", choices=("psd-fc-wpli",), default="psd-fc-wpli")
    parser.add_argument("--objective", choices=("barlow",), default="barlow")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ssl-data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--ssl-checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-tag", default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--feature-mask-prob", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--masked-latent-loss", choices=("cosine", "mse"), default="cosine")
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    parser.add_argument("--loss-name", choices=("bce", "weighted_bce", "asymmetric_focal", "asymmetric_focal_fp_margin"), default="asymmetric_focal_fp_margin")
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--negative-class-weight", type=float, default=1.5)
    parser.add_argument("--focal-gamma-pos", type=float, default=1.0)
    parser.add_argument("--focal-gamma-neg", type=float, default=2.0)
    parser.add_argument("--fp-margin", type=float, default=0.60)
    parser.add_argument("--fp-penalty-weight", type=float, default=0.25)
    parser.add_argument("--supervised-lr", type=float, default=0.002)
    parser.add_argument("--supervised-weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--output-tag", default="hardneg")
    args = parser.parse_args()

    if not args.reuse_ssl_encoders or not args.reuse_only:
        raise SystemExit("Hard-negative fine-tuning requires --reuse-ssl-encoders and --reuse-only.")

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(
        path_config,
        labels,
        feature_kind=args.supervised_feature_kind,
    )
    segment_records = _load_segment_records(output_root, args.segment_feature_kind)
    folds = make_loso_folds([record.subject_id for record in supervised_records])

    for seed in args.seeds:
        pretrained_state_by_test_subject = {}
        for fold in folds:
            fold_segments = segment_records_for_scope(
                segment_records,
                data_scope=args.ssl_data_scope,
                strict_loso_test_subject_id=fold.test_subject_id,
                supervised_subject_ids=supervised_ids,
            )
            branch = _branch_for_segment_feature_kind(args.segment_feature_kind)
            checkpoint_path = _checkpoint_path(args, path_config, branch, seed, fold)
            current_hash = segment_records_manifest_hash(fold_segments)
            source_hash = _source_manifest_hash_for_checkpoint_reuse(
                fold_segments,
                output_root=output_root,
                checkpoint_path=checkpoint_path,
                current_hash=current_hash,
            )
            expected_metadata = _checkpoint_metadata(
                args=args,
                branch=branch,
                fold=fold,
                seed=seed,
                n_ssl_segments=len(fold_segments),
                source_feature_manifest_hash=source_hash,
            )
            prefixed_state, checkpoint = _load_reusable_branch_checkpoint(
                checkpoint_path,
                expected_metadata=expected_metadata,
                branch=branch,
            )
            pretrained_state_by_test_subject[fold.test_subject_id] = prefixed_state
            print(
                f"[hardneg] feature={args.segment_feature_kind} seed={seed} "
                f"fold={fold.fold_index} test={fold.test_subject_id} "
                f"checkpoint_reused=True tensors={len(checkpoint['encoder_state_dict'])}"
            )

        training_config = SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind=args.supervised_feature_kind,
            fusion="gated",
            encoder_kind="cnn",
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.supervised_lr,
            weight_decay=args.supervised_weight_decay,
            loss_name=args.loss_name,
            positive_class_weight=args.positive_class_weight,
            negative_class_weight=args.negative_class_weight,
            focal_gamma_pos=args.focal_gamma_pos,
            focal_gamma_neg=args.focal_gamma_neg,
            fp_margin=args.fp_margin,
            fp_penalty_weight=args.fp_penalty_weight,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            seed=seed,
            pretrained_transfer_mode="finetune",
        )
        predictions, metrics, loss_history = run_loso_supervised_with_history(
            supervised_records,
            training_config,
            pretrained_state_by_test_subject=pretrained_state_by_test_subject,
        )
        model_group = f"{_output_prefix(args.segment_feature_kind)}_segbarlow_hardneg"
        _annotate_outputs((predictions, metrics, loss_history), args=args, seed=seed, model_group=model_group)
        metric_update = classification_metrics_from_scores(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_score"].to_numpy(dtype=float),
            y_pred=predictions["y_pred"].to_numpy(dtype=int),
        )
        for key, value in metric_update.items():
            metrics[key] = value
        prediction_path, metric_path = _hardneg_output_paths(output_root, segment_feature_kind=args.segment_feature_kind, seed=seed)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
        metrics.to_csv(metric_path, index=False)
        _write_loss_history(output_root, args.segment_feature_kind, seed, loss_history)
        print(f"Wrote predictions: {prediction_path}")
        print(f"Wrote metrics: {metric_path}")

    _write_hardneg_ensemble_if_available(output_root, args.seeds)
    _write_seed0_comparison_if_available(output_root)


def _load_segment_records(output_root: Path, segment_feature_kind: str):
    segment_root = output_root / "data" / "features" / "segment_level"
    if segment_feature_kind == "psd":
        return load_segment_ssl_records(segment_root / "psd", branches=("psd",))
    return load_segment_ssl_records(segment_root / "fc", branches=("wpli",))


def _branch_for_segment_feature_kind(segment_feature_kind: str) -> str:
    branches = branches_for_feature_kind(segment_feature_kind)
    if len(branches) != 1:
        raise ValueError("Hard-negative script expects one segment feature branch at a time.")
    return branches[0]


def _checkpoint_path(args, path_config, branch: str, seed: int, fold) -> Path:
    checkpoint_dir = (
        Path(args.ssl_checkpoint_dir)
        if args.ssl_checkpoint_dir
        else Path(path_config.output_root) / "results" / "checkpoints" / "ssl_encoders"
    )
    return checkpoint_dir / make_ssl_checkpoint_name(
        method="segssl",
        branch=branch,
        ssl_objective=args.objective,
        seed=seed,
        fold_index=fold.fold_index,
        test_subject_id=fold.test_subject_id,
        embedding_dim=args.embedding_dim,
        pretrain_epochs=args.pretrain_epochs,
        ssl_data_scope=args.ssl_data_scope,
        tag=args.checkpoint_tag,
    )


def _checkpoint_metadata(
    *,
    args,
    branch: str,
    fold,
    seed: int,
    n_ssl_segments: int,
    source_feature_manifest_hash: str,
) -> dict[str, object]:
    return {
        "checkpoint_type": "segment_ssl_encoder",
        "segment_ssl_method": f"segment_{args.objective}",
        "ssl_objective": args.objective,
        "branch": branch,
        "base_seed": seed,
        "effective_seed": seed + fold.fold_index,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": args.ssl_data_scope,
        "historical_unlabeled_pretraining": args.ssl_data_scope in {"all-patient", "all-patient-health"},
        "segment_feature_kind": args.segment_feature_kind,
        "supervised_feature_kind": args.supervised_feature_kind,
        "feature_kind": args.segment_feature_kind,
        "encoder_kind": "cnn",
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "projection_dim": args.projection_dim,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_lr": args.pretrain_lr,
        "batch_size": args.pretrain_batch_size,
        "feature_mask_prob": args.feature_mask_prob,
        "noise_std": args.noise_std,
        "lambda_latent": args.lambda_latent,
        "lambda_local": args.lambda_local,
        "masked_latent_loss": args.masked_latent_loss,
        "barlow_offdiag_weight": args.barlow_offdiag_weight,
        "n_ssl_segments": n_ssl_segments,
        "source_feature_manifest_hash": source_feature_manifest_hash,
    }


def _load_reusable_branch_checkpoint(
    checkpoint_path: Path,
    *,
    expected_metadata: dict[str, object],
    branch: str,
) -> tuple[dict, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing reusable Segment Barlow checkpoint: {checkpoint_path}")
    checkpoint = load_reusable_ssl_encoder_checkpoint(
        checkpoint_path,
        expected_metadata=expected_metadata,
        reuse_only=True,
    )
    encoder_state = checkpoint.get("encoder_state_dict")
    if not isinstance(encoder_state, dict):
        raise ValueError(f"SSL checkpoint {checkpoint_path} is missing encoder_state_dict.")
    return prefix_branch_encoder_state_dict(branch, encoder_state), checkpoint


def _source_manifest_hash_for_checkpoint_reuse(records, *, output_root: Path, checkpoint_path: Path, current_hash: str) -> str:
    observed_hash = _checkpoint_source_manifest_hash(checkpoint_path)
    if not observed_hash or observed_hash == current_hash:
        return current_hash
    for source_root in _legacy_source_roots_from_checkpoint_manifest(checkpoint_path):
        legacy_hash = _segment_records_manifest_hash_with_source_root(records, output_root=output_root, source_root=source_root)
        if legacy_hash == observed_hash:
            return legacy_hash
    return current_hash


def _checkpoint_source_manifest_hash(checkpoint_path: Path) -> str | None:
    try:
        checkpoint = load_ssl_encoder_checkpoint(checkpoint_path)
    except (EOFError, OSError, RuntimeError, ValueError):
        return None
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("source_feature_manifest_hash")
    return value if isinstance(value, str) and value else None


def _legacy_source_roots_from_checkpoint_manifest(checkpoint_path: Path) -> list[Path]:
    manifest_path = checkpoint_path.parent / "segment_ssl_checkpoint_manifest.csv"
    if not manifest_path.exists():
        return []
    try:
        manifest = pd.read_csv(manifest_path, usecols=["checkpoint_path"])
    except (OSError, ValueError):
        return []

    roots: list[Path] = []
    seen: set[str] = set()
    for checkpoint_text in manifest["checkpoint_path"].dropna().astype(str):
        if Path(checkpoint_text).name != checkpoint_path.name:
            continue
        root = _project_root_from_checkpoint_text(checkpoint_text)
        if root is None:
            continue
        key = str(root)
        if key not in seen:
            roots.append(root)
            seen.add(key)
    return roots


def _project_root_from_checkpoint_text(checkpoint_text: str) -> Path | None:
    normalized = checkpoint_text.replace("/", "\\")
    marker = "\\results\\checkpoints\\ssl_encoders\\"
    index = normalized.lower().find(marker)
    if index < 0:
        return None
    return Path(normalized[:index])


def _segment_records_manifest_hash_with_source_root(records, *, output_root: Path, source_root: Path) -> str:
    output_root = Path(output_root).resolve()
    manifest_rows = []
    for record in records:
        record_source_path = Path(record.source_path).resolve()
        try:
            source_path = Path(source_root) / record_source_path.relative_to(output_root)
        except ValueError:
            source_path = record_source_path
        manifest_rows.append(
            {
                "group": record.group,
                "subject_id": record.subject_id,
                "subject_key": record.subject_key,
                "stage": record.stage,
                "state": record.state,
                "segment_index": record.segment_index,
                "branches": sorted(record.features),
                "source_path": str(source_path),
            }
        )
    payload = json.dumps(
        sorted(
            manifest_rows,
            key=lambda row: (
                row["group"],
                row["subject_key"],
                row["stage"],
                row["state"],
                row["segment_index"],
                row["source_path"],
            ),
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _annotate_outputs(frames, *, args, seed: int, model_group: str) -> None:
    for frame in frames:
        frame["model_group"] = model_group
        frame["segment_ssl"] = True
        frame["segment_ssl_objective"] = args.objective
        frame["segment_ssl_feature_kind"] = args.segment_feature_kind
        frame["ssl_data_scope"] = args.ssl_data_scope
        frame["hard_negative_finetune"] = True
        frame["output_tag"] = args.output_tag
        frame["loss_name"] = args.loss_name
        frame["positive_class_weight"] = args.positive_class_weight
        frame["negative_class_weight"] = args.negative_class_weight
        frame["focal_gamma_pos"] = args.focal_gamma_pos
        frame["focal_gamma_neg"] = args.focal_gamma_neg
        frame["fp_margin"] = args.fp_margin
        frame["fp_penalty_weight"] = args.fp_penalty_weight
        frame["seed"] = seed


def _hardneg_output_paths(output_root: Path, *, segment_feature_kind: str, seed: int) -> tuple[Path, Path]:
    prefix = _output_prefix(segment_feature_kind)
    prediction_path = output_root / "results" / "predictions" / f"dl_loso_predictions_{prefix}_segbarlow_hardneg_seed{seed}.csv"
    metric_path = output_root / "results" / "metrics" / f"dl_model_comparison_{prefix}_segbarlow_hardneg_seed{seed}.csv"
    return prediction_path, metric_path


def _output_prefix(segment_feature_kind: str) -> str:
    return "wpli" if segment_feature_kind == "fc-wpli" else "psd"


def _write_loss_history(output_root: Path, segment_feature_kind: str, seed: int, loss_history: pd.DataFrame) -> None:
    path = output_root / "results" / "training_logs" / f"dl_loss_history_{_output_prefix(segment_feature_kind)}_segbarlow_hardneg_seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    loss_history.to_csv(path, index=False)


def _write_hardneg_ensemble_if_available(output_root: Path, seeds: list[int]) -> None:
    per_seed = {}
    for seed in seeds:
        psd_path, _ = _hardneg_output_paths(output_root, segment_feature_kind="psd", seed=seed)
        wpli_path, _ = _hardneg_output_paths(output_root, segment_feature_kind="fc-wpli", seed=seed)
        if not psd_path.exists() or not wpli_path.exists():
            return
        psd = pd.read_csv(psd_path)
        wpli = pd.read_csv(wpli_path)
        ensemble = equal_weight_model_ensemble(
            {"psd_hardneg": psd, "wpli_hardneg": wpli},
            model_group="psd_wpli_segbarlow_hardneg_equal_weight",
            seed=seed,
        )
        ensemble["y_pred"] = (ensemble["y_score"].astype(float) >= 0.5).astype(int)
        per_seed[seed] = ensemble

    output_name = "seed0" if seeds == [0] else "10seed"
    predictions = pd.concat(per_seed.values(), ignore_index=True)
    prediction_path = output_root / "results" / "predictions" / f"ensemble_predictions_psd_wpli_segbarlow_hardneg_{output_name}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)

    metric_rows = []
    for seed, frame in per_seed.items():
        row = {
            "model_group": "psd_wpli_segbarlow_hardneg_equal_weight",
            "seed": seed,
            "threshold": 0.5,
            "threshold_method": "fixed_0.5",
            "ensemble_members": "psd_hardneg+wpli_hardneg",
            "ensemble_weighting": "equal",
            "n_subjects": len(frame),
            "n_seeds": len(per_seed),
        }
        row.update(
            classification_metrics_from_scores(
                frame["y_true"].to_numpy(dtype=int),
                frame["y_score"].to_numpy(dtype=float),
                y_pred=frame["y_pred"].to_numpy(dtype=int),
            )
        )
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metric_name = (
        "dl_model_comparison_psd_wpli_segbarlow_hardneg_ensemble_seed0.csv"
        if seeds == [0]
        else "segment_barlow_hardneg_10seed_ensemble_summary.csv"
    )
    metric_path = output_root / "results" / "metrics" / metric_name
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metric_path, index=False)


def _write_seed0_comparison_if_available(output_root: Path) -> None:
    hardneg_paths = {
        "wpli_hardneg": _hardneg_output_paths(output_root, segment_feature_kind="fc-wpli", seed=0)[0],
        "psd_hardneg": _hardneg_output_paths(output_root, segment_feature_kind="psd", seed=0)[0],
        "psd_wpli_hardneg_equal": output_root / "results" / "predictions" / "ensemble_predictions_psd_wpli_segbarlow_hardneg_seed0.csv",
    }
    if any(not path.exists() for path in hardneg_paths.values()):
        return

    rows = []
    for model_group, path in {
        "wpli_bce_baseline": _find_baseline_prediction(output_root, model_kind="wpli", seed=0),
        "psd_bce_baseline": _find_baseline_prediction(output_root, model_kind="psd", seed=0),
        "psd_wpli_bce_equal": _original_psd_wpli_ensemble_seed_path(output_root),
        **hardneg_paths,
    }.items():
        if path is None or not path.exists():
            continue
        frame = pd.read_csv(path)
        if "seed" in frame.columns and model_group == "psd_wpli_bce_equal":
            frame = frame.loc[frame["seed"].astype(str) == "0"].copy()
        rows.append(_comparison_row(frame, model_group=model_group, prediction_path=path))
    comparison = pd.DataFrame(rows)
    comparison_path = output_root / "results" / "metrics" / "segment_barlow_hardneg_seed0_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    _write_seed0_doc(output_root, comparison)


def _comparison_row(frame: pd.DataFrame, *, model_group: str, prediction_path: Path) -> dict[str, object]:
    y_pred = frame["y_pred"].to_numpy(dtype=int) if "y_pred" in frame.columns else (frame["y_score"].astype(float) >= 0.5).astype(int)
    row = {
        "model_group": model_group,
        "prediction_path": str(prediction_path),
        "n_subjects": int(frame["subject_id"].nunique()),
    }
    row.update(
        classification_metrics_from_scores(
            frame["y_true"].to_numpy(dtype=int),
            frame["y_score"].to_numpy(dtype=float),
            y_pred=y_pred,
        )
    )
    for subject_id in ("sub09", "sub14"):
        subject = frame.loc[frame["subject_id"].astype(str) == subject_id]
        if subject.empty:
            row[f"{subject_id}_y_score"] = pd.NA
            row[f"{subject_id}_y_pred"] = pd.NA
        else:
            row[f"{subject_id}_y_score"] = float(subject["y_score"].iloc[0])
            row[f"{subject_id}_y_pred"] = int(subject["y_pred"].iloc[0]) if "y_pred" in subject.columns else int(float(subject["y_score"].iloc[0]) >= 0.5)
    return row


def _find_baseline_prediction(output_root: Path, *, model_kind: str, seed: int) -> Path | None:
    roots = [output_root]
    if model_kind == "wpli":
        roots.append(output_root / "docs" / "experiment_exports" / "20260527_remote_fc_wpli_segssl")
    if model_kind == "psd":
        roots.append(output_root / "docs" / "experiment_exports" / "20260527_local_psd_segssl")
    for root in roots:
        for search_dir in (root, root / "predictions", root / "results" / "predictions"):
            if not search_dir.exists():
                continue
            for path in sorted(search_dir.glob(f"dl_loso_predictions_segssl_barlow_all-patient_*seed{seed}_*.csv")):
                name = path.name
                if "seedensemble" in name or "vicreg" in name:
                    continue
                if model_kind == "wpli" and _is_wpli_baseline_name(name):
                    return path
                if model_kind == "psd" and _is_psd_baseline_name(name):
                    return path
    return None


def _is_wpli_baseline_name(name: str) -> bool:
    return "_fc-wpli_" in name and "psd-fc-wpli" not in name


def _is_psd_baseline_name(name: str) -> bool:
    return "_psd_" in name or "psd-fc-wpli" in name


def _original_psd_wpli_ensemble_seed_path(output_root: Path) -> Path | None:
    path = output_root / "results" / "predictions" / "ensemble_predictions_psd_wpli_segbarlow_equal_weight_10seed.csv"
    return path if path.exists() else None


def _write_seed0_doc(output_root: Path, comparison: pd.DataFrame) -> None:
    path = output_root / "docs" / "segment_barlow_hard_negative_seed0_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Segment Barlow Hard-Negative Seed0 Pilot",
        "",
        "This pilot evaluates a specificity-aware hard-negative supervised fine-tuning objective using existing fold-specific Segment Barlow SSL encoder checkpoints. No SSL pretraining, MIL, or dual-encoder expansion was run.",
        "",
        "Repeated false positives were observed in prior locked results. A specificity-aware hard-negative objective was evaluated; sub09/sub14 are monitored only as post-hoc error-analysis subjects.",
        "",
        "## Seed0 Metrics",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Pilot Decision",
        "",
        _seed0_pilot_decision(comparison),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_delta(rows: dict[str, pd.Series], left: str, right: str, metric: str) -> float | None:
    if left not in rows or right not in rows:
        return None
    left_value = rows[left].get(metric)
    right_value = rows[right].get(metric)
    if pd.isna(left_value) or pd.isna(right_value):
        return None
    return float(left_value) - float(right_value)


def _seed0_pilot_decision(comparison: pd.DataFrame) -> str:
    rows = {str(row["model_group"]): row for _, row in comparison.iterrows()}
    wpli_ba = _metric_delta(rows, "wpli_hardneg", "wpli_bce_baseline", "balanced_accuracy")
    wpli_spec = _metric_delta(rows, "wpli_hardneg", "wpli_bce_baseline", "specificity")
    wpli_brier = _metric_delta(rows, "wpli_hardneg", "wpli_bce_baseline", "brier_score")
    psd_ba = _metric_delta(rows, "psd_hardneg", "psd_bce_baseline", "balanced_accuracy")
    psd_brier = _metric_delta(rows, "psd_hardneg", "psd_bce_baseline", "brier_score")
    ensemble_ba = _metric_delta(rows, "psd_wpli_hardneg_equal", "psd_wpli_bce_equal", "balanced_accuracy")
    ensemble_brier = _metric_delta(rows, "psd_wpli_hardneg_equal", "psd_wpli_bce_equal", "brier_score")

    return (
        "This seed0 pilot is treated as mixed/negative, so the 10-seed hard-negative run was not started from this pilot. "
        f"WPLI hard-negative changed balanced accuracy by {_format_delta(wpli_ba)}, specificity by {_format_delta(wpli_spec)}, "
        f"and Brier by {_format_delta(wpli_brier)} versus WPLI BCE. "
        f"PSD hard-negative changed balanced accuracy by {_format_delta(psd_ba)} and Brier by {_format_delta(psd_brier)} versus PSD BCE, "
        "so the calibration gain came with weaker classification. "
        f"The hard-negative equal-weight ensemble changed balanced accuracy by {_format_delta(ensemble_ba)} and Brier by {_format_delta(ensemble_brier)} "
        "versus the original equal-weight ensemble."
    )


def _format_delta(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


if __name__ == "__main__":
    main()
