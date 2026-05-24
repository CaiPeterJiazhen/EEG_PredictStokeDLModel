from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.config import PathConfig
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.dual_state_model import DualStateEEGModel
from eeg_recovery.models.fusion_3d_model import Fusion3DEEGModel
from eeg_recovery.models.multimodal_model import MultimodalEEGModel, branches_for_feature_kind
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.schedulers import EarlyStopping, build_reduce_on_plateau


@dataclass(frozen=True)
class SupervisedFeatureRecord:
    subject_id: str
    label: int
    eo: np.ndarray
    ec: np.ndarray
    modalities: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    architecture: str = "dual_state"
    feature_kind: str = "psd"
    fusion: str = "concat"
    encoder_kind: str = "cnn"
    device: str = "auto"
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    positive_class_weight: float = 1.0
    negative_class_weight: float = 1.0
    embedding_dim: int = 16
    dropout: float = 0.1
    seed: int = 42
    pretrained_transfer_mode: str = "finetune"


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)


def load_supervised_feature_records(
    config: PathConfig,
    label_table: pd.DataFrame,
    feature_kind: str = "psd",
) -> list[SupervisedFeatureRecord]:
    """Load full EO/EC tensors for supervised deep-learning LOSO training."""

    feature_kind = _normalize_feature_kind(feature_kind)
    rows = label_table.loc[:, ["subject_id", "label"]].copy()
    rows["subject_id"] = rows["subject_id"].map(normalize_subject_id)
    records: list[SupervisedFeatureRecord] = []
    branches = branches_for_feature_kind(feature_kind)

    for row in rows.itertuples(index=False):
        subject_id = normalize_subject_id(row.subject_id)
        modalities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for branch in branches:
            eo = _load_branch_array(config.output_root, subject_id, "EO", branch)
            ec = _load_branch_array(config.output_root, subject_id, "EC", branch)
            _validate_feature_shape(eo, branch, subject_id, "EO")
            _validate_feature_shape(ec, branch, subject_id, "EC")
            modalities[branch] = (
                eo.astype(np.float32, copy=False),
                ec.astype(np.float32, copy=False),
            )
        first_eo, first_ec = modalities[branches[0]]
        records.append(
            SupervisedFeatureRecord(
                subject_id=subject_id,
                label=int(row.label),
                eo=first_eo,
                ec=first_ec,
                modalities=modalities,
            )
        )
    return records


def run_loso_supervised(
    records: Iterable[SupervisedFeatureRecord],
    training_config: SupervisedTrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, metrics, _ = run_loso_supervised_with_history(records, training_config)
    return predictions, metrics


def run_loso_supervised_with_history(
    records: Iterable[SupervisedFeatureRecord],
    training_config: SupervisedTrainingConfig | None = None,
    *,
    pretrained_state_by_test_subject: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train a small supervised neural model with patient-level LOSO folds."""

    resolved = training_config or SupervisedTrainingConfig()
    _validate_architecture_feature_kind(resolved)
    _seed_everything(resolved.seed)
    device = resolve_device(resolved.device)
    records = list(records)
    if len(records) < 2:
        raise ValueError("Supervised LOSO requires at least 2 records.")

    subject_ids = [record.subject_id for record in records]
    record_by_subject = {record.subject_id: record for record in records}
    folds = make_loso_folds(subject_ids)
    rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    model_name = f"{resolved.architecture}_{resolved.feature_kind}_{resolved.fusion}_{resolved.encoder_kind}"

    for fold in folds:
        train_subjects = list(fold.train_subject_ids)
        fit_subjects, val_subjects = _train_validation_subjects(train_subjects, fold.fold_index)
        scaler = _fit_state_scaler(record_by_subject[subject_id] for subject_id in fit_subjects)
        model = _build_model(resolved).to(device)
        pretrained_loaded = _load_fold_pretrained_state(
            model,
            fold.test_subject_id,
            pretrained_state_by_test_subject,
        )
        transfer_mode = resolved.pretrained_transfer_mode if pretrained_loaded else "none"
        if pretrained_loaded:
            _apply_pretrained_transfer_mode(model, resolved.pretrained_transfer_mode)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable model parameters remain after applying transfer mode.")
        optimizer = _build_optimizer(trainable_parameters, resolved)
        scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, resolved.patience // 2))
        early_stopping = EarlyStopping(patience=resolved.patience, mode="min")

        train_batch = _make_batch(
            (record_by_subject[subject_id] for subject_id in fit_subjects),
            scaler,
            device,
            resolved.architecture,
        )
        val_batch = _make_batch(
            (record_by_subject[subject_id] for subject_id in val_subjects),
            scaler,
            device,
            resolved.architecture,
        )

        for epoch in range(1, resolved.epochs + 1):
            model.train()
            optimizer.zero_grad()
            probabilities = _model_probabilities(model, train_batch)
            loss = _weighted_binary_cross_entropy(probabilities, train_batch["y"], resolved)
            loss.backward()
            optimizer.step()

            train_loss = float(loss.detach().cpu().item())
            val_loss = _evaluate_loss(model, val_batch, resolved)
            scheduler.step(val_loss)
            is_best_epoch = early_stopping.best_score is None or val_loss < early_stopping.best_score - early_stopping.min_delta
            stopped_early = early_stopping.step(val_loss, model)
            loss_rows.append(
                {
                    "model": model_name,
                    "fold_index": fold.fold_index,
                    "test_subject_id": fold.test_subject_id,
                    "fit_subject_ids": ";".join(fit_subjects),
                    "val_subject_ids": ";".join(val_subjects),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "weight_decay": resolved.weight_decay,
                    "positive_class_weight": resolved.positive_class_weight,
                    "negative_class_weight": resolved.negative_class_weight,
                    "is_best_epoch": bool(is_best_epoch),
                    "stopped_early": bool(stopped_early),
                    "architecture": resolved.architecture,
                    "feature_kind": resolved.feature_kind,
                    "fusion": resolved.fusion,
                    "encoder_kind": resolved.encoder_kind,
                    "pretrained": bool(pretrained_loaded),
                    "pretrained_transfer_mode": transfer_mode,
                }
            )
            if stopped_early:
                break
        early_stopping.restore_best_weights(model)

        test_record = record_by_subject[fold.test_subject_id]
        test_batch = _make_batch([test_record], scaler, device, resolved.architecture)
        model.eval()
        with torch.no_grad():
            probability = float(_model_probabilities(model, test_batch).detach().cpu().numpy()[0, 0])
        rows.append(
            {
                "model": model_name,
                "fold_index": fold.fold_index,
                "subject_id": fold.test_subject_id,
                "y_true": int(test_record.label),
                "y_score": probability,
                "y_pred": int(probability >= 0.5),
                "architecture": resolved.architecture,
                "feature_kind": resolved.feature_kind,
                "fusion": resolved.fusion,
                "encoder_kind": resolved.encoder_kind,
                "pretrained": bool(pretrained_loaded),
                "pretrained_transfer_mode": transfer_mode,
                "status": "trained",
                "skip_reason": "",
                "learning_rate": resolved.lr,
                "weight_decay": resolved.weight_decay,
                "positive_class_weight": resolved.positive_class_weight,
                "negative_class_weight": resolved.negative_class_weight,
                "embedding_dim": resolved.embedding_dim,
                "dropout": resolved.dropout,
                "seed": resolved.seed,
            }
        )

    sample_predictions = pd.DataFrame(rows)
    predictions = aggregate_patient_probabilities(sample_predictions)
    for column in (
        "model",
        "architecture",
        "feature_kind",
        "fusion",
        "encoder_kind",
        "pretrained",
        "pretrained_transfer_mode",
        "status",
        "skip_reason",
    ):
        predictions[column] = sample_predictions[column].iloc[0]
    metric_values = binary_classification_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_score"].to_numpy(dtype=float),
    )
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "architecture": resolved.architecture,
                "feature_kind": resolved.feature_kind,
                "fusion": resolved.fusion,
                "encoder_kind": resolved.encoder_kind,
                "pretrained": bool(
                    pretrained_state_by_test_subject is not None
                    and len(pretrained_state_by_test_subject) > 0
                ),
                "pretrained_transfer_mode": resolved.pretrained_transfer_mode
                if pretrained_state_by_test_subject
                else "none",
                "status": "trained",
                "skip_reason": "",
                "learning_rate": resolved.lr,
                "weight_decay": resolved.weight_decay,
                "positive_class_weight": resolved.positive_class_weight,
                "negative_class_weight": resolved.negative_class_weight,
                "embedding_dim": resolved.embedding_dim,
                "dropout": resolved.dropout,
                "seed": resolved.seed,
                **metric_values,
            }
        ]
    )
    loss_history = pd.DataFrame(loss_rows)
    return predictions, metrics, loss_history


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: SupervisedTrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)


def aggregate_patient_probabilities(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Aggregate sample-level predictions to patient-level mean probabilities."""

    required = {"subject_id", "y_true", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required column(s): {', '.join(sorted(missing))}")
    grouped = (
        predictions.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_score=("y_score", "mean"),
            fold_index=("fold_index", "first") if "fold_index" in predictions.columns else ("subject_id", "size"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    grouped["y_true"] = grouped["y_true"].astype(int)
    grouped["y_pred"] = (grouped["y_score"] >= threshold).astype(int)
    return grouped


def write_dl_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_root: str | Path,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    root = Path(output_root)
    if run_name is None:
        prediction_name = "dl_loso_predictions.csv"
        metric_name = "dl_model_comparison.csv"
    else:
        token = _safe_filename_token(run_name)
        prediction_name = f"dl_loso_predictions_{token}.csv"
        metric_name = f"dl_model_comparison_{token}.csv"
    prediction_path = root / "results" / "predictions" / prediction_name
    metric_path = root / "results" / "metrics" / metric_name
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(prediction_path, index=False)
    metrics_df.to_csv(metric_path, index=False)
    return prediction_path, metric_path


def write_loss_history_outputs(
    loss_history_df: pd.DataFrame,
    output_root: str | Path,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    """Write per-fold epoch loss values and a loss-vs-epoch curve."""

    if loss_history_df.empty:
        raise ValueError("loss_history_df is empty.")
    if "model" not in loss_history_df.columns:
        raise ValueError("loss_history_df must include a model column.")
    model_names = loss_history_df["model"].dropna().astype(str).unique()
    if len(model_names) != 1:
        raise ValueError("loss_history_df must contain exactly one model.")

    root = Path(output_root)
    output_token = _safe_filename_token(run_name or model_names[0])
    history_path = root / "results" / "training_logs" / f"dl_loss_history_{output_token}.csv"
    figure_path = root / "results" / "figures" / f"dl_loss_curve_{output_token}.png"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    loss_history_df.to_csv(history_path, index=False)
    _plot_loss_history(loss_history_df, figure_path)
    return history_path, figure_path


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not token:
        raise ValueError("model name cannot be converted to a safe filename.")
    return token


def _plot_loss_history(loss_history_df: pd.DataFrame, figure_path: Path) -> None:
    required = {"model", "fold_index", "epoch", "train_loss", "val_loss"}
    missing = required - set(loss_history_df.columns)
    if missing:
        raise ValueError(f"loss_history_df missing required column(s): {', '.join(sorted(missing))}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = loss_history_df.sort_values(["fold_index", "epoch"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, fold_rows in ordered.groupby("fold_index"):
        ax.plot(fold_rows["epoch"], fold_rows["train_loss"], color="#1f77b4", alpha=0.18, linewidth=0.8)
        ax.plot(fold_rows["epoch"], fold_rows["val_loss"], color="#ff7f0e", alpha=0.18, linewidth=0.8)

    epoch_means = ordered.groupby("epoch", as_index=False).agg(
        train_loss=("train_loss", "mean"),
        val_loss=("val_loss", "mean"),
    )
    ax.plot(epoch_means["epoch"], epoch_means["train_loss"], color="#1f77b4", linewidth=2.5, label="Train loss mean")
    ax.plot(epoch_means["epoch"], epoch_means["val_loss"], color="#ff7f0e", linewidth=2.5, label="Validation loss mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCELoss")
    ax.set_title(str(ordered["model"].iloc[0]))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def _build_model(config: SupervisedTrainingConfig) -> nn.Module:
    _validate_architecture_feature_kind(config)
    if config.architecture == "dual_state":
        return DualStateEEGModel(
            feature_kind=config.feature_kind,
            fusion=config.fusion,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
        )
    if config.architecture == "fusion_3d":
        return Fusion3DEEGModel(
            feature_kind=config.feature_kind,
            fusion=config.fusion,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
        )
    if config.architecture == "multimodal":
        return MultimodalEEGModel(
            feature_kind=config.feature_kind,
            fusion=config.fusion,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
        )
    raise ValueError("architecture must be 'dual_state', 'fusion_3d', or 'multimodal'.")


def _load_fold_pretrained_state(
    model: nn.Module,
    test_subject_id: str,
    pretrained_state_by_test_subject: Mapping[str, Mapping[str, torch.Tensor]] | None,
) -> bool:
    if not pretrained_state_by_test_subject:
        return False
    normalized_lookup = {
        normalize_subject_id(subject_id): state
        for subject_id, state in pretrained_state_by_test_subject.items()
    }
    state = normalized_lookup.get(normalize_subject_id(test_subject_id))
    if state is None:
        return False
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        joined = ", ".join(incompatible.unexpected_keys)
        raise ValueError(f"Pretrained state contains unexpected key(s): {joined}")
    return True


def _apply_pretrained_transfer_mode(model: nn.Module, transfer_mode: str) -> None:
    if transfer_mode == "finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if transfer_mode == "freeze-encoder":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not _is_encoder_parameter_name(name)
        return
    raise ValueError("pretrained_transfer_mode must be 'finetune' or 'freeze-encoder'.")


def _is_encoder_parameter_name(name: str) -> bool:
    if name.startswith("branch_models.") and ".encoder." in name:
        return True
    return name.startswith("encoder.")


def _validate_architecture_feature_kind(config: SupervisedTrainingConfig) -> None:
    if config.encoder_kind not in {"cnn", "linear"}:
        raise ValueError("encoder_kind must be 'cnn' or 'linear'.")
    if config.pretrained_transfer_mode not in {"finetune", "freeze-encoder"}:
        raise ValueError("pretrained_transfer_mode must be 'finetune' or 'freeze-encoder'.")
    if config.positive_class_weight <= 0 or config.negative_class_weight <= 0:
        raise ValueError("positive_class_weight and negative_class_weight must be positive.")
    branches = branches_for_feature_kind(config.feature_kind)
    if len(branches) > 1 and config.architecture != "multimodal":
        raise ValueError(
            f"feature_kind={config.feature_kind} uses multiple feature branches "
            f"{branches} and requires architecture=multimodal; got architecture={config.architecture}."
        )


def _model_probabilities(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "x" in batch:
        return model(batch["x"])
    if "eo" in batch and "ec" in batch:
        return model(batch["eo"], batch["ec"])
    return model(batch)


def _evaluate_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
) -> float:
    model.eval()
    with torch.no_grad():
        probabilities = _model_probabilities(model, batch)
        loss = _weighted_binary_cross_entropy(probabilities, batch["y"], config)
    return float(loss.detach().cpu().item())


def _weighted_binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    config: SupervisedTrainingConfig,
) -> torch.Tensor:
    weights = torch.where(
        targets >= 0.5,
        torch.as_tensor(config.positive_class_weight, dtype=probabilities.dtype, device=probabilities.device),
        torch.as_tensor(config.negative_class_weight, dtype=probabilities.dtype, device=probabilities.device),
    )
    return F.binary_cross_entropy(probabilities, targets, weight=weights)


def _make_batch(
    records: Iterable[SupervisedFeatureRecord],
    scaler: tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    architecture: str,
) -> dict[str, torch.Tensor]:
    records = list(records)
    y = np.array([[record.label] for record in records], dtype=np.float32)
    if architecture == "multimodal":
        if not isinstance(scaler, dict):
            raise ValueError("Multimodal batches require branch-specific scalers.")
        batch = {"y": torch.as_tensor(y, device=device)}
        for branch, (mean, std) in scaler.items():
            eo, ec = _branch_state_arrays(records, branch)
            batch[f"{branch}_eo"] = torch.as_tensor(((eo - mean) / std).astype(np.float32), device=device)
            batch[f"{branch}_ec"] = torch.as_tensor(((ec - mean) / std).astype(np.float32), device=device)
        return batch

    if isinstance(scaler, dict):
        if len(scaler) != 1:
            raise ValueError("dual_state and fusion_3d architectures require exactly one feature branch.")
        branch = next(iter(scaler))
        mean, std = scaler[branch]
        eo, ec = _branch_state_arrays(records, branch)
    else:
        mean, std = scaler
        eo = np.stack([record.eo for record in records]).astype(np.float32)
        ec = np.stack([record.ec for record in records]).astype(np.float32)
    eo = ((eo - mean) / std).astype(np.float32)
    ec = ((ec - mean) / std).astype(np.float32)
    if architecture == "fusion_3d":
        x = np.stack([eo, ec], axis=1)
        return {"x": torch.as_tensor(x, device=device), "y": torch.as_tensor(y, device=device)}
    return {
        "eo": torch.as_tensor(eo, device=device),
        "ec": torch.as_tensor(ec, device=device),
        "y": torch.as_tensor(y, device=device),
    }


def _fit_state_scaler(
    records: Iterable[SupervisedFeatureRecord],
) -> tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]]:
    records = list(records)
    if any(record.modalities is not None for record in records):
        branches = list(records[0].modalities or {})
        if not branches:
            raise ValueError("No modality branches are available to fit feature scaler.")
        return {
            branch: _fit_array_scaler(
                [
                    state
                    for record in records
                    for state in (record.modalities or {})[branch]
                ]
            )
            for branch in branches
        }

    states = []
    for record in records:
        states.extend([record.eo, record.ec])
    return _fit_array_scaler(states)


def _fit_array_scaler(states: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    states = list(states)
    if not states:
        raise ValueError("No records available to fit feature scaler.")
    stacked = np.stack(states).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def _branch_state_arrays(records: list[SupervisedFeatureRecord], branch: str) -> tuple[np.ndarray, np.ndarray]:
    eo = np.stack([(record.modalities or {})[branch][0] for record in records]).astype(np.float32)
    ec = np.stack([(record.modalities or {})[branch][1] for record in records]).astype(np.float32)
    return eo, ec


def _train_validation_subjects(train_subjects: list[str], fold_index: int) -> tuple[list[str], list[str]]:
    if len(train_subjects) < 3:
        return train_subjects, train_subjects
    validation_index = fold_index % len(train_subjects)
    val_subjects = [train_subjects[validation_index]]
    fit_subjects = [
        subject_id
        for index, subject_id in enumerate(train_subjects)
        if index != validation_index
    ]
    return fit_subjects, val_subjects


def _load_branch_array(output_root: Path, subject_id: str, state: str, branch: str) -> np.ndarray:
    if branch == "psd":
        return _load_feature_array(
            output_root / "data" / "features" / "psd" / f"{subject_id}_{state}_psd.npz",
            "psd",
        )
    if branch in {"wpli", "icoh"}:
        key = "wpli" if branch == "wpli" else "imaginary_coherence"
        return _load_feature_array(
            output_root / "data" / "features" / "fc" / f"{subject_id}_{state}_fc.npz",
            key,
        )
    raise ValueError(f"Unsupported feature branch: {branch}")


def _load_feature_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required feature file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise KeyError(f"Feature file {path} is missing required key {key!r}.")
        return np.asarray(payload[key], dtype=np.float32)


def _validate_feature_shape(array: np.ndarray, branch: str, subject_id: str, state: str) -> None:
    expected = (62, 90) if branch == "psd" else (1891, 6)
    if array.shape != expected:
        raise ValueError(
            f"Expected {branch} shape {expected} for {subject_id} {state}, got {array.shape}."
        )


def _normalize_feature_kind(feature_kind: str) -> str:
    if feature_kind in {"psd", "fc-wpli", "fc-icoh", "fc-both", "psd-fc-wpli", "psd-fc-icoh", "psd-fc-both"}:
        return feature_kind
    if feature_kind == "fc":
        return "fc-wpli"
    raise ValueError(
        "feature_kind must be 'psd', 'fc-wpli', 'fc-icoh', 'fc-both', "
        "'psd-fc-wpli', 'psd-fc-icoh', or 'psd-fc-both'."
    )


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
