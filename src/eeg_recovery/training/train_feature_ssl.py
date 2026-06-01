from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.config import PathConfig
from eeg_recovery.features.connectivity import compute_connectivity_for_eeg_record
from eeg_recovery.features.psd import compute_psd_for_eeg_record
from eeg_recovery.io.index import EEGFileRecord
from eeg_recovery.metadata.labels import read_clinical_metadata
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.multimodal_model import MultimodalEEGModel, branches_for_feature_kind
from eeg_recovery.models.ssl_model import NTXentLoss
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.train_ssl import normalize_ssl_data_scope
from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    aggregate_patient_probabilities,
    _fit_state_scaler,
    _load_branch_array,
    _make_batch,
    _plot_loss_history,
    resolve_device,
)

FEATURE_SSL_OBJECTIVES = {
    "ntxent",
    "vicreg",
    "barlow",
    "branch-barlow",
    "masked-barlow",
    "masked-vicreg",
    "masked-reconstruction",
    "masked-reconstruction-vicreg",
    "byol",
    "simsiam",
}


@dataclass(frozen=True)
class FeatureSSLPairRecord:
    group: str
    subject_id: str
    subject_key: str
    stage: str
    is_supervised_subject: bool
    modalities: dict[str, tuple[np.ndarray, np.ndarray]]
    qeeg_target: float | None = None


@dataclass(frozen=True)
class FeatureSSLTrainingConfig:
    data_scope: str = "supervised-baseline"
    feature_kind: str = "psd-fc-wpli"
    fusion: str = "gated"
    encoder_kind: str = "cnn"
    epochs: int = 20
    batch_size: int = 8
    embedding_dim: int = 32
    projection_dim: int = 16
    dropout: float = 0.0
    lr: float = 1e-3
    ssl_objective: str = "ntxent"
    temperature: float = 0.2
    noise_std: float = 0.02
    feature_mask_prob: float = 0.05
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_variance_target: float = 1.0
    vicreg_eps: float = 1e-4
    barlow_offdiag_weight: float = 0.005
    barlow_eps: float = 1e-9
    branch_barlow_weight: float = 0.5
    latent_modality_mask_prob: float = 0.5
    byol_momentum: float = 0.99
    byol_predictor_hidden_dim: int = 64
    device: str = "auto"
    seed: int = 42


def feature_ssl_pairs_for_scope(
    pairs: Iterable[FeatureSSLPairRecord],
    *,
    data_scope: str,
    strict_loso_test_subject_id: str | None = None,
) -> list[FeatureSSLPairRecord]:
    scope = normalize_ssl_data_scope(data_scope)
    excluded_subject = (
        normalize_subject_id(strict_loso_test_subject_id)
        if strict_loso_test_subject_id is not None
        else None
    )

    selected = []
    for pair in pairs:
        if (
            excluded_subject is not None
            and pair.group == "patient"
            and pair.subject_id == excluded_subject
        ):
            continue
        if _pair_in_scope(pair, scope):
            selected.append(pair)

    return sorted(
        selected,
        key=lambda pair: (pair.group, pair.stage, pair.subject_key),
    )


def widest_feature_ssl_scope(data_scopes: Iterable[str]) -> str:
    """Return the smallest supported scope that covers every requested scope."""

    normalized = [normalize_ssl_data_scope(scope) for scope in data_scopes]
    if not normalized:
        raise ValueError("At least one data scope is required.")
    scope_set = set(normalized)
    if "all-patient-health" in scope_set:
        return "all-patient-health"
    if "health-only" in scope_set:
        return "health-only" if scope_set == {"health-only"} else "all-patient-health"
    ranks = {
        "supervised-baseline": 0,
        "all-patient-baseline": 1,
        "all-patient": 2,
    }
    return max(normalized, key=lambda scope: ranks[scope])


def build_feature_ssl_pair_records_from_feature_records(
    eeg_records: Iterable[EEGFileRecord],
    feature_records: Mapping[tuple[str, str, str], Mapping[str, np.ndarray]],
) -> list[FeatureSSLPairRecord]:
    grouped: dict[tuple[str, str], dict[str, EEGFileRecord]] = defaultdict(dict)
    for record in eeg_records:
        grouped[(record.subject_key, record.stage)][record.state] = record

    pairs: list[FeatureSSLPairRecord] = []
    incomplete = []
    for (subject_key, stage), state_records in grouped.items():
        if set(state_records) != {"EO", "EC"}:
            incomplete.append(f"{subject_key} {stage}")
            continue
        eo_record = state_records["EO"]
        ec_record = state_records["EC"]
        eo_features = feature_records.get((subject_key, stage, "EO"))
        ec_features = feature_records.get((subject_key, stage, "EC"))
        if eo_features is None or ec_features is None:
            incomplete.append(f"{subject_key} {stage}")
            continue
        branches = sorted(eo_features)
        if branches != sorted(ec_features):
            raise ValueError(f"EO/EC feature branches do not match for {subject_key} {stage}.")
        pairs.append(
            FeatureSSLPairRecord(
                group=eo_record.group,
                subject_id=eo_record.subject_id,
                subject_key=subject_key,
                stage=stage,
                is_supervised_subject=eo_record.is_supervised_subject,
                modalities={
                    branch: (
                        np.asarray(eo_features[branch], dtype=np.float32),
                        np.asarray(ec_features[branch], dtype=np.float32),
                    )
                    for branch in branches
                },
            )
        )
    if incomplete:
        joined = "; ".join(sorted(incomplete))
        raise ValueError(f"Feature SSL requires complete EO/EC pairs; incomplete: {joined}")
    return sorted(pairs, key=lambda pair: (pair.group, pair.stage, pair.subject_key))


def compute_feature_records_for_eeg_records(
    config: PathConfig,
    eeg_records: Iterable[EEGFileRecord],
    *,
    feature_kind: str = "psd-fc-wpli",
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    branches = branches_for_feature_kind(feature_kind)
    if not set(branches).issubset({"psd", "wpli", "icoh"}):
        raise ValueError(f"Unsupported feature_kind for feature SSL: {feature_kind}")

    clinical = read_clinical_metadata(config.patient_info_clinical_xlsx)
    affected_by_subject = {
        normalize_subject_id(row.subject_id): str(row.affected_hand)
        for row in clinical.itertuples(index=False)
    }

    features: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for record in eeg_records:
        cached = _load_cached_supervised_baseline_features(config, record, branches)
        if cached is not None:
            features[(record.subject_key, record.stage, record.state)] = cached
            continue

        affected_hand = _affected_hand_for_record(record, affected_by_subject)
        branch_features: dict[str, np.ndarray] = {}
        if "psd" in branches:
            branch_features["psd"] = compute_psd_for_eeg_record(record, affected_hand).psd.astype(np.float32)
        if "wpli" in branches or "icoh" in branches:
            fc = compute_connectivity_for_eeg_record(record, affected_hand)
            if "wpli" in branches:
                branch_features["wpli"] = fc.wpli.astype(np.float32)
            if "icoh" in branches:
                branch_features["icoh"] = fc.imaginary_coherence.astype(np.float32)
        features[(record.subject_key, record.stage, record.state)] = branch_features
    return features


def run_feature_ssl_pretraining(
    pairs: Iterable[FeatureSSLPairRecord],
    training_config: FeatureSSLTrainingConfig,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    config = _validate_feature_ssl_config(training_config)
    _seed_feature_ssl_training(config.seed)
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Feature SSL pretraining requires at least one EO/EC pair.")

    device = resolve_device(config.device)
    records = [_pair_to_supervised_record(pair) for pair in pairs]
    scaler = _fit_state_scaler(records)
    model = MultimodalEEGModel(
        config.feature_kind,
        fusion=config.fusion,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        encoder_kind=config.encoder_kind,
    ).to(device)
    projection_input_dim = config.embedding_dim * len(branches_for_feature_kind(config.feature_kind))
    projection_head = nn.Linear(projection_input_dim, config.projection_dim).to(device)
    reconstruction_head = (
        nn.Linear(projection_input_dim, _feature_reconstruction_output_dim(records[0], model.branches)).to(device)
        if config.ssl_objective in {"masked-reconstruction", "masked-reconstruction-vicreg"}
        else None
    )
    branch_projection_heads = (
        nn.ModuleDict(
            {
                branch: nn.Linear(config.embedding_dim, config.projection_dim)
                for branch in branches_for_feature_kind(config.feature_kind)
            }
        ).to(device)
        if config.ssl_objective == "branch-barlow"
        else None
    )
    predictor_head = _build_byol_predictor(config).to(device) if config.ssl_objective in {"byol", "simsiam"} else None
    target_model = copy.deepcopy(model).to(device) if config.ssl_objective == "byol" else None
    target_projection_head = copy.deepcopy(projection_head).to(device) if config.ssl_objective == "byol" else None
    if target_model is not None and target_projection_head is not None:
        _freeze_module(target_model)
        _freeze_module(target_projection_head)
    optimizer_parameters = [*model.branch_models.parameters(), *projection_head.parameters()]
    if reconstruction_head is not None:
        optimizer_parameters.extend(reconstruction_head.parameters())
    if branch_projection_heads is not None:
        optimizer_parameters.extend(branch_projection_heads.parameters())
    if predictor_head is not None:
        optimizer_parameters.extend(predictor_head.parameters())
    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=config.lr,
    )
    loss_fn = NTXentLoss(temperature=config.temperature)
    rng = np.random.default_rng(config.seed)

    rows: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(records))
        epoch_losses: list[float] = []
        alignment_losses: list[float] = []
        vicreg_invariance_losses: list[float] = []
        vicreg_variance_losses: list[float] = []
        vicreg_covariance_losses: list[float] = []
        barlow_on_diag_losses: list[float] = []
        barlow_off_diag_losses: list[float] = []
        global_barlow_losses: list[float] = []
        branch_barlow_losses: list[float] = []
        byol_losses: list[float] = []
        reconstruction_losses: list[float] = []
        model.train()
        projection_head.train()
        if reconstruction_head is not None:
            reconstruction_head.train()
        if branch_projection_heads is not None:
            branch_projection_heads.train()
        if predictor_head is not None:
            predictor_head.train()
        for start in range(0, len(order), config.batch_size):
            batch_records = [records[int(index)] for index in order[start : start + config.batch_size]]
            batch = _make_batch(batch_records, scaler, device, "multimodal")
            view_a = _augment_feature_batch(
                batch,
                noise_std=config.noise_std,
                feature_mask_prob=config.feature_mask_prob,
                seed=config.seed + epoch * 100_000 + start,
            )
            view_b = _augment_feature_batch(
                batch,
                noise_std=config.noise_std,
                feature_mask_prob=config.feature_mask_prob,
                seed=config.seed + epoch * 100_000 + start + 1,
            )

            optimizer.zero_grad()
            if config.ssl_objective in {"masked-reconstruction", "masked-reconstruction-vicreg"}:
                if reconstruction_head is None:
                    raise ValueError("Masked reconstruction objective requires a reconstruction head.")
                if config.ssl_objective == "masked-reconstruction-vicreg":
                    components = _feature_masked_reconstruction_vicreg_components(
                        model,
                        projection_head,
                        reconstruction_head,
                        batch,
                        config,
                        seed=config.seed + epoch * 100_000 + start + 20_000,
                    )
                else:
                    components = _feature_masked_reconstruction_components(
                        model,
                        reconstruction_head,
                        batch,
                        config,
                        seed=config.seed + epoch * 100_000 + start + 20_000,
                    )
            elif config.ssl_objective == "byol":
                components = _feature_byol_components(
                    model,
                    projection_head,
                    predictor_head,
                    target_model,
                    target_projection_head,
                    view_a,
                    view_b,
                )
            elif config.ssl_objective == "simsiam":
                components = _feature_simsiam_components(
                    model,
                    projection_head,
                    predictor_head,
                    view_a,
                    view_b,
                )
            else:
                if config.ssl_objective in {"branch-barlow", "masked-barlow", "masked-vicreg"}:
                    branch_embeddings_a = model.extract_branch_embeddings(view_a)
                    branch_embeddings_b = model.extract_branch_embeddings(view_b)
                    if config.ssl_objective in {"masked-barlow", "masked-vicreg"}:
                        generator_a = torch.Generator(device=device)
                        generator_a.manual_seed(config.seed + epoch * 100_000 + start + 10_000)
                        generator_b = torch.Generator(device=device)
                        generator_b.manual_seed(config.seed + epoch * 100_000 + start + 10_001)
                        embedding_a = _apply_latent_modality_mask(
                            branch_embeddings_a,
                            branches=model.branches,
                            mask_prob=config.latent_modality_mask_prob,
                            generator=generator_a,
                        )
                        embedding_b = _apply_latent_modality_mask(
                            branch_embeddings_b,
                            branches=model.branches,
                            mask_prob=config.latent_modality_mask_prob,
                            generator=generator_b,
                        )
                    else:
                        embedding_a = torch.cat([branch_embeddings_a[branch] for branch in model.branches], dim=1)
                        embedding_b = torch.cat([branch_embeddings_b[branch] for branch in model.branches], dim=1)
                    projection_a = projection_head(embedding_a)
                    projection_b = projection_head(embedding_b)
                    if config.ssl_objective in {"masked-barlow", "masked-vicreg"}:
                        components = _feature_alignment_components(projection_a, projection_b, config, loss_fn)
                    else:
                        branch_projection_a = {
                            branch: branch_projection_heads[branch](branch_embeddings_a[branch])
                            for branch in model.branches
                        }
                        branch_projection_b = {
                            branch: branch_projection_heads[branch](branch_embeddings_b[branch])
                            for branch in model.branches
                        }
                        components = _feature_branch_barlow_twins_components(
                            projection_a,
                            projection_b,
                            branch_projection_a,
                            branch_projection_b,
                            config,
                        )
                else:
                    embedding_a = model.extract_embedding(view_a)
                    embedding_b = model.extract_embedding(view_b)
                    projection_a = projection_head(embedding_a)
                    projection_b = projection_head(embedding_b)
                    components = _feature_alignment_components(projection_a, projection_b, config, loss_fn)
            loss = components["loss"]
            loss.backward()
            optimizer.step()
            if target_model is not None and target_projection_head is not None:
                _update_target_module(target_model, model, config.byol_momentum)
                _update_target_module(target_projection_head, projection_head, config.byol_momentum)
            epoch_losses.append(float(loss.detach().cpu().item()))
            alignment_losses.append(float(components["loss"].detach().cpu().item()))
            vicreg_invariance_losses.append(float(components["invariance"].detach().cpu().item()))
            vicreg_variance_losses.append(float(components["variance"].detach().cpu().item()))
            vicreg_covariance_losses.append(float(components["covariance"].detach().cpu().item()))
            barlow_on_diag_losses.append(float(components["on_diag"].detach().cpu().item()))
            barlow_off_diag_losses.append(float(components["off_diag"].detach().cpu().item()))
            global_barlow_losses.append(float(components.get("global_barlow", components["on_diag"] + components["off_diag"]).detach().cpu().item()))
            branch_barlow_losses.append(float(components.get("branch_barlow", torch.zeros((), device=loss.device)).detach().cpu().item()))
            byol_losses.append(float(components["byol"].detach().cpu().item()))
            reconstruction_losses.append(float(components.get("reconstruction", torch.zeros((), device=loss.device)).detach().cpu().item()))

        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "objective": _feature_ssl_objective_label(config.ssl_objective),
                "ssl_objective": config.ssl_objective,
                "alignment_loss": float(np.mean(alignment_losses)),
                "vicreg_invariance_loss": float(np.mean(vicreg_invariance_losses)),
                "vicreg_variance_loss": float(np.mean(vicreg_variance_losses)),
                "vicreg_covariance_loss": float(np.mean(vicreg_covariance_losses)),
                "barlow_on_diag_loss": float(np.mean(barlow_on_diag_losses)),
                "barlow_off_diag_loss": float(np.mean(barlow_off_diag_losses)),
                "global_barlow_loss": float(np.mean(global_barlow_losses)),
                "branch_barlow_loss": float(np.mean(branch_barlow_losses)),
                "branch_barlow_weight": config.branch_barlow_weight,
                "latent_modality_mask_prob": config.latent_modality_mask_prob,
                "byol_loss": float(np.mean(byol_losses)),
                "reconstruction_loss": float(np.mean(reconstruction_losses)),
                "data_scope": normalize_ssl_data_scope(config.data_scope),
                "feature_kind": config.feature_kind,
                "fusion": config.fusion,
                "encoder_kind": config.encoder_kind,
                "n_pairs": len(records),
                "batch_size": config.batch_size,
                "embedding_dim": config.embedding_dim,
                "projection_dim": config.projection_dim,
                "learning_rate": config.lr,
            }
        )

    pretrained_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("branch_models.")
    }
    return pretrained_state, pd.DataFrame(rows)


def _seed_feature_ssl_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def aggregate_seed_ensemble_predictions(
    prediction_frames: Iterable[pd.DataFrame],
    *,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [frame.copy() for frame in prediction_frames]
    if not frames:
        raise ValueError("At least one prediction dataframe is required for seed ensembling.")
    for frame in frames:
        missing = {"subject_id", "fold_index", "y_true", "y_score"} - set(frame.columns)
        if missing:
            raise ValueError(f"Prediction dataframe missing required column(s): {', '.join(sorted(missing))}")
    merged = pd.concat(frames, ignore_index=True)
    label_counts = merged.groupby("subject_id")["y_true"].nunique()
    if int(label_counts.max()) != 1:
        raise ValueError("Seed prediction dataframes contain inconsistent y_true values.")

    ensemble = (
        merged.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_score=("y_score", "mean"),
            fold_index=("fold_index", "first"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    ensemble["y_pred"] = (ensemble["y_score"] >= threshold).astype(int)
    metrics = pd.DataFrame(
        [
            {
                "model": "seed_ensemble",
                "ensemble_size": len(frames),
                **binary_classification_metrics(
                    ensemble["y_true"].to_numpy(dtype=int),
                    ensemble["y_score"].to_numpy(dtype=float),
                ),
            }
        ]
    )
    return aggregate_patient_probabilities(ensemble, threshold=threshold), metrics


def write_feature_ssl_transfer_outputs(
    *,
    output_root: str | Path,
    run_name: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    ssl_history: pd.DataFrame,
    supervised_loss_history: pd.DataFrame,
) -> dict[str, Path]:
    root = Path(output_root)
    safe = _safe_run_name(run_name)
    paths = {
        "ssl_history": root / "results" / "ssl" / f"feature_ssl_pretraining_history_{safe}.csv",
        "predictions": root / "results" / "predictions" / f"dl_loso_predictions_{safe}.csv",
        "metrics": root / "results" / "metrics" / f"dl_model_comparison_{safe}.csv",
        "loss_history": root / "results" / "training_logs" / f"dl_loss_history_{safe}.csv",
        "loss_curve": root / "results" / "figures" / f"dl_loss_curve_{safe}.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    ssl_history.to_csv(paths["ssl_history"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    supervised_loss_history.to_csv(paths["loss_history"], index=False)
    _plot_loss_history(supervised_loss_history, paths["loss_curve"])
    return paths


def feature_ssl_transfer_run_name(
    *,
    data_scope: str,
    feature_kind: str,
    fusion: str,
    encoder_kind: str,
    transfer_mode: str,
    seed: str | int,
    ssl_objective: str = "ntxent",
    pretrain_epochs: int,
    temperature: float,
    noise_std: float,
    feature_mask_prob: float,
    pretrain_batch_size: int,
    supervised_epochs: int,
    embedding_dim: int = 32,
    projection_dim: int = 16,
    dropout: float = 0.0,
    pretrain_lr: float = 1e-3,
    supervised_lr: float = 1e-3,
    encoder_lr_multiplier: float = 1.0,
    transfer_head_lr_multiplier: float = 1.0,
    freeze_pretrained_encoder_epochs: int = 0,
    ssl_bridge_enabled: bool = False,
    ssl_consistency_weight: float = 0.0,
    ssl_two_head_enabled: bool = False,
    ssl_aux_head_weight: float = 0.5,
    ssl_fusion_weight: float = 0.5,
    ssl_residual_enabled: bool = False,
    ssl_residual_weight: float = 0.14,
    bridge_residual_weight: float = 0.04,
    ssl_residual_random_main: bool = False,
    ssl_residual_main_loss_only: bool = False,
    ssl_residual_preserve_main: bool = False,
    ssl_residual_select_weights_on_val: bool = False,
    ssl_residual_logit_preservation_weight: float = 0.0,
    ssl_residual_logit_delta_enabled: bool = False,
    ssl_residual_logit_delta_scale: float = 0.5,
    embedding_adapter_dim: int = 0,
    embedding_adapter_scale: float = 1.0,
    latent_modality_mask_prob: float = 0.0,
) -> str:
    objective_token = "" if ssl_objective == "ntxent" else f"_{ssl_objective}"
    tuning_tokens = []
    if embedding_dim != 32:
        tuning_tokens.append(f"emb{embedding_dim}")
    if projection_dim != 16:
        tuning_tokens.append(f"proj{projection_dim}")
    if dropout != 0:
        tuning_tokens.append(f"drop{_number_token(dropout)}")
    if pretrain_lr != 1e-3:
        tuning_tokens.append(f"plr{_number_token(pretrain_lr)}")
    if supervised_lr != 1e-3:
        tuning_tokens.append(f"slr{_number_token(supervised_lr)}")
    if encoder_lr_multiplier != 1.0:
        tuning_tokens.append(f"encLRx{_number_token(encoder_lr_multiplier)}")
    if transfer_head_lr_multiplier != 1.0:
        tuning_tokens.append(f"headLRx{_number_token(transfer_head_lr_multiplier)}")
    if freeze_pretrained_encoder_epochs != 0:
        tuning_tokens.append(f"frzenc{freeze_pretrained_encoder_epochs}")
    if ssl_bridge_enabled:
        tuning_tokens.append("sslbridge")
    if ssl_consistency_weight != 0:
        tuning_tokens.append(f"sslcons{_number_token(ssl_consistency_weight)}")
    if ssl_two_head_enabled:
        tuning_tokens.append("ssltwohead")
    if ssl_residual_enabled:
        tuning_tokens.append("sslresidual")
    if ssl_residual_enabled and ssl_residual_random_main:
        tuning_tokens.append("randommain")
    if ssl_residual_enabled and ssl_residual_main_loss_only:
        tuning_tokens.append("mainloss")
    if ssl_residual_enabled and ssl_residual_preserve_main:
        tuning_tokens.append("preservemain")
    if ssl_residual_enabled and ssl_residual_select_weights_on_val:
        tuning_tokens.append("valtune")
    if ssl_residual_enabled and ssl_residual_logit_preservation_weight != 0:
        tuning_tokens.append(f"logitpres{_number_token(ssl_residual_logit_preservation_weight)}")
    if ssl_residual_enabled and ssl_residual_logit_delta_enabled:
        tuning_tokens.append(f"logitdelta{_number_token(ssl_residual_logit_delta_scale)}")
    if ssl_aux_head_weight != 0.5:
        tuning_tokens.append(f"sslaux{_number_token(ssl_aux_head_weight)}")
    if ssl_fusion_weight != 0.5:
        tuning_tokens.append(f"sslfuse{_number_token(ssl_fusion_weight)}")
    if ssl_residual_enabled and ssl_residual_weight != 0.14:
        tuning_tokens.append(f"sslres{_number_token(ssl_residual_weight)}")
    if ssl_residual_enabled and bridge_residual_weight != 0.04:
        tuning_tokens.append(f"bridgeres{_number_token(bridge_residual_weight)}")
    if embedding_adapter_dim != 0:
        tuning_tokens.append(f"adapter{embedding_adapter_dim}")
    if embedding_adapter_scale != 1.0:
        tuning_tokens.append(f"adapterscale{_number_token(embedding_adapter_scale)}")
    if latent_modality_mask_prob != 0:
        tuning_tokens.append(f"latentmask{_number_token(latent_modality_mask_prob)}")
    tuning_token = "" if not tuning_tokens else "_" + "_".join(tuning_tokens)
    return (
        f"feature_ssl_{data_scope}_{feature_kind}_{fusion}_{encoder_kind}"
        f"{objective_token}_{transfer_mode}_seed{seed}_pre{pretrain_epochs}_temp{_number_token(temperature)}"
        f"_noise{_number_token(noise_std)}_mask{_number_token(feature_mask_prob)}"
        f"_bs{pretrain_batch_size}_sup{supervised_epochs}{tuning_token}"
    )


def _pair_in_scope(pair: FeatureSSLPairRecord, scope: str) -> bool:
    if scope == "supervised-baseline":
        return pair.group == "patient" and pair.stage == "基线" and pair.is_supervised_subject
    if scope == "all-patient-baseline":
        return pair.group == "patient" and pair.stage == "基线"
    if scope == "all-patient":
        return pair.group == "patient"
    if scope == "all-patient-health":
        return pair.group in {"patient", "health"}
    if scope == "health-only":
        return pair.group == "health"
    raise ValueError(f"Unsupported feature SSL data scope: {scope}")


def _pair_to_supervised_record(pair: FeatureSSLPairRecord) -> SupervisedFeatureRecord:
    first_branch = next(iter(pair.modalities))
    eo, ec = pair.modalities[first_branch]
    return SupervisedFeatureRecord(
        subject_id=pair.subject_id,
        label=0,
        eo=eo,
        ec=ec,
        modalities=pair.modalities,
    )


def _augment_feature_batch(
    batch: dict[str, torch.Tensor],
    *,
    noise_std: float,
    feature_mask_prob: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    augmented: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "y":
            continue
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed + _stable_key_offset(key))
        view = value
        if noise_std > 0:
            view = view + torch.randn(value.shape, generator=generator, device=value.device) * noise_std
        if feature_mask_prob > 0:
            mask = torch.rand(value.shape, generator=generator, device=value.device) < feature_mask_prob
            view = view.masked_fill(mask, 0.0)
        augmented[key] = view
    return augmented


def _apply_latent_modality_mask(
    branch_embeddings: Mapping[str, torch.Tensor],
    *,
    branches: tuple[str, ...],
    mask_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not branches:
        raise ValueError("Latent modality masking requires at least one branch.")
    if not 0 <= mask_prob <= 1:
        raise ValueError("mask_prob must be between 0 and 1.")
    missing = [branch for branch in branches if branch not in branch_embeddings]
    if missing:
        raise KeyError(f"Missing branch embedding(s): {', '.join(missing)}")
    first = branch_embeddings[branches[0]]
    batch_size = first.shape[0]
    device = first.device
    if len(branches) == 1 or mask_prob == 0:
        return torch.cat([branch_embeddings[branch] for branch in branches], dim=1)

    keep = torch.rand(
        (batch_size, len(branches)),
        generator=generator,
        device=device,
    ) >= mask_prob
    dropped_all = ~keep.any(dim=1)
    if dropped_all.any():
        rescue = torch.randint(
            len(branches),
            (int(dropped_all.sum().item()),),
            generator=generator,
            device=device,
        )
        keep[dropped_all] = False
        keep[dropped_all, rescue] = True

    masked: list[torch.Tensor] = []
    for branch_index, branch in enumerate(branches):
        embedding = branch_embeddings[branch]
        if embedding.shape[0] != batch_size:
            raise ValueError("All branch embeddings must have the same batch size.")
        branch_keep = keep[:, branch_index].to(dtype=embedding.dtype).unsqueeze(1)
        masked.append(embedding * branch_keep)
    return torch.cat(masked, dim=1)


def _feature_reconstruction_output_dim(
    record: SupervisedFeatureRecord,
    branches: tuple[str, ...],
) -> int:
    total = 0
    for branch in branches:
        if branch not in record.modalities:
            raise KeyError(f"Missing modality {branch!r} for masked reconstruction.")
        eo, ec = record.modalities[branch]
        total += int(np.prod(eo.shape))
        total += int(np.prod(ec.shape))
    return total


def _feature_masked_reconstruction_components(
    model: MultimodalEEGModel,
    reconstruction_head: nn.Module,
    batch: dict[str, torch.Tensor],
    config: FeatureSSLTrainingConfig,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    corrupted_batch, target, reconstruction_mask = _masked_reconstruction_batch(
        batch,
        branches=model.branches,
        noise_std=config.noise_std,
        feature_mask_prob=config.feature_mask_prob,
        seed=seed,
    )
    embedding = model.extract_embedding(corrupted_batch)
    reconstructed = reconstruction_head(embedding)
    reconstruction_loss = _masked_reconstruction_loss(reconstructed, target, reconstruction_mask)
    zero = torch.zeros((), device=target.device)
    return {
        "loss": reconstruction_loss,
        "invariance": zero,
        "variance": zero,
        "covariance": zero,
        "on_diag": zero,
        "off_diag": zero,
        "byol": zero,
        "reconstruction": reconstruction_loss,
    }


def _feature_masked_reconstruction_vicreg_components(
    model: MultimodalEEGModel,
    projection_head: nn.Module,
    reconstruction_head: nn.Module,
    batch: dict[str, torch.Tensor],
    config: FeatureSSLTrainingConfig,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    corrupted_a, target_a, mask_a = _masked_reconstruction_batch(
        batch,
        branches=model.branches,
        noise_std=config.noise_std,
        feature_mask_prob=config.feature_mask_prob,
        seed=seed,
    )
    corrupted_b, target_b, mask_b = _masked_reconstruction_batch(
        batch,
        branches=model.branches,
        noise_std=config.noise_std,
        feature_mask_prob=config.feature_mask_prob,
        seed=seed + 1,
    )
    embedding_a = model.extract_embedding(corrupted_a)
    embedding_b = model.extract_embedding(corrupted_b)
    reconstruction = 0.5 * (
        _masked_reconstruction_loss(reconstruction_head(embedding_a), target_a, mask_a)
        + _masked_reconstruction_loss(reconstruction_head(embedding_b), target_b, mask_b)
    )
    vicreg = _feature_vicreg_loss(projection_head(embedding_a), projection_head(embedding_b), config)
    loss = reconstruction + vicreg["loss"]
    zero = torch.zeros((), device=loss.device)
    return {
        "loss": loss,
        "invariance": vicreg["invariance"],
        "variance": vicreg["variance"],
        "covariance": vicreg["covariance"],
        "on_diag": zero,
        "off_diag": zero,
        "byol": zero,
        "reconstruction": reconstruction,
    }


def _masked_reconstruction_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    reconstruction_mask: torch.Tensor,
) -> torch.Tensor:
    if reconstructed.shape != target.shape:
        raise ValueError("Reconstruction output shape does not match flattened feature target.")
    if reconstruction_mask.any():
        return F.mse_loss(reconstructed[reconstruction_mask], target[reconstruction_mask])
    return F.mse_loss(reconstructed, target)


def _masked_reconstruction_batch(
    batch: dict[str, torch.Tensor],
    *,
    branches: tuple[str, ...],
    noise_std: float,
    feature_mask_prob: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    corrupted: dict[str, torch.Tensor] = {}
    target_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    for branch in branches:
        for state in ("eo", "ec"):
            key = f"{branch}_{state}"
            if key not in batch:
                raise KeyError(f"Masked reconstruction batch is missing key {key!r}.")
            value = batch[key].float()
            generator = torch.Generator(device=value.device)
            generator.manual_seed(seed + _stable_key_offset(key))
            mask = torch.rand(value.shape, generator=generator, device=value.device) < feature_mask_prob
            view = value.masked_fill(mask, 0.0)
            if noise_std > 0:
                view = view + torch.randn(value.shape, generator=generator, device=value.device) * noise_std
            corrupted[key] = view
            target_parts.append(value.flatten(start_dim=1))
            mask_parts.append(mask.flatten(start_dim=1))
    return corrupted, torch.cat(target_parts, dim=1), torch.cat(mask_parts, dim=1)


def _load_cached_supervised_baseline_features(
    config: PathConfig,
    record: EEGFileRecord,
    branches: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    if not (
        record.group == "patient"
        and record.stage == "基线"
        and record.is_supervised_subject
    ):
        return None
    try:
        return {
            branch: _load_branch_array(config.output_root, record.subject_id, record.state, branch)
            for branch in branches
        }
    except (FileNotFoundError, KeyError):
        return None


def _affected_hand_for_record(record: EEGFileRecord, affected_by_subject: Mapping[str, str]) -> str:
    if record.group == "health":
        return "右"
    affected = affected_by_subject.get(record.subject_id)
    if affected not in {"右", "左"}:
        raise ValueError(f"Missing valid affected hand for patient {record.subject_id}.")
    return affected


def _validate_feature_ssl_config(config: FeatureSSLTrainingConfig) -> FeatureSSLTrainingConfig:
    normalize_ssl_data_scope(config.data_scope)
    branches_for_feature_kind(config.feature_kind)
    if config.fusion not in {"concat", "gated"}:
        raise ValueError("fusion must be 'concat' or 'gated'.")
    if config.encoder_kind not in {"cnn", "linear", "gncnn", "rescnn"}:
        raise ValueError("encoder_kind must be 'cnn', 'linear', 'gncnn', or 'rescnn'.")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.embedding_dim < 1:
        raise ValueError("embedding_dim must be positive.")
    if config.projection_dim < 1:
        raise ValueError("projection_dim must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if config.ssl_objective not in FEATURE_SSL_OBJECTIVES:
        raise ValueError(f"ssl_objective must be one of {', '.join(sorted(FEATURE_SSL_OBJECTIVES))}.")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if not 0 <= config.feature_mask_prob <= 1:
        raise ValueError("feature_mask_prob must be between 0 and 1.")
    if config.vicreg_variance_target <= 0 or config.vicreg_eps <= 0:
        raise ValueError("vicreg_variance_target and vicreg_eps must be positive.")
    if config.barlow_offdiag_weight < 0 or config.barlow_eps <= 0:
        raise ValueError("barlow_offdiag_weight must be non-negative and barlow_eps must be positive.")
    if config.branch_barlow_weight < 0:
        raise ValueError("branch_barlow_weight must be non-negative.")
    if not 0 <= config.latent_modality_mask_prob <= 1:
        raise ValueError("latent_modality_mask_prob must be between 0 and 1.")
    if not 0 <= config.byol_momentum < 1:
        raise ValueError("byol_momentum must be in [0, 1).")
    if config.byol_predictor_hidden_dim <= 0:
        raise ValueError("byol_predictor_hidden_dim must be positive.")
    return config


def _stable_key_offset(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def _safe_run_name(run_name: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in run_name)
    safe = safe.strip("_-")
    if not safe:
        raise ValueError("run_name must contain at least one filename-safe character.")
    if len(safe) > 120:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]
        prefix = safe[: 120 - len(digest) - 1].rstrip("_-")
        safe = f"{prefix}_{digest}" if prefix else digest
    return safe


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


def _feature_alignment_components(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    config: FeatureSSLTrainingConfig,
    ntxent_loss_fn: NTXentLoss,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=projection_a.device)
    if config.ssl_objective == "ntxent":
        loss = ntxent_loss_fn(projection_a, projection_b)
        return {
            "loss": loss,
            "invariance": zero,
            "variance": zero,
            "covariance": zero,
            "on_diag": zero,
            "off_diag": zero,
            "byol": zero,
        }
    if config.ssl_objective in {"vicreg", "masked-vicreg"}:
        components = _feature_vicreg_loss(projection_a, projection_b, config)
        return {**components, "on_diag": zero, "off_diag": zero, "byol": zero}
    if config.ssl_objective in {"barlow", "masked-barlow"}:
        components = _feature_barlow_twins_loss(projection_a, projection_b, config)
        return {
            "loss": components["loss"],
            "invariance": zero,
            "variance": zero,
            "covariance": zero,
            "on_diag": components["on_diag"],
            "off_diag": components["off_diag"],
            "byol": zero,
        }
    if config.ssl_objective == "branch-barlow":
        raise ValueError("Branch-aware Barlow requires branch projections.")
    raise ValueError(f"Unsupported feature SSL objective: {config.ssl_objective}")


def _feature_vicreg_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    config: FeatureSSLTrainingConfig,
) -> dict[str, torch.Tensor]:
    if projection_a.shape != projection_b.shape:
        raise ValueError("Projection shapes must match for VICReg.")
    if projection_a.ndim != 2:
        raise ValueError("VICReg projections must be two-dimensional.")
    if projection_a.shape[0] < 2:
        zero = torch.zeros((), device=projection_a.device)
        return {"loss": zero, "invariance": zero, "variance": zero, "covariance": zero}
    projection_a = projection_a.float()
    projection_b = projection_b.float()
    invariance = F.mse_loss(projection_a, projection_b)
    variance = 0.5 * (
        F.relu(config.vicreg_variance_target - torch.sqrt(projection_a.var(dim=0, unbiased=False) + config.vicreg_eps)).mean()
        + F.relu(config.vicreg_variance_target - torch.sqrt(projection_b.var(dim=0, unbiased=False) + config.vicreg_eps)).mean()
    )
    covariance = 0.5 * (
        _feature_off_diagonal_covariance_loss(projection_a)
        + _feature_off_diagonal_covariance_loss(projection_b)
    )
    loss = (
        config.vicreg_invariance_weight * invariance
        + config.vicreg_variance_weight * variance
        + config.vicreg_covariance_weight * covariance
    )
    return {"loss": loss, "invariance": invariance, "variance": variance, "covariance": covariance}


def _feature_off_diagonal_covariance_loss(projection: torch.Tensor) -> torch.Tensor:
    batch_size, feature_dim = projection.shape
    centered = projection - projection.mean(dim=0)
    covariance = centered.T @ centered / max(batch_size - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.pow(2).sum() / feature_dim


def _feature_barlow_twins_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    config: FeatureSSLTrainingConfig,
) -> dict[str, torch.Tensor]:
    if projection_a.shape != projection_b.shape:
        raise ValueError("Projection shapes must match for Barlow Twins.")
    if projection_a.ndim != 2:
        raise ValueError("Barlow Twins projections must be two-dimensional.")
    if projection_a.shape[0] < 2:
        zero = torch.zeros((), device=projection_a.device)
        return {"loss": zero, "on_diag": zero, "off_diag": zero}
    projection_a = _standardize_feature_projection(projection_a.float(), config.barlow_eps)
    projection_b = _standardize_feature_projection(projection_b.float(), config.barlow_eps)
    cross_correlation = projection_a.T @ projection_b / projection_a.shape[0]
    on_diag = torch.diagonal(cross_correlation).add(-1).pow(2).sum()
    off_diag = (cross_correlation - torch.diag(torch.diagonal(cross_correlation))).pow(2).sum()
    loss = on_diag + config.barlow_offdiag_weight * off_diag
    return {"loss": loss, "on_diag": on_diag, "off_diag": off_diag}


def _feature_branch_barlow_twins_components(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    branch_projection_a: Mapping[str, torch.Tensor],
    branch_projection_b: Mapping[str, torch.Tensor],
    config: FeatureSSLTrainingConfig,
) -> dict[str, torch.Tensor]:
    global_components = _feature_barlow_twins_loss(projection_a, projection_b, config)
    if set(branch_projection_a) != set(branch_projection_b):
        raise ValueError("Branch projection keys must match for branch-aware Barlow.")
    if not branch_projection_a:
        raise ValueError("Branch-aware Barlow requires at least one branch projection.")
    branch_losses = [
        _feature_barlow_twins_loss(branch_projection_a[branch], branch_projection_b[branch], config)["loss"]
        for branch in sorted(branch_projection_a)
    ]
    branch_barlow = torch.stack(branch_losses).mean()
    loss = global_components["loss"] + config.branch_barlow_weight * branch_barlow
    zero = torch.zeros((), device=projection_a.device)
    return {
        "loss": loss,
        "invariance": zero,
        "variance": zero,
        "covariance": zero,
        "on_diag": global_components["on_diag"],
        "off_diag": global_components["off_diag"],
        "global_barlow": global_components["loss"],
        "branch_barlow": branch_barlow,
        "byol": zero,
    }


def _standardize_feature_projection(projection: torch.Tensor, eps: float) -> torch.Tensor:
    return (projection - projection.mean(dim=0)) / (projection.std(dim=0, unbiased=False) + eps)


def _feature_byol_components(
    model: MultimodalEEGModel,
    projection_head: nn.Module,
    predictor_head: nn.Module | None,
    target_model: MultimodalEEGModel | None,
    target_projection_head: nn.Module | None,
    view_a: dict[str, torch.Tensor],
    view_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if predictor_head is None or target_model is None or target_projection_head is None:
        raise ValueError("BYOL objective requires online predictor and target networks.")
    prediction_a = predictor_head(projection_head(model.extract_embedding(view_a)))
    prediction_b = predictor_head(projection_head(model.extract_embedding(view_b)))
    with torch.no_grad():
        target_a = target_projection_head(target_model.extract_embedding(view_a))
        target_b = target_projection_head(target_model.extract_embedding(view_b))
    loss = 0.5 * (
        _feature_negative_cosine_similarity(prediction_a, target_b.detach())
        + _feature_negative_cosine_similarity(prediction_b, target_a.detach())
    )
    zero = torch.zeros((), device=prediction_a.device)
    return {
        "loss": loss,
        "invariance": zero,
        "variance": zero,
        "covariance": zero,
        "on_diag": zero,
        "off_diag": zero,
        "byol": loss,
    }


def _feature_simsiam_components(
    model: MultimodalEEGModel,
    projection_head: nn.Module,
    predictor_head: nn.Module | None,
    view_a: dict[str, torch.Tensor],
    view_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if predictor_head is None:
        raise ValueError("SimSiam objective requires an online predictor.")
    projection_a = projection_head(model.extract_embedding(view_a))
    projection_b = projection_head(model.extract_embedding(view_b))
    prediction_a = predictor_head(projection_a)
    prediction_b = predictor_head(projection_b)
    loss = 0.5 * (
        _feature_negative_cosine_similarity(prediction_a, projection_b.detach())
        + _feature_negative_cosine_similarity(prediction_b, projection_a.detach())
    )
    zero = torch.zeros((), device=prediction_a.device)
    return {
        "loss": loss,
        "invariance": zero,
        "variance": zero,
        "covariance": zero,
        "on_diag": zero,
        "off_diag": zero,
        "byol": loss,
    }


def _build_byol_predictor(config: FeatureSSLTrainingConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.projection_dim, config.byol_predictor_hidden_dim),
        nn.ReLU(),
        nn.Linear(config.byol_predictor_hidden_dim, config.projection_dim),
    )


def _feature_negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    return 2 - 2 * (prediction * target).sum(dim=1).mean()


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


@torch.no_grad()
def _update_target_module(target_module: nn.Module, online_module: nn.Module, momentum: float) -> None:
    for target_parameter, online_parameter in zip(target_module.parameters(), online_module.parameters(), strict=True):
        target_parameter.data.mul_(momentum).add_(online_parameter.data, alpha=1.0 - momentum)
    for target_buffer, online_buffer in zip(target_module.buffers(), online_module.buffers(), strict=True):
        target_buffer.data.copy_(online_buffer.data)


def _feature_ssl_objective_label(ssl_objective: str) -> str:
    return "feature_contrastive" if ssl_objective == "ntxent" else f"feature_{ssl_objective.replace('-', '_')}"
