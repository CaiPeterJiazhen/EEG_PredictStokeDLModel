from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.features.connectivity import build_edge_list
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.multimodal_model import MultimodalEEGModel, branches_for_feature_kind
from eeg_recovery.models.ssl_model import NTXentLoss
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.optimizers import build_swa_model, update_swa_batch_norm
from eeg_recovery.training.residual_aware_losses import residual_aware_multitask_loss
from eeg_recovery.training.residual_targets import (
    compute_signed_distance_from_label_table,
    fold_local_standardize_signed_distance,
)
from eeg_recovery.training.schedulers import EarlyStopping, build_reduce_on_plateau
from eeg_recovery.training.train_feature_ssl import (
    FeatureSSLPairRecord,
    FeatureSSLTrainingConfig,
    build_feature_ssl_pair_records_from_feature_records,
    compute_feature_records_for_eeg_records,
    feature_ssl_pairs_for_scope,
    normalize_ssl_data_scope,
    _augment_feature_batch,
    _feature_alignment_components,
    _feature_ssl_objective_label,
    _pair_to_supervised_record,
    _seed_feature_ssl_training,
)
from eeg_recovery.training.train_ssl import select_ssl_records
from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    _fit_state_scaler,
    _make_batch,
    _train_validation_subjects,
    load_supervised_feature_records,
    resolve_device,
)


SEEDS_10 = (0, 1, 2, 3, 4, 5, 7, 13, 21, 42)
MOTOR_CHANNELS = (
    "C3",
    "C4",
    "C1",
    "C2",
    "C5",
    "C6",
    "FC3",
    "FC4",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "CP3",
    "CP4",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
)
PSD_BANDS = (
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta Low", 13.0, 18.0),
    ("Beta Medium", 18.0, 21.0),
    ("Beta High", 21.0, 30.0),
    ("Gamma", 30.0, 45.5),
)
MAIN_ABLATIONS = (
    "full_psd_wpli",
    "psd_only",
    "wpli_only",
    "eo_only",
    "ec_only",
    "beta_medium_beta_high",
    "full_minus_beta_medium_beta_high",
    "motor_wpli_edges_only",
    "full_minus_motor_wpli_edges",
)
METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    family: str
    input_description: str
    keep_psd: bool = True
    keep_wpli: bool = True
    states: tuple[str, ...] = ("EO", "EC")
    only_bands: tuple[str, ...] = ()
    drop_bands: tuple[str, ...] = ()
    psd_only_bands: tuple[str, ...] = ()
    psd_drop_bands: tuple[str, ...] = ()
    wpli_only_bands: tuple[str, ...] = ()
    wpli_drop_bands: tuple[str, ...] = ()
    wpli_edge_mode: str = "all"
    table_group: str = "supplementary"


@dataclass(frozen=True)
class FeatureMetadata:
    channel_names: tuple[str, ...]
    frequency_bins: np.ndarray
    edge_list: tuple[tuple[str, str], ...]
    wpli_band_names: tuple[str, ...]


@dataclass(frozen=True)
class AblationMask:
    psd_eo: np.ndarray
    psd_ec: np.ndarray
    wpli_eo: np.ndarray
    wpli_ec: np.ndarray
    n_active_psd_values: int
    n_active_wpli_values: int
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final residual-aware SSL-CNN feature/state/band ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--output-root", default=RUN_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--analysis-mode",
        choices=("independent_train", "fixed_final_checkpoint_masked_inference"),
        default="independent_train",
        help=(
            "independent_train retrains SSL + supervised heads per ablation. "
            "fixed_final_checkpoint_masked_inference is a compute-limited approximate analysis "
            "using existing final SWA checkpoints with standardized-space masks."
        ),
    )
    parser.add_argument(
        "--final-checkpoint-dir",
        default=PROJECT_ROOT / "results" / "checkpoints" / "supervised" / "residualaware_highrank_swa_clsalpha1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_10))
    parser.add_argument("--ablation-set", choices=("main", "supplementary", "all"), default="all")
    parser.add_argument("--ablation-names", nargs="+", default=None)
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--ssl-data-scope", default="supervised-baseline")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--feature-mask-prob", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--swa-start-epoch", type=int, default=50)
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    parser.add_argument("--skip-existing-predictions", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source_config = load_path_config(args.source_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs(output_root)

    audit_old_logistic_outputs(source_config.output_root, output_root)
    if args.audit_only:
        print(f"Wrote audit outputs under {output_root}")
        return

    labels = load_supervised_label_table(source_config)
    supervised_ids = [normalize_subject_id(value) for value in labels["subject_id"].tolist()]
    supervised_records = load_supervised_feature_records(source_config, labels, feature_kind="psd-fc-wpli")
    targets = compute_signed_distance_from_label_table(labels, threshold=1.5)
    metadata = load_feature_metadata(source_config.output_root, supervised_ids[0])
    specs = select_ablation_specs(args.ablation_set, args.ablation_names)
    masks = {spec.name: build_ablation_mask(spec, metadata) for spec in specs}
    device = resolve_device(args.device)

    if args.analysis_mode == "fixed_final_checkpoint_masked_inference":
        predictions, seed_metrics = run_fixed_final_checkpoint_masked_inference(
            specs=specs,
            masks=masks,
            records=supervised_records,
            checkpoint_dir=Path(args.final_checkpoint_dir),
            seeds=[int(seed) for seed in args.seeds],
            device=device,
            limit_folds=args.limit_folds,
        )
        outputs = write_incremental_outputs(
            output_root,
            [predictions],
            [seed_metrics],
            [],
            [],
            specs,
            masks,
        )
        print("Wrote compute-limited fixed-final-checkpoint masked-inference outputs:")
        for path in outputs:
            print(path)
        return

    eeg_records = build_eeg_file_index(
        source_config.patient_eeg_root,
        source_config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )
    eeg_records = select_ssl_records(eeg_records, data_scope=args.ssl_data_scope)
    feature_records = compute_feature_records_for_eeg_records(
        source_config,
        eeg_records,
        feature_kind="psd-fc-wpli",
    )
    all_ssl_pairs = build_feature_ssl_pair_records_from_feature_records(eeg_records, feature_records)

    prediction_frames: list[pd.DataFrame] = []
    seed_metric_frames: list[pd.DataFrame] = []
    ssl_history_frames: list[pd.DataFrame] = []
    loss_history_frames: list[pd.DataFrame] = []

    pred_path = output_root / "results" / "predictions" / "final_ssl_cnn_feature_state_band_ablation_predictions.csv"
    existing_predictions = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    completed = set()
    if args.skip_existing_predictions and not existing_predictions.empty:
        completed = {
            (str(row.ablation_name), int(row.seed))
            for row in existing_predictions[["ablation_name", "seed"]].drop_duplicates().itertuples(index=False)
        }
        prediction_frames.append(existing_predictions)

    for spec in specs:
        mask = masks[spec.name]
        for seed in args.seeds:
            if (spec.name, int(seed)) in completed:
                continue
            predictions, seed_metrics, ssl_history, loss_history = run_one_ablation_seed(
                spec=spec,
                mask=mask,
                records=supervised_records,
                labels=labels,
                targets=targets,
                all_ssl_pairs=all_ssl_pairs,
                ssl_data_scope=args.ssl_data_scope,
                seed=int(seed),
                device=device,
                limit_folds=args.limit_folds,
                pretrain_epochs=args.pretrain_epochs,
                pretrain_batch_size=args.pretrain_batch_size,
                pretrain_lr=args.pretrain_lr,
                projection_dim=args.projection_dim,
                temperature=args.temperature,
                noise_std=args.noise_std,
                feature_mask_prob=args.feature_mask_prob,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                rank_margin=args.rank_margin,
                swa_start_epoch=args.swa_start_epoch,
                swa_lr=args.swa_lr,
            )
            prediction_frames.append(predictions)
            seed_metric_frames.append(seed_metrics)
            ssl_history_frames.append(ssl_history)
            loss_history_frames.append(loss_history)
            write_incremental_outputs(
                output_root,
                prediction_frames,
                seed_metric_frames,
                ssl_history_frames,
                loss_history_frames,
                specs,
                masks,
            )

    outputs = write_incremental_outputs(
        output_root,
        prediction_frames,
        seed_metric_frames,
        ssl_history_frames,
        loss_history_frames,
        specs,
        masks,
    )
    print("Wrote final SSL-CNN ablation outputs:")
    for path in outputs:
        print(path)


def ensure_output_dirs(output_root: Path) -> None:
    for relative in (
        "docs",
        "results/predictions",
        "results/metrics",
        "results/tables",
        "results/figures/revised_initial",
        "results/ssl",
        "results/loss_history",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def ablation_specs() -> list[AblationSpec]:
    main = "main"
    supp = "supplementary"
    return [
        AblationSpec("full_psd_wpli", "full", "PSD + WPLI, EO + EC, all allowed bands", table_group=main),
        AblationSpec("psd_only", "modality", "PSD only, EO + EC; WPLI zeroed", keep_wpli=False, table_group=main),
        AblationSpec("wpli_only", "modality", "WPLI only, EO + EC; PSD zeroed", keep_psd=False, table_group=main),
        AblationSpec("eo_only", "state", "EO PSD + WPLI only; EC zeroed", states=("EO",), table_group=main),
        AblationSpec("ec_only", "state", "EC PSD + WPLI only; EO zeroed", states=("EC",), table_group=main),
        AblationSpec(
            "beta_medium_beta_high",
            "band",
            "PSD/WPLI Beta Medium + Beta High only",
            only_bands=("Beta Medium", "Beta High"),
            table_group=main,
        ),
        AblationSpec(
            "full_minus_beta_medium_beta_high",
            "leave_band_out",
            "Full PSD + WPLI minus Beta Medium and Beta High",
            drop_bands=("Beta Medium", "Beta High"),
            table_group=main,
        ),
        AblationSpec(
            "motor_wpli_edges_only",
            "motor_network",
            "Motor-related WPLI edges only; PSD zeroed",
            keep_psd=False,
            wpli_edge_mode="motor",
            table_group=main,
        ),
        AblationSpec(
            "full_minus_motor_wpli_edges",
            "motor_network",
            "Full PSD + WPLI minus motor-related WPLI edges",
            wpli_edge_mode="non_motor",
            table_group=main,
        ),
        AblationSpec("psd_eo_only", "state_modality", "PSD EO only", keep_wpli=False, states=("EO",), table_group=supp),
        AblationSpec("psd_ec_only", "state_modality", "PSD EC only", keep_wpli=False, states=("EC",), table_group=supp),
        AblationSpec("wpli_eo_only", "state_modality", "WPLI EO only", keep_psd=False, states=("EO",), table_group=supp),
        AblationSpec("wpli_ec_only", "state_modality", "WPLI EC only", keep_psd=False, states=("EC",), table_group=supp),
        AblationSpec("delta_only", "band", "PSD/WPLI Delta only", only_bands=("Delta",), table_group=supp),
        AblationSpec("theta_only", "band", "PSD/WPLI Theta only", only_bands=("Theta",), table_group=supp),
        AblationSpec("alpha_only", "band", "PSD/WPLI Alpha only", only_bands=("Alpha",), table_group=supp),
        AblationSpec("beta_low_only", "band", "PSD/WPLI Beta Low only", only_bands=("Beta Low",), table_group=supp),
        AblationSpec("beta_medium_only", "band", "PSD/WPLI Beta Medium only", only_bands=("Beta Medium",), table_group=supp),
        AblationSpec("beta_high_only", "band", "PSD/WPLI Beta High only", only_bands=("Beta High",), table_group=supp),
        AblationSpec("psd_gamma_only", "band", "PSD Gamma only; WPLI zeroed", keep_wpli=False, psd_only_bands=("Gamma",), table_group=supp),
        AblationSpec("full_minus_delta", "leave_band_out", "Full PSD + WPLI minus Delta", drop_bands=("Delta",), table_group=supp),
        AblationSpec("full_minus_theta", "leave_band_out", "Full PSD + WPLI minus Theta", drop_bands=("Theta",), table_group=supp),
        AblationSpec("full_minus_alpha", "leave_band_out", "Full PSD + WPLI minus Alpha", drop_bands=("Alpha",), table_group=supp),
        AblationSpec("full_minus_beta_low", "leave_band_out", "Full PSD + WPLI minus Beta Low", drop_bands=("Beta Low",), table_group=supp),
        AblationSpec("full_minus_beta_medium", "leave_band_out", "Full PSD + WPLI minus Beta Medium", drop_bands=("Beta Medium",), table_group=supp),
        AblationSpec("full_minus_beta_high", "leave_band_out", "Full PSD + WPLI minus Beta High", drop_bands=("Beta High",), table_group=supp),
        AblationSpec("full_minus_psd_gamma", "leave_band_out", "Full input minus PSD Gamma", psd_drop_bands=("Gamma",), table_group=supp),
        AblationSpec("non_motor_wpli_edges_only", "motor_network", "Non-motor WPLI edges only; PSD zeroed", keep_psd=False, wpli_edge_mode="non_motor", table_group=supp),
        AblationSpec("full_minus_non_motor_wpli_edges", "motor_network", "Full PSD + WPLI minus non-motor WPLI edges", wpli_edge_mode="motor", table_group=supp),
    ]


def select_ablation_specs(ablation_set: str, ablation_names: list[str] | None) -> list[AblationSpec]:
    specs = ablation_specs()
    if ablation_set == "main":
        specs = [spec for spec in specs if spec.table_group == "main"]
    elif ablation_set == "supplementary":
        specs = [spec for spec in specs if spec.table_group == "supplementary"]
    if ablation_names:
        requested = set(ablation_names)
        known = {spec.name for spec in specs}
        missing = sorted(requested - {spec.name for spec in ablation_specs()})
        if missing:
            raise ValueError(f"Unknown ablation name(s): {', '.join(missing)}")
        specs = [spec for spec in ablation_specs() if spec.name in requested]
    return specs


def load_feature_metadata(source_output_root: Path, example_subject_id: str) -> FeatureMetadata:
    subject_id = normalize_subject_id(example_subject_id)
    psd_path = source_output_root / "data" / "features" / "psd" / f"{subject_id}_EO_psd.npz"
    fc_path = source_output_root / "data" / "features" / "fc" / f"{subject_id}_EO_fc.npz"
    with np.load(psd_path, allow_pickle=False) as payload:
        frequency_bins = np.asarray(payload["frequency_bins"], dtype=float)
        channel_names = tuple(str(value) for value in payload["channel_names_after_alignment"].tolist())
    with np.load(fc_path, allow_pickle=False) as payload:
        wpli_band_names = tuple(str(value) for value in payload["band_names"].tolist())
        edge_list = tuple((str(first), str(second)) for first, second in payload["edge_list"].tolist())
    if edge_list != build_edge_list():
        raise ValueError("Saved WPLI edge_list does not match the canonical 62-channel upper triangle.")
    if channel_names != CANONICAL_CHANNELS_62:
        raise ValueError("Saved PSD channel order does not match CANONICAL_CHANNELS_62.")
    if "Gamma" in wpli_band_names:
        raise ValueError("WPLI band names unexpectedly include Gamma.")
    return FeatureMetadata(channel_names, frequency_bins, edge_list, wpli_band_names)


def psd_band_labels(frequency_bins: Iterable[float]) -> np.ndarray:
    labels = []
    for frequency in frequency_bins:
        label = "Other"
        for name, low, high in PSD_BANDS:
            if low <= float(frequency) < high:
                label = name
                break
        labels.append(label)
    result = np.asarray(labels, dtype=object)
    if np.any(result == "Other"):
        missing = ", ".join(str(value) for value in np.asarray(list(frequency_bins))[result == "Other"])
        raise ValueError(f"PSD frequency bins fell outside fixed bands: {missing}")
    return result


def motor_edge_mask(edge_list: Iterable[tuple[str, str]]) -> np.ndarray:
    motor = {channel.upper() for channel in MOTOR_CHANNELS}
    return np.asarray(
        [
            str(first).upper() in motor or str(second).upper() in motor
            for first, second in edge_list
        ],
        dtype=bool,
    )


def build_ablation_mask(spec: AblationSpec, metadata: FeatureMetadata) -> AblationMask:
    psd_labels = psd_band_labels(metadata.frequency_bins)
    wpli_labels = np.asarray(metadata.wpli_band_names, dtype=object)
    psd_freq_keep = np.ones_like(psd_labels, dtype=bool)
    wpli_band_keep = np.ones_like(wpli_labels, dtype=bool)

    shared_only = set(spec.only_bands)
    shared_drop = set(spec.drop_bands)
    psd_only = set(spec.psd_only_bands) or shared_only
    psd_drop = set(spec.psd_drop_bands) | shared_drop
    wpli_only = set(spec.wpli_only_bands) or {band for band in shared_only if band != "Gamma"}
    wpli_drop = set(spec.wpli_drop_bands) | {band for band in shared_drop if band != "Gamma"}

    if psd_only:
        psd_freq_keep &= np.isin(psd_labels, sorted(psd_only))
    if psd_drop:
        psd_freq_keep &= ~np.isin(psd_labels, sorted(psd_drop))
    if wpli_only:
        wpli_band_keep &= np.isin(wpli_labels, sorted(wpli_only))
    if wpli_drop:
        wpli_band_keep &= ~np.isin(wpli_labels, sorted(wpli_drop))

    psd_base = np.broadcast_to(psd_freq_keep.reshape(1, -1), (len(metadata.channel_names), len(psd_labels))).copy()
    edge_keep = np.ones(len(metadata.edge_list), dtype=bool)
    if spec.wpli_edge_mode == "motor":
        edge_keep &= motor_edge_mask(metadata.edge_list)
    elif spec.wpli_edge_mode == "non_motor":
        edge_keep &= ~motor_edge_mask(metadata.edge_list)
    elif spec.wpli_edge_mode != "all":
        raise ValueError(f"Unsupported wpli_edge_mode: {spec.wpli_edge_mode}")
    wpli_base = edge_keep.reshape(-1, 1) & wpli_band_keep.reshape(1, -1)

    psd_eo = psd_base.copy() if spec.keep_psd and "EO" in spec.states else np.zeros_like(psd_base, dtype=bool)
    psd_ec = psd_base.copy() if spec.keep_psd and "EC" in spec.states else np.zeros_like(psd_base, dtype=bool)
    wpli_eo = wpli_base.copy() if spec.keep_wpli and "EO" in spec.states else np.zeros_like(wpli_base, dtype=bool)
    wpli_ec = wpli_base.copy() if spec.keep_wpli and "EC" in spec.states else np.zeros_like(wpli_base, dtype=bool)
    description = (
        f"{spec.input_description}; standardized-space zero mask; "
        f"active_psd={int(psd_eo.sum() + psd_ec.sum())}; "
        f"active_wpli={int(wpli_eo.sum() + wpli_ec.sum())}"
    )
    return AblationMask(
        psd_eo=psd_eo,
        psd_ec=psd_ec,
        wpli_eo=wpli_eo,
        wpli_ec=wpli_ec,
        n_active_psd_values=int(psd_eo.sum() + psd_ec.sum()),
        n_active_wpli_values=int(wpli_eo.sum() + wpli_ec.sum()),
        description=description,
    )


def apply_ablation_mask_to_batch(batch: dict[str, torch.Tensor], mask: AblationMask) -> dict[str, torch.Tensor]:
    masked = dict(batch)
    for key, array in (
        ("psd_eo", mask.psd_eo),
        ("psd_ec", mask.psd_ec),
        ("wpli_eo", mask.wpli_eo),
        ("wpli_ec", mask.wpli_ec),
    ):
        if key not in masked:
            continue
        tensor_mask = torch.as_tensor(array, dtype=masked[key].dtype, device=masked[key].device).unsqueeze(0)
        masked[key] = masked[key] * tensor_mask
    return masked


def run_masked_feature_ssl_pretraining(
    pairs: Iterable[FeatureSSLPairRecord],
    training_config: FeatureSSLTrainingConfig,
    mask: AblationMask,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    config = training_config
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
    optimizer = torch.optim.Adam([*model.branch_models.parameters(), *projection_head.parameters()], lr=config.lr)
    loss_fn = NTXentLoss(temperature=config.temperature)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(records))
        epoch_losses: list[float] = []
        on_diag_losses: list[float] = []
        off_diag_losses: list[float] = []
        model.train()
        projection_head.train()
        for start in range(0, len(order), config.batch_size):
            batch_records = [records[int(index)] for index in order[start : start + config.batch_size]]
            batch = _make_batch(batch_records, scaler, device, "multimodal")
            batch = apply_ablation_mask_to_batch(batch, mask)
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
            projection_a = projection_head(model.extract_embedding(view_a))
            projection_b = projection_head(model.extract_embedding(view_b))
            components = _feature_alignment_components(projection_a, projection_b, config, loss_fn)
            loss = components["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
            on_diag_losses.append(float(components["on_diag"].detach().cpu().item()))
            off_diag_losses.append(float(components["off_diag"].detach().cpu().item()))
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "objective": _feature_ssl_objective_label(config.ssl_objective),
                "ssl_objective": config.ssl_objective,
                "barlow_on_diag_loss": float(np.mean(on_diag_losses)),
                "barlow_off_diag_loss": float(np.mean(off_diag_losses)),
                "data_scope": normalize_ssl_data_scope(config.data_scope),
                "feature_kind": config.feature_kind,
                "fusion": config.fusion,
                "encoder_kind": config.encoder_kind,
                "n_pairs": len(records),
                "batch_size": config.batch_size,
                "embedding_dim": config.embedding_dim,
                "projection_dim": config.projection_dim,
                "learning_rate": config.lr,
                "input_mask_description": mask.description,
            }
        )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("branch_models.")
    }
    return state, pd.DataFrame(rows)


def run_one_ablation_seed(
    *,
    spec: AblationSpec,
    mask: AblationMask,
    records: list[SupervisedFeatureRecord],
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    all_ssl_pairs: list[FeatureSSLPairRecord],
    ssl_data_scope: str,
    seed: int,
    device: torch.device,
    limit_folds: int | None,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_lr: float,
    projection_dim: int,
    temperature: float,
    noise_std: float,
    feature_mask_prob: float,
    embedding_dim: int,
    dropout: float,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    rank_margin: float,
    swa_start_epoch: int,
    swa_lr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual_module = load_residual_module()
    variant = residual_module.VARIANTS["highrank"]
    residual_module._seed_everything(seed)
    record_by_subject = {record.subject_id: record for record in records}
    folds = make_loso_folds([record.subject_id for record in records])
    if limit_folds is not None:
        folds = folds[: int(limit_folds)]
    prediction_rows: list[dict[str, Any]] = []
    ssl_history_frames: list[pd.DataFrame] = []
    loss_rows: list[dict[str, Any]] = []

    for fold in folds:
        ssl_pairs = feature_ssl_pairs_for_scope(
            all_ssl_pairs,
            data_scope=ssl_data_scope,
            strict_loso_test_subject_id=fold.test_subject_id,
        )
        ssl_config = FeatureSSLTrainingConfig(
            data_scope=ssl_data_scope,
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="cnn",
            epochs=pretrain_epochs,
            batch_size=pretrain_batch_size,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            lr=pretrain_lr,
            ssl_objective="barlow",
            temperature=temperature,
            noise_std=noise_std,
            feature_mask_prob=feature_mask_prob,
            device=str(device),
            seed=seed,
        )
        pretrained_state, ssl_history = run_masked_feature_ssl_pretraining(ssl_pairs, ssl_config, mask)
        ssl_history = ssl_history.copy()
        ssl_history.insert(0, "ablation_name", spec.name)
        ssl_history.insert(1, "seed", seed)
        ssl_history.insert(2, "fold_id", fold.fold_index)
        ssl_history.insert(3, "test_subject_id", fold.test_subject_id)
        ssl_history_frames.append(ssl_history)

        train_subjects = list(fold.train_subject_ids)
        fit_subjects, val_subjects = _train_validation_subjects(train_subjects, fold.fold_index)
        scaler = _fit_state_scaler(record_by_subject[subject_id] for subject_id in fit_subjects)
        target_scaler = fold_local_standardize_signed_distance(
            targets,
            fit_subject_ids=fit_subjects,
            transform_subject_ids=[*fit_subjects, *val_subjects, fold.test_subject_id],
        )
        train_batch = make_masked_residual_batch(
            (record_by_subject[subject_id] for subject_id in fit_subjects),
            fit_subjects,
            scaler,
            target_scaler.frame,
            device,
            mask,
        )
        val_batch = make_masked_residual_batch(
            (record_by_subject[subject_id] for subject_id in val_subjects),
            val_subjects,
            scaler,
            target_scaler.frame,
            device,
            mask,
        )
        test_batch = _make_batch([record_by_subject[fold.test_subject_id]], scaler, device, "multimodal")
        test_batch = apply_ablation_mask_to_batch(test_batch, mask)

        model = residual_module.ResidualAwarePatientBarlowModel(
            embedding_dim=embedding_dim,
            dropout=dropout,
            residual_alpha=1.0,
            residual_probability_scale=1.0,
        ).to(device)
        model.load_patient_barlow_state(pretrained_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, patience // 2))
        swa_model = build_swa_model(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
        early_stopping = EarlyStopping(patience=patience, mode="min")

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_batch)
            loss, components = residual_aware_multitask_loss(
                outputs,
                train_batch,
                lambda_reg=variant.lambda_reg,
                lambda_rank=variant.lambda_rank,
                lambda_soft=variant.lambda_soft,
                rank_margin=rank_margin,
            )
            loss.backward()
            optimizer.step()
            val_loss, val_components = evaluate_residual_loss(model, val_batch, variant, rank_margin)
            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step(val_loss)
            stopped = early_stopping.step(val_loss, model)
            loss_rows.append(
                {
                    "ablation_name": spec.name,
                    "seed": seed,
                    "fold_id": fold.fold_index,
                    "test_subject_id": fold.test_subject_id,
                    "fit_subject_ids": ";".join(fit_subjects),
                    "val_subject_ids": ";".join(val_subjects),
                    "epoch": epoch,
                    "training_phase": "residual_aware_supervised",
                    "train_loss": float(loss.detach().cpu().item()),
                    "val_loss": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "lambda_reg": variant.lambda_reg,
                    "lambda_rank": variant.lambda_rank,
                    "lambda_soft": variant.lambda_soft,
                    "rank_margin": rank_margin,
                    "uses_swa": True,
                    "swa_start_epoch": swa_start_epoch,
                    "swa_lr": swa_lr,
                    "swa_n_averaged": int(swa_model.n_averaged.detach().cpu().item()),
                    "target_mean": target_scaler.mean,
                    "target_std": target_scaler.std,
                    "target_tau": target_scaler.tau,
                    "input_mask_description": mask.description,
                    **{f"train_{key}": float(value.detach().cpu().item()) for key, value in components.items()},
                    **{f"val_{key}": value for key, value in val_components.items()},
                    "stopped_early": bool(stopped),
                }
            )
            if stopped:
                break

        if int(swa_model.n_averaged.detach().cpu().item()) > 0:
            update_swa_batch_norm([train_batch], swa_model)
            eval_model = swa_model
            inference_weight_source = "swa_averaged_weights"
        else:
            early_stopping.restore_best_weights(model)
            eval_model = model
            inference_weight_source = "early_stopping_best_weights"
        eval_model.eval()
        with torch.no_grad():
            outputs = eval_model(test_batch)
            probability = float(outputs["classification_probability"].detach().cpu().numpy()[0, 0])
            residual_probability = float(outputs["residual_probability"].detach().cpu().numpy()[0, 0])
            residual_score = float(outputs["residual_score"].detach().cpu().numpy()[0, 0])
        logit = probability_to_logit(probability)
        test_record = record_by_subject[fold.test_subject_id]
        prediction_rows.append(
            {
                "ablation_name": spec.name,
                "seed": seed,
                "fold_id": fold.fold_index,
                "test_subject_id": fold.test_subject_id,
                "subject_id": fold.test_subject_id,
                "y_true": int(test_record.label),
                "y_prob": probability,
                "y_score": probability,
                "y_pred": int(probability >= 0.5),
                "logit": logit,
                "classification_y_score": probability,
                "residual_y_score": residual_probability,
                "residual_score_z": residual_score,
                "model_type": "final_residual_aware_patient_barlow_ssl_cnn_highrank",
                "model": "final_residual_aware_patient_barlow_ssl_cnn_highrank",
                "uses_ssl": True,
                "uses_residual_aware_heads": True,
                "uses_swa": True,
                "ssl_pretraining_scope": "strict_loso_supervised_training_pool_only",
                "ssl_data_scope": normalize_ssl_data_scope(ssl_data_scope),
                "input_mask_description": mask.description,
                "inference_head": "classification_head",
                "inference_weight_source": inference_weight_source,
                "ablation_family": spec.family,
                "input_description": spec.input_description,
                "n_active_psd_values": mask.n_active_psd_values,
                "n_active_wpli_values": mask.n_active_wpli_values,
                "n_active_eeg_feature_values": mask.n_active_psd_values + mask.n_active_wpli_values,
                "variant": variant.name,
                "lambda_reg": variant.lambda_reg,
                "lambda_rank": variant.lambda_rank,
                "lambda_soft": variant.lambda_soft,
                "rank_margin": rank_margin,
                "residual_alpha": 1.0,
                "swa_start_epoch": swa_start_epoch,
                "swa_lr": swa_lr,
                "swa_n_averaged": int(swa_model.n_averaged.detach().cpu().item()),
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    metrics = metrics_for_prediction_frame(predictions)
    metrics.insert(0, "ablation_name", spec.name)
    metrics.insert(1, "seed", seed)
    metrics.insert(2, "ablation_family", spec.family)
    metrics.insert(3, "model_type", "final_residual_aware_patient_barlow_ssl_cnn_highrank")
    metrics["uses_ssl"] = True
    metrics["uses_residual_aware_heads"] = True
    metrics["uses_swa"] = True
    metrics["ssl_pretraining_scope"] = "strict_loso_supervised_training_pool_only"
    metrics["analysis_status"] = "independent_ablation_ssl_pretraining_and_supervised_finetuning"
    metrics["n_subjects"] = int(predictions["test_subject_id"].nunique()) if not predictions.empty else 0
    metrics["n_active_psd_values"] = mask.n_active_psd_values
    metrics["n_active_wpli_values"] = mask.n_active_wpli_values
    metrics["input_mask_description"] = mask.description
    return predictions, metrics, pd.concat(ssl_history_frames, ignore_index=True), pd.DataFrame(loss_rows)


def run_fixed_final_checkpoint_masked_inference(
    *,
    specs: list[AblationSpec],
    masks: Mapping[str, AblationMask],
    records: list[SupervisedFeatureRecord],
    checkpoint_dir: Path,
    seeds: list[int],
    device: torch.device,
    limit_folds: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Final checkpoint directory does not exist: {checkpoint_dir}")
    residual_module = load_residual_module()
    record_by_subject = {record.subject_id: record for record in records}
    wanted_seeds = set(int(seed) for seed in seeds)
    prediction_rows: list[dict[str, Any]] = []
    seed_metric_rows: list[dict[str, Any]] = []
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No final checkpoint .pt files found in {checkpoint_dir}")

    loaded_by_seed: dict[int, int] = {}
    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        seed = int(metadata.get("seed", -1))
        if seed not in wanted_seeds:
            continue
        fold_id = int(metadata["fold_index"])
        if limit_folds is not None and fold_id >= int(limit_folds):
            continue
        test_subject_id = normalize_subject_id(metadata["test_subject_id"])
        if test_subject_id not in record_by_subject:
            continue
        scaler = checkpoint_scaler_to_numpy(payload["state_scaler"])
        test_record = record_by_subject[test_subject_id]
        model = residual_module.ResidualAwarePatientBarlowModel(
            embedding_dim=int(metadata.get("embedding_dim", 32)),
            dropout=float(metadata.get("dropout", 0.0)),
            residual_alpha=1.0,
            residual_probability_scale=float(metadata.get("residual_probability_scale", 1.0)),
        ).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        loaded_by_seed[seed] = loaded_by_seed.get(seed, 0) + 1
        for spec in specs:
            mask = masks[spec.name]
            batch = _make_batch([test_record], scaler, device, "multimodal")
            batch = apply_ablation_mask_to_batch(batch, mask)
            with torch.no_grad():
                outputs = model(batch)
                probability = float(outputs["classification_probability"].detach().cpu().numpy()[0, 0])
                residual_probability = float(outputs["residual_probability"].detach().cpu().numpy()[0, 0])
                residual_score = float(outputs["residual_score"].detach().cpu().numpy()[0, 0])
            prediction_rows.append(
                {
                    "ablation_name": spec.name,
                    "seed": seed,
                    "fold_id": fold_id,
                    "test_subject_id": test_subject_id,
                    "subject_id": test_subject_id,
                    "y_true": int(test_record.label),
                    "y_prob": probability,
                    "y_score": probability,
                    "y_pred": int(probability >= 0.5),
                    "logit": probability_to_logit(probability),
                    "classification_y_score": probability,
                    "residual_y_score": residual_probability,
                    "residual_score_z": residual_score,
                    "model_type": "final_residual_aware_patient_barlow_ssl_cnn_highrank",
                    "model": "final_residual_aware_patient_barlow_ssl_cnn_highrank",
                    "uses_ssl": True,
                    "uses_residual_aware_heads": True,
                    "uses_swa": bool(metadata.get("use_swa", True)),
                    "ssl_pretraining_scope": "fixed_full_input_final_checkpoint",
                    "ssl_data_scope": "fixed_existing_final_model",
                    "input_mask_description": mask.description,
                    "inference_head": "classification_head",
                    "inference_weight_source": "saved_final_swa_checkpoint",
                    "analysis_status": "approximate_fixed_final_checkpoint_masked_inference",
                    "ablation_family": spec.family,
                    "input_description": spec.input_description,
                    "n_active_psd_values": mask.n_active_psd_values,
                    "n_active_wpli_values": mask.n_active_wpli_values,
                    "n_active_eeg_feature_values": mask.n_active_psd_values + mask.n_active_wpli_values,
                    "variant": str(metadata.get("variant", "highrank")),
                    "lambda_reg": float(metadata.get("lambda_reg", 0.3)),
                    "lambda_rank": float(metadata.get("lambda_rank", 0.3)),
                    "lambda_soft": float(metadata.get("lambda_soft", 0.1)),
                    "rank_margin": float(metadata.get("rank_margin", 0.5)),
                    "residual_alpha": 1.0,
                    "swa_start_epoch": int(metadata.get("swa_start_epoch", 50)),
                    "swa_lr": float(metadata.get("swa_lr", 5e-4)),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError(
            f"No predictions were produced from {checkpoint_dir}; loaded_by_seed={loaded_by_seed!r}"
        )
    for (ablation_name, seed), frame in predictions.groupby(["ablation_name", "seed"], sort=False):
        metric_row = metrics_for_prediction_frame(frame).iloc[0].to_dict()
        spec = next(item for item in specs if item.name == ablation_name)
        mask = masks[ablation_name]
        seed_metric_rows.append(
            {
                "ablation_name": ablation_name,
                "seed": int(seed),
                "ablation_family": spec.family,
                "model_type": "final_residual_aware_patient_barlow_ssl_cnn_highrank",
                "uses_ssl": True,
                "uses_residual_aware_heads": True,
                "uses_swa": True,
                "ssl_pretraining_scope": "fixed_full_input_final_checkpoint",
                "analysis_status": "approximate_fixed_final_checkpoint_masked_inference",
                "n_subjects": int(frame["test_subject_id"].nunique()),
                "n_active_psd_values": mask.n_active_psd_values,
                "n_active_wpli_values": mask.n_active_wpli_values,
                "input_mask_description": mask.description,
                **metric_row,
            }
        )
    seed_metrics = pd.DataFrame(seed_metric_rows)
    return predictions.sort_values(["ablation_name", "seed", "fold_id"]).reset_index(drop=True), seed_metrics


def checkpoint_scaler_to_numpy(payload: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for branch, values in payload.items():
        scalers[str(branch)] = (
            torch.as_tensor(values["mean"]).detach().cpu().numpy().astype(np.float32),
            torch.as_tensor(values["std"]).detach().cpu().numpy().astype(np.float32),
        )
    return scalers


def make_masked_residual_batch(
    records: Iterable[SupervisedFeatureRecord],
    subject_ids: list[str],
    scaler: Any,
    target_frame: pd.DataFrame,
    device: torch.device,
    mask: AblationMask,
) -> dict[str, torch.Tensor]:
    batch = _make_batch(records, scaler, device, "multimodal")
    batch = apply_ablation_mask_to_batch(batch, mask)
    ordered = [normalize_subject_id(subject_id) for subject_id in subject_ids]
    batch["signed_distance_z"] = torch.as_tensor(
        target_frame.loc[ordered, "signed_distance_z"].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    batch["soft_y"] = torch.as_tensor(
        target_frame.loc[ordered, "soft_y"].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    return batch


def evaluate_residual_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    variant: Any,
    rank_margin: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        loss, components = residual_aware_multitask_loss(
            model(batch),
            batch,
            lambda_reg=variant.lambda_reg,
            lambda_rank=variant.lambda_rank,
            lambda_soft=variant.lambda_soft,
            rank_margin=rank_margin,
        )
    return float(loss.detach().cpu().item()), {key: float(value.detach().cpu().item()) for key, value in components.items()}


def metrics_for_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = binary_classification_metrics(
        frame["y_true"].to_numpy(dtype=int),
        frame["y_prob"].to_numpy(dtype=float),
        y_pred=frame["y_pred"].to_numpy(dtype=int),
    )
    return pd.DataFrame([{key: metrics[key] for key in METRIC_COLUMNS}])


def seedmean_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ablation_name, frame in predictions.groupby("ablation_name", sort=False):
        averaged = (
            frame.groupby("test_subject_id", as_index=False)
            .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
            .sort_values("test_subject_id")
        )
        averaged["y_pred"] = (averaged["y_prob"] >= 0.5).astype(int)
        metrics = binary_classification_metrics(
            averaged["y_true"].to_numpy(dtype=int),
            averaged["y_prob"].to_numpy(dtype=float),
            y_pred=averaged["y_pred"].to_numpy(dtype=int),
        )
        rows.append(
            {
                "ablation_name": ablation_name,
                "n_subjects": int(averaged["test_subject_id"].nunique()),
                "n_seeds": int(frame["seed"].nunique()),
                **{key: metrics[key] for key in METRIC_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def summarize_for_tables(
    seed_metrics: pd.DataFrame,
    seedmean: pd.DataFrame,
    specs: list[AblationSpec],
    masks: Mapping[str, AblationMask],
) -> pd.DataFrame:
    spec_by_name = {spec.name: spec for spec in specs}
    rows = []
    for ablation_name, frame in seed_metrics.groupby("ablation_name", sort=False):
        spec = spec_by_name[ablation_name]
        mask = masks[ablation_name]
        seedmean_row = seedmean[seedmean["ablation_name"] == ablation_name].iloc[0].to_dict()
        row: dict[str, Any] = {
            "ablation_name": ablation_name,
            "ablation_family": spec.family,
            "input_description": spec.input_description,
            "model_type": "final_residual_aware_patient_barlow_ssl_cnn_highrank",
            "uses_ssl": True,
            "uses_residual_aware_heads": True,
            "uses_swa": True,
            "ssl_pretraining_scope": "strict_loso_supervised_training_pool_only",
            "analysis_status": str(frame["analysis_status"].iloc[0]) if "analysis_status" in frame else "independent_ablation_ssl_pretraining_and_supervised_finetuning",
            "n_subjects": int(frame["n_subjects"].max()),
            "n_seeds": int(frame["seed"].nunique()),
            "n_active_psd_values": mask.n_active_psd_values,
            "n_active_wpli_values": mask.n_active_wpli_values,
        }
        for metric in METRIC_COLUMNS:
            output_name = "brier" if metric == "brier_score" else metric
            row[f"{output_name}_mean"] = float(frame[metric].mean())
            row[f"{output_name}_sd"] = float(frame[metric].std(ddof=1)) if frame[metric].shape[0] > 1 else 0.0
            row[f"{output_name}_seedmean"] = float(seedmean_row[metric])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    full = table[table["ablation_name"] == "full_psd_wpli"]
    if not full.empty:
        full_row = full.iloc[0]
        table["reference_ablation"] = "full_psd_wpli"
        table["delta_balanced_accuracy_vs_full"] = table["balanced_accuracy_seedmean"] - float(full_row["balanced_accuracy_seedmean"])
        table["delta_roc_auc_vs_full"] = table["roc_auc_seedmean"] - float(full_row["roc_auc_seedmean"])
        table["delta_pr_auc_vs_full"] = table["pr_auc_seedmean"] - float(full_row["pr_auc_seedmean"])
        table["delta_brier_vs_full"] = table["brier_seedmean"] - float(full_row["brier_seedmean"])
    else:
        table["reference_ablation"] = "full_psd_wpli"
        for column in ("delta_balanced_accuracy_vs_full", "delta_roc_auc_vs_full", "delta_pr_auc_vs_full", "delta_brier_vs_full"):
            table[column] = np.nan
    table["interpretation_for_paper"] = table.apply(interpretation_for_row, axis=1)
    order = {name: index for index, name in enumerate([spec.name for spec in ablation_specs()])}
    return table.sort_values("ablation_name", key=lambda s: s.map(order)).reset_index(drop=True)


def interpretation_for_row(row: pd.Series) -> str:
    if row["ablation_name"] == "full_psd_wpli":
        return "Final residual-aware SSL-CNN reference input under strict patient-level LOSO."
    delta = row.get("delta_roc_auc_vs_full", np.nan)
    if pd.isna(delta):
        return "Exploratory final-model feature-subset ablation; reference full model was not in this partial run."
    direction = "retained comparable ranking information" if float(delta) >= -0.05 else "showed lower ranking performance than full input"
    return f"Exploratory final-model feature-subset ablation; {direction}."


def write_incremental_outputs(
    output_root: Path,
    prediction_frames: list[pd.DataFrame],
    seed_metric_frames: list[pd.DataFrame],
    ssl_history_frames: list[pd.DataFrame],
    loss_history_frames: list[pd.DataFrame],
    specs: list[AblationSpec],
    masks: Mapping[str, AblationMask],
) -> list[Path]:
    ensure_output_dirs(output_root)
    outputs: list[Path] = []
    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True) if prediction_frames else pd.DataFrame()
    seed_metrics = pd.concat([frame for frame in seed_metric_frames if not frame.empty], ignore_index=True) if seed_metric_frames else pd.DataFrame()
    if not predictions.empty:
        predictions = predictions.drop_duplicates(["ablation_name", "seed", "fold_id", "test_subject_id"], keep="last")
        path = output_root / "results" / "predictions" / "final_ssl_cnn_feature_state_band_ablation_predictions.csv"
        predictions.to_csv(path, index=False)
        outputs.append(path)
    if not seed_metrics.empty:
        seed_metrics = seed_metrics.drop_duplicates(["ablation_name", "seed"], keep="last")
        path = output_root / "results" / "metrics" / "final_ssl_cnn_feature_state_band_ablation_seed_metrics.csv"
        seed_metrics.to_csv(path, index=False)
        outputs.append(path)
    if not predictions.empty:
        seedmean = seedmean_metrics(predictions)
        path = output_root / "results" / "metrics" / "final_ssl_cnn_feature_state_band_ablation_seedmean_metrics.csv"
        seedmean.to_csv(path, index=False)
        outputs.append(path)
    else:
        seedmean = pd.DataFrame()
    if not seed_metrics.empty and not seedmean.empty:
        table = summarize_for_tables(seed_metrics, seedmean, specs, masks)
        delta = table[
            [
                "ablation_name",
                "reference_ablation",
                "delta_balanced_accuracy_vs_full",
                "delta_roc_auc_vs_full",
                "delta_pr_auc_vs_full",
                "delta_brier_vs_full",
            ]
        ].copy()
        path = output_root / "results" / "metrics" / "final_ssl_cnn_feature_state_band_ablation_delta_vs_full.csv"
        delta.to_csv(path, index=False)
        outputs.append(path)
        main_table = table[table["ablation_name"].isin(MAIN_ABLATIONS)].copy()
        supp_table = table[~table["ablation_name"].isin(MAIN_ABLATIONS)].copy()
        table_path = output_root / "results" / "tables" / "table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv"
        main_table.to_csv(table_path, index=False)
        outputs.append(table_path)
        md_path = table_path.with_suffix(".md")
        md_path.write_text(to_markdown(main_table), encoding="utf-8")
        outputs.append(md_path)
        supp_path = output_root / "results" / "tables" / "supplementary_final_ssl_cnn_feature_state_band_ablation.csv"
        supp_table.to_csv(supp_path, index=False)
        outputs.append(supp_path)
        supp_md_path = supp_path.with_suffix(".md")
        supp_md_path.write_text(to_markdown(supp_table), encoding="utf-8")
        outputs.append(supp_md_path)
        if not main_table.empty:
            outputs.extend(write_figures(output_root, main_table))
        write_result_docs(output_root, table, main_table, supp_table)
        outputs.extend(
            [
                output_root / "docs" / "final_ssl_cnn_feature_state_band_ablation_results_for_section3.md",
                output_root / "docs" / "final_ssl_cnn_feature_state_band_ablation_numeric_claims.md",
            ]
        )
    if ssl_history_frames:
        ssl_history = pd.concat([frame for frame in ssl_history_frames if not frame.empty], ignore_index=True)
        path = output_root / "results" / "ssl" / "final_ssl_cnn_feature_state_band_ablation_ssl_history.csv"
        ssl_history.to_csv(path, index=False)
        outputs.append(path)
    if loss_history_frames:
        loss_history = pd.concat([frame for frame in loss_history_frames if not frame.empty], ignore_index=True)
        path = output_root / "results" / "loss_history" / "final_ssl_cnn_feature_state_band_ablation_supervised_loss_history.csv"
        loss_history.to_csv(path, index=False)
        outputs.append(path)
    return outputs


def write_figures(output_root: Path, main_table: pd.DataFrame) -> list[Path]:
    out_dir = output_root / "results" / "figures" / "revised_initial"
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = main_table.copy()
    ordered["active_values"] = ordered["n_active_psd_values"] + ordered["n_active_wpli_values"]
    ordered = ordered.sort_values("balanced_accuracy_seedmean", ascending=True)
    label_map = {
        "full_psd_wpli": "Full",
        "psd_only": "PSD only",
        "wpli_only": "WPLI only",
        "eo_only": "EO only",
        "ec_only": "EC only",
        "beta_medium_beta_high": "Beta mid/high",
        "full_minus_beta_medium_beta_high": "Minus beta mid/high",
        "motor_wpli_edges_only": "Motor WPLI",
        "full_minus_motor_wpli_edges": "Minus motor WPLI",
    }
    labels = [label_map.get(name, name) for name in ordered["ablation_name"]]
    metrics = [
        ("balanced_accuracy_seedmean", "Balanced accuracy", "higher is better"),
        ("roc_auc_seedmean", "ROC-AUC", "higher is better"),
        ("pr_auc_seedmean", "PR-AUC", "higher is better"),
        ("brier_seedmean", "Brier", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, max(4.5, 0.42 * len(ordered))), sharey=True)
    for ax, (column, title, note) in zip(axes, metrics, strict=True):
        values = ordered[column].astype(float)
        colors = ["#4C78A8" if name == "full_psd_wpli" else "#72B7B2" for name in ordered["ablation_name"]]
        ax.barh(np.arange(len(ordered)), values, color=colors, edgecolor="#263238", linewidth=0.4)
        ax.set_title(f"{title}\n{note}", fontsize=9)
        ax.grid(axis="x", color="#d0d7de", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        if column != "brier_seedmean":
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(0, max(0.05, float(values.max()) * 1.18))
        for idx, value in enumerate(values):
            ax.text(float(value) + (0.01 if column != "brier_seedmean" else 0.003), idx, f"{value:.3f}", va="center", fontsize=7)
    axes[0].set_yticks(np.arange(len(ordered)))
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Final residual-aware SSL-CNN feature-subset ablation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ranking_base = out_dir / "figure5c_final_ssl_cnn_feature_state_band_ablation_ranking"
    outputs = save_figure_all_formats(fig, ranking_base)
    plt.close(fig)

    selected = main_table[main_table["ablation_name"].isin(
        ["full_psd_wpli", "psd_only", "wpli_only", "beta_medium_beta_high", "motor_wpli_edges_only"]
    )].copy()
    selected["active_values"] = selected["n_active_psd_values"] + selected["n_active_wpli_values"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(
        selected["active_values"].astype(float),
        selected["roc_auc_seedmean"].astype(float),
        s=90,
        color="#E45756",
        edgecolor="#263238",
        linewidth=0.6,
    )
    for row in selected.itertuples(index=False):
        ax.annotate(label_map.get(row.ablation_name, row.ablation_name), (row.active_values, row.roc_auc_seedmean), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    x_values = selected["active_values"].astype(float)
    ax.set_xlim(float(x_values.min()) / 1.15, float(x_values.max()) * 1.35)
    ax.set_xlabel("Active EEG feature values (log scale)")
    ax.set_ylabel("Seed-mean ROC-AUC")
    ax.set_title("Final SSL-CNN information efficiency")
    ax.grid(True, color="#d0d7de", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    efficiency_base = out_dir / "figure5d_final_ssl_cnn_information_efficiency"
    outputs.extend(save_figure_all_formats(fig, efficiency_base))
    plt.close(fig)
    return outputs


def save_figure_all_formats(fig: plt.Figure, base_path: Path) -> list[Path]:
    outputs = []
    for suffix in ("png", "svg", "pdf", "tiff"):
        path = base_path.with_suffix(f".{suffix}")
        kwargs = {"dpi": 600} if suffix in {"png", "tiff"} else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    return outputs


def write_result_docs(output_root: Path, table: pd.DataFrame, main_table: pd.DataFrame, supp_table: pd.DataFrame) -> None:
    doc_path = output_root / "docs" / "final_ssl_cnn_feature_state_band_ablation_results_for_section3.md"
    numeric_path = output_root / "docs" / "final_ssl_cnn_feature_state_band_ablation_numeric_claims.md"
    table_path = output_root / "results" / "tables" / "table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv"
    seedmean_path = output_root / "results" / "metrics" / "final_ssl_cnn_feature_state_band_ablation_seedmean_metrics.csv"
    seed_metric_path = output_root / "results" / "metrics" / "final_ssl_cnn_feature_state_band_ablation_seed_metrics.csv"
    fig5c = output_root / "results" / "figures" / "revised_initial" / "figure5c_final_ssl_cnn_feature_state_band_ablation_ranking.png"
    fig5d = output_root / "results" / "figures" / "revised_initial" / "figure5d_final_ssl_cnn_information_efficiency.png"
    analysis_status = str(table["analysis_status"].iloc[0]) if "analysis_status" in table.columns and not table.empty else "unknown"
    approximate = "approximate" in analysis_status

    def row(name: str) -> pd.Series | None:
        subset = table[table["ablation_name"] == name]
        return None if subset.empty else subset.iloc[0]

    lines = [
        "# 最终 residual-aware SSL-CNN 特征、状态和频段消融结果",
        "",
        "## 1. 为什么旧 Logistic 消融不能作为最终模型消融主结果",
        "",
        "旧 `modality_state_band_ablation_rerun.csv` 是 tabular Logistic Regression feature-subset support analysis。它使用折内特征选择和线性分类器，不是 10-seed residual-aware SSL-CNN，也没有对每个消融组重新执行 Patient-level Barlow 预训练和残差感知微调。因此旧结果只能作为补充支持分析，不能替代主文 Results 3.3 的最终模型消融。",
        "",
        "## 2. 本轮固定口径",
        "",
        (
            "本轮结果固定为 final residual-aware patient-level Barlow SSL-CNN，highrank variant，patient-level LOSO，10 seeds，SWA，推断时只读取 classification head。"
            if approximate
            else "本轮结果固定为 final residual-aware patient-level Barlow SSL-CNN，highrank variant，patient-level LOSO，10 seeds，SWA，推断时只读取 classification head。每个 LOSO fold 的测试患者在 scaler、SSL 预训练、残差目标标准化、监督微调和阈值前均被排除。"
        ),
        "",
        (
            "计算资源说明：当前输出标记为 `approximate_fixed_final_checkpoint_masked_inference`。它使用已保存的 full-input final SWA checkpoint，在各 checkpoint 的 fold-local scaler 后对输入子集置零并重新推断；没有对每个消融组重新进行 Barlow 预训练和监督微调。因此它是最终模型 masked-input retention/occlusion 近似分析，不能写成独立重训主消融。"
            if approximate
            else "本轮为每个消融组独立执行 Barlow 预训练和 residual-aware supervised fine-tuning。"
        ),
        "",
        "## 3. 输入构造",
        "",
        "每个样本包含 PSD_EO、PSD_EC、WPLI_EO、WPLI_EC。被移除的输入部分在 fold-local standardized space 中置零，因此零表示该训练折标准化后的均值。PSD 30-45 Hz 统一作为 Gamma；WPLI 只包含 Delta、Theta、Alpha、Beta Low、Beta Medium、Beta High，不构造 WPLI Gamma。",
        "",
    ]
    report_order = [
        ("full_psd_wpli", "## 4. full_psd_wpli 表现"),
        ("psd_only", "## 5. psd_only 与 full_psd_wpli"),
        ("wpli_only", "## 5. wpli_only 与 full_psd_wpli"),
        ("eo_only", "## 6. EO only 与 full_psd_wpli"),
        ("ec_only", "## 6. EC only 与 full_psd_wpli"),
        ("beta_medium_beta_high", "## 7. beta_medium_beta_high"),
        ("full_minus_beta_medium_beta_high", "## 7. full_minus_beta_medium_beta_high"),
        ("motor_wpli_edges_only", "## 8. motor_wpli_edges_only"),
        ("full_minus_motor_wpli_edges", "## 8. full_minus_motor_wpli_edges"),
    ]
    for name, heading in report_order:
        current = row(name)
        lines.extend([heading, ""])
        if current is None:
            lines.extend([f"`{name}` 尚未出现在当前输出表中。", ""])
            continue
        lines.extend(
            [
                (
                    f"`{name}` 的 seed-mean balanced accuracy 为 {current['balanced_accuracy_seedmean']:.3f}，"
                    f"ROC-AUC 为 {current['roc_auc_seedmean']:.3f}，PR-AUC 为 {current['pr_auc_seedmean']:.3f}，"
                    f"Brier 为 {current['brier_seedmean']:.3f}。相对 full_psd_wpli 的 "
                    f"delta balanced accuracy 为 {current['delta_balanced_accuracy_vs_full']:.3f}，"
                    f"delta ROC-AUC 为 {current['delta_roc_auc_vs_full']:.3f}。"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 9. 解释范围",
            "",
            (
                "本轮结果回答的是：在已训练最终 residual-aware SSL-CNN checkpoint 下，不同 EEG 输入子集在 masked inference 时保留多少预测信息。结果不直接等同于独立重训消融、因果机制，也不证明某个频段或网络决定恢复结局。"
                if approximate
                else "本轮结果回答的是：在最终 residual-aware SSL-CNN 框架下，不同 EEG 输入子集保留多少预测信息。结果不直接等同于因果机制，也不证明某个频段或网络决定恢复结局。"
            ),
            "",
            "## 10. 谨慎结论",
            "",
        (
            "这些结果应表述为 exploratory approximate masked-inference evidence。建议使用“提示”“保留信息”“模型依赖”这类措辞，避免使用“证明”“决定”“因果机制”或“独立重训消融主结果”等表述。"
            if approximate
            else "这些消融结果应表述为 exploratory final-model evidence。建议使用“提示”“保留信息”“模型依赖”这类措辞，避免使用“证明”“决定”“因果机制”等表述。"
        ),
            "",
            f"来源表：`{table_path}`",
            f"种子均值指标：`{seedmean_path}`",
            f"种子级指标：`{seed_metric_path}`",
            f"Figure 5c：`{fig5c}`",
            f"Figure 5d：`{fig5d}`",
        ]
    )
    doc_path.write_text("\n".join(lines), encoding="utf-8")

    claim_lines = [
        "# 可写入论文正文的数字结论",
        "",
        (
            "所有数字均来自本目录下重新生成的最终 residual-aware SSL-CNN 近似 masked-inference 结果，"
            "analysis_status = approximate_fixed_final_checkpoint_masked_inference，均为 exploratory，不能写成独立重训消融。"
            if approximate
            else "所有数字均来自本目录下重新生成的最终 residual-aware SSL-CNN 消融结果，均为 exploratory。"
        ),
        "",
    ]
    for current in main_table.itertuples(index=False):
        claim_lines.extend(
            [
                f"## {current.ablation_name}",
                "",
                f"- balanced accuracy = {current.balanced_accuracy_seedmean:.3f}，来源 `{table_path}`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。",
                f"- ROC-AUC = {current.roc_auc_seedmean:.3f}，来源 `{table_path}`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。",
                f"- PR-AUC = {current.pr_auc_seedmean:.3f}，来源 `{table_path}`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。",
                f"- Brier = {current.brier_seedmean:.3f}，来源 `{table_path}`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。",
                f"- delta ROC-AUC vs full = {current.delta_roc_auc_vs_full:.3f}，来源 `{table_path}`，口径：seedmean prediction，对应 Table 3，exploratory: yes。",
                "",
            ]
        )
    if supp_table.empty:
        claim_lines.extend(["## Supplementary", "", "当前未生成补充消融组的完整表。", ""])
    numeric_path.write_text("\n".join(claim_lines), encoding="utf-8")


def audit_old_logistic_outputs(source_root: Path, output_root: Path) -> None:
    docs_dir = output_root / "docs"
    metrics_dir = output_root / "results" / "metrics"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prompt_metrics_rerun": source_root / "results" / "metrics" / "modality_state_band_ablation_rerun.csv",
        "prompt_predictions_rerun": source_root / "results" / "predictions" / "modality_state_band_ablation_predictions_rerun.csv",
        "prompt_table3_logistic": source_root / "results" / "tables" / "table3_feature_state_band_ablation_for_paper.csv",
        "legacy_metrics_non_rerun": source_root / "results" / "metrics" / "modality_state_band_ablation.csv",
        "legacy_predictions_non_rerun": source_root / "results" / "predictions" / "modality_state_band_ablation_predictions.csv",
        "secondary_table3_logistic": source_root
        / "rerun_ablation_explainability_secondary_20260609"
        / "results"
        / "tables"
        / "table3_feature_state_band_ablation_for_paper.csv",
        "secondary_metrics_logistic": source_root
        / "rerun_ablation_explainability_secondary_20260609"
        / "results"
        / "metrics"
        / "modality_state_band_ablation_final_metrics.csv",
        "secondary_predictions_logistic": source_root
        / "rerun_ablation_explainability_secondary_20260609"
        / "results"
        / "predictions"
        / "modality_state_band_ablation_final_predictions.csv",
    }
    rows = []
    for key, path in paths.items():
        exists = path.exists()
        row: dict[str, Any] = {
            "artifact": key,
            "path": str(path),
            "exists": exists,
            "is_logistic_regression": False,
            "is_tabular_loso": False,
            "is_10_seed_cnn": False,
            "baseline_clinical_string_status": "not_present",
            "conclusion": "missing",
        }
        if exists:
            frame = pd.read_csv(path)
            text_blob = " ".join(str(value) for value in frame.astype(str).head(200).to_numpy().ravel())
            columns_blob = " ".join(frame.columns.astype(str))
            combined = f"{columns_blob} {text_blob}".lower()
            row["is_logistic_regression"] = "logistic" in combined
            row["is_tabular_loso"] = "feature_selection" in combined or "selectk" in combined or "n_input" in combined
            row["is_10_seed_cnn"] = "seed" in frame.columns and frame["seed"].nunique() == 10 if "seed" in frame.columns else False
            if "baseline_clinical" in combined or "baseline clinical" in combined:
                row["baseline_clinical_string_status"] = "legacy_string_only; source script loads PSD/WPLI EEG tables only"
            row["n_rows"] = len(frame)
            row["columns"] = ";".join(frame.columns.astype(str))
            row["conclusion"] = "tabular feature-subset support analysis; not final residual-aware SSL-CNN ablation"
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit_path = metrics_dir / "ablation_type_audit_logistic_vs_final_cnn.csv"
    audit.to_csv(audit_path, index=False)
    doc_lines = [
        "# Logistic 消融与最终 CNN 消融审计",
        "",
        "结论：prompt 指定的根目录 `_rerun` 文件在当前仓库根目录中不存在；当前可找到的旧消融输出包括非 `_rerun` 版本和上一轮二次输出目录中的 table3/metrics/predictions。可找到的旧输出均属于 tabular Logistic Regression feature-subset support analysis，不是最终 residual-aware SSL-CNN 的 10-seed LOSO 消融主结果。",
        "",
        "审计要点：",
        "",
        "1. 旧结果来自 tabular 模型流程，模型字段/脚本来源包含 Logistic Regression 或 SelectK 特征选择信息。",
        "2. 旧结果是表格特征 LOSO，不是每个 seed、每个 LOSO fold 都重新训练 CNN 的 10-seed final pipeline。",
        "3. 如果旧表的 `input_features` 出现 baseline_clinical，这是复用临床模块留下的字符串；旧消融脚本实际加载的是 PSD/WPLI EEG 表格特征。",
        "4. 旧结果可以保留为 supplementary tabular feature-subset support analysis，但不能作为 Results 3.3 的最终 residual-aware SSL-CNN 主消融。",
        "",
        to_markdown(audit),
    ]
    (docs_dir / "ablation_type_audit_logistic_vs_final_cnn.md").write_text("\n".join(doc_lines), encoding="utf-8")
    note_lines = [
        "# Tabular Logistic Feature-Subset Support Analysis Note",
        "",
        "旧 `modality_state_band_ablation_rerun.csv` 是 tabular Logistic Regression 消融。",
        "",
        "它的 `full_psd_wpli` performance 反映的是高维表格 EEG 特征在线性模型和 fold-local SelectK 下的表现，不能与最终 residual-aware SSL-CNN 的 `full_psd_wpli` 直接等价比较。",
        "",
        "如果保留在论文中，建议只放入 Supplementary Table，并命名为 tabular feature-subset support analysis。",
    ]
    (docs_dir / "tabular_logistic_feature_subset_support_analysis_note.md").write_text("\n".join(note_lines), encoding="utf-8")


def load_residual_module():
    script_path = PROJECT_ROOT / "scripts" / "30_train_residual_aware_patient_barlow.py"
    spec = spec_from_file_location("residual_aware_patient_barlow_for_final_ablation", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def probability_to_logit(probability: float, eps: float = 1e-6) -> float:
    clipped = min(max(float(probability), eps), 1.0 - eps)
    return float(np.log(clipped / (1.0 - clipped)))


def to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
