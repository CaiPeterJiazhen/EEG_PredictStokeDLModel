from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
import torch

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.explainability.attribution import (
    classification_logit,
    classification_probability,
    manual_integrated_gradients,
    smoothgrad_integrated_gradients,
)
from eeg_recovery.explainability.occlusion import clone_tensor_batch
from eeg_recovery.features.connectivity import build_edge_list, connectivity_bands
from eeg_recovery.features.feature_tables import load_fc_feature_table, merge_feature_tables
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.residual_targets import compute_signed_distance_from_label_table
from eeg_recovery.training.train_supervised import _make_batch, load_supervised_feature_records, resolve_device


SEEDS_10 = (0, 1, 2, 3, 4, 5, 7, 13, 21, 42)
MODEL_GROUP = "residualaware_highrank_swa_clsalpha1"
STATES = ("EO", "EC")
PSD_BANDS = (
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta Low", 13.0, 18.0),
    ("Beta Medium", 18.0, 21.0),
    ("Beta High", 21.0, 30.0),
    ("Gamma", 30.0, 45.5),
)
WPLI_BANDS = tuple(name for name, _ in connectivity_bands())
MOTOR_CHANNELS = {
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
}
FRIENDLY_NAMES = {
    "full_psd_wpli": "PSD + WPLI",
    "psd_only": "PSD only",
    "wpli_only": "WPLI only",
    "eo_only": "EO only",
    "ec_only": "EC only",
    "beta_medium_beta_high": "Beta medium + high",
    "motor_wpli_edges_only": "Motor WPLI edges",
    "occlude_psd_branch": "PSD",
    "occlude_wpli_branch": "WPLI",
    "occlude_eo_state": "EO",
    "occlude_ec_state": "EC",
    "occlude_motor_wpli_edges": "Motor WPLI edges",
    "occlude_non_motor_wpli_edges": "Non-motor WPLI edges",
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    family: str
    description: str
    keep_psd: bool = True
    keep_wpli: bool = True
    states: tuple[str, ...] = STATES
    psd_bands: tuple[str, ...] | None = None
    wpli_bands: tuple[str, ...] | None = None
    exclude_psd_bands: tuple[str, ...] = ()
    exclude_wpli_bands: tuple[str, ...] = ()
    motor_wpli: str | None = None
    custom: str | None = None


def psd_band_label(frequency_hz: float) -> str:
    for name, low, high in PSD_BANDS:
        if low <= float(frequency_hz) < high:
            return name
    return "Outside_0p5_45Hz"


def make_ablation_specs() -> list[AblationSpec]:
    six_bands = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High")
    specs = [
        AblationSpec("full_psd_wpli", "modality", "PSD + WPLI, EO + EC, full input"),
        AblationSpec("psd_only", "modality", "PSD only, EO + EC", keep_wpli=False),
        AblationSpec("wpli_only", "modality", "WPLI only, EO + EC", keep_psd=False),
        AblationSpec("eo_only", "state", "EO only, PSD + WPLI", states=("EO",)),
        AblationSpec("ec_only", "state", "EC only, PSD + WPLI", states=("EC",)),
        AblationSpec("psd_eo_only", "state", "PSD EO only", keep_wpli=False, states=("EO",)),
        AblationSpec("psd_ec_only", "state", "PSD EC only", keep_wpli=False, states=("EC",)),
        AblationSpec("wpli_eo_only", "state", "WPLI EO only", keep_psd=False, states=("EO",)),
        AblationSpec("wpli_ec_only", "state", "WPLI EC only", keep_psd=False, states=("EC",)),
        AblationSpec("psd_eo_wpli_ec", "state", "PSD EO + WPLI EC", custom="psd_eo_wpli_ec"),
        AblationSpec("psd_ec_wpli_eo", "state", "PSD EC + WPLI EO", custom="psd_ec_wpli_eo"),
    ]
    for band in six_bands:
        slug = _slug(band)
        specs.append(AblationSpec(f"{slug}_only", "band_only", f"PSD + WPLI {band} only", psd_bands=(band,), wpli_bands=(band,)))
    specs.extend(
        [
            AblationSpec(
                "beta_medium_beta_high",
                "band_only",
                "PSD/WPLI beta medium + beta high",
                psd_bands=("Beta Medium", "Beta High"),
                wpli_bands=("Beta Medium", "Beta High"),
            ),
            AblationSpec("psd_gamma_only", "band_only", "PSD Gamma only", keep_wpli=False, psd_bands=("Gamma",)),
        ]
    )
    for band in six_bands:
        specs.append(
            AblationSpec(
                f"full_minus_{_slug(band)}",
                "leave_one_band_out",
                f"Full input minus {band}",
                exclude_psd_bands=(band,),
                exclude_wpli_bands=(band,),
            )
        )
    specs.append(
        AblationSpec(
            "full_minus_psd_gamma",
            "leave_one_band_out",
            "Full input minus PSD Gamma",
            exclude_psd_bands=("Gamma",),
        )
    )
    specs.extend(
        [
            AblationSpec("motor_wpli_edges_only", "motor_network", "Motor-related WPLI edges only", keep_psd=False, motor_wpli="only_motor"),
            AblationSpec("non_motor_wpli_edges_only", "motor_network", "Non-motor WPLI edges only", keep_psd=False, motor_wpli="only_non_motor"),
            AblationSpec("full_minus_motor_wpli_edges", "motor_network", "Full input minus motor-related WPLI edges", motor_wpli="remove_motor"),
            AblationSpec("full_minus_non_motor_wpli_edges", "motor_network", "Full input minus non-motor WPLI edges", motor_wpli="remove_non_motor"),
        ]
    )
    return specs


def select_feature_columns(columns: Sequence[str], spec: AblationSpec) -> list[str]:
    selected = ["subject_id"]
    for column in columns:
        if column == "subject_id":
            continue
        parsed = parse_feature_column(column)
        if parsed is None:
            continue
        family, state, band, first, second = parsed
        if spec.custom == "psd_eo_wpli_ec":
            if family == "psd" and state == "EO":
                selected.append(column)
            elif family == "wpli" and state == "EC":
                selected.append(column)
            continue
        if spec.custom == "psd_ec_wpli_eo":
            if family == "psd" and state == "EC":
                selected.append(column)
            elif family == "wpli" and state == "EO":
                selected.append(column)
            continue
        if state not in spec.states:
            continue
        if family == "psd":
            if not spec.keep_psd:
                continue
            if spec.psd_bands is not None and band not in spec.psd_bands:
                continue
            if band in spec.exclude_psd_bands:
                continue
            selected.append(column)
        elif family == "wpli":
            if not spec.keep_wpli:
                continue
            if spec.wpli_bands is not None and band not in spec.wpli_bands:
                continue
            if band in spec.exclude_wpli_bands:
                continue
            is_motor = bool(first in MOTOR_CHANNELS or second in MOTOR_CHANNELS)
            if spec.motor_wpli == "only_motor" and not is_motor:
                continue
            if spec.motor_wpli == "only_non_motor" and is_motor:
                continue
            if spec.motor_wpli == "remove_motor" and is_motor:
                continue
            if spec.motor_wpli == "remove_non_motor" and not is_motor:
                continue
            selected.append(column)
    return selected


def parse_feature_column(column: str) -> tuple[str, str, str, str | None, str | None] | None:
    if column.startswith("psd_"):
        parts = column.split("_")
        if len(parts) < 4:
            return None
        state = parts[1]
        channel = parts[2]
        band = _band_from_slug("_".join(parts[3:]))
        return ("psd", state, band, channel, None)
    if column.startswith("fc_wpli_"):
        parts = column.split("_")
        if len(parts) < 6:
            return None
        state = parts[2]
        pair = parts[4]
        if "-" not in pair:
            return None
        first, second = pair.split("-", 1)
        band = _band_from_slug("_".join(parts[5:]))
        return ("wpli", state, band, first, second)
    return None


def checkpoint_audit_row(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    model_group = str(metadata.get("model_group", path.parent.name))
    payload_keys = set(payload.keys())
    metadata_keys = set(metadata.keys())
    has_swa_key = any("swa" in key.lower() for key in payload_keys | metadata_keys)
    use_swa_value = metadata.get("use_swa", metadata.get("swa", None))
    name_mentions_swa = "swa" in model_group.lower() or "swa" in path.name.lower()
    selected_alpha = metadata.get("selected_alpha", metadata.get("residual_alpha", np.nan))
    residual_alpha = metadata.get("residual_alpha", selected_alpha)
    state_dict = payload.get("state_dict", {})
    classification_head_available = any("classification_head" in str(key) for key in getattr(state_dict, "keys", lambda: [])())
    uses_swa_confirmed = bool(use_swa_value is True or name_mentions_swa or has_swa_key)
    missing_items = []
    if use_swa_value is not True and not name_mentions_swa:
        missing_items.append("explicit_use_swa")
    if not _float_is_close(residual_alpha, 1.0) or not _float_is_close(selected_alpha, 1.0):
        missing_items.append("residual_alpha_1.0")
    if not classification_head_available:
        missing_items.append("classification_head_state_dict")
    return {
        "checkpoint_path": str(path),
        "model_group": model_group,
        "model_name": metadata.get("model_name", ""),
        "seed": int(metadata.get("seed", -1)),
        "fold_index": int(metadata.get("fold_index", -1)),
        "test_subject_id": normalize_subject_id(metadata.get("test_subject_id", "")),
        "checkpoint_type": payload.get("checkpoint_type", ""),
        "metadata_use_swa": use_swa_value,
        "metadata_swa_lr": metadata.get("swa_lr", ""),
        "metadata_swa_start_epoch": metadata.get("swa_start_epoch", ""),
        "name_mentions_swa": bool(name_mentions_swa),
        "has_swa_metadata_or_payload_key": bool(has_swa_key),
        "uses_swa_confirmed": uses_swa_confirmed,
        "residual_alpha": residual_alpha,
        "selected_alpha": selected_alpha,
        "residual_alpha_is_1": bool(_float_is_close(residual_alpha, 1.0) and _float_is_close(selected_alpha, 1.0)),
        "classification_head_available": bool(classification_head_available),
        "classification_head_inference_confirmed": bool(classification_head_available),
        "state_scaler_available": "state_scaler" in payload,
        "actual_loaded_weight_source": "state_dict in checkpoint payload; metadata/use_swa or model-group name identifies SWA final checkpoint",
        "missing_or_uncertain_items": "; ".join(missing_items),
    }


def normalize_attributions_by_sample(frame: pd.DataFrame, sample_cols: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    max_abs = result.groupby(list(sample_cols))["signed_attribution"].transform(lambda s: max(float(np.max(np.abs(s))), 1e-12))
    result["norm_signed_attribution"] = result["signed_attribution"] / max_abs
    result["norm_abs_attribution"] = result["norm_signed_attribution"].abs()
    return result


def summarize_normalized_attribution(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    return (
        frame.groupby(list(group_cols), as_index=False)
        .agg(
            mean_signed_attribution=("norm_signed_attribution", "mean"),
            mean_abs_attribution=("norm_abs_attribution", "mean"),
            sd_abs_attribution=("norm_abs_attribution", "std"),
            n_subjects=("subject_id", "nunique") if "subject_id" in frame.columns else ("norm_abs_attribution", "size"),
        )
        .sort_values("mean_abs_attribution", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Secondary rerun for ablation and explainability outputs.")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--models", nargs="+", default=["logistic_l2"])
    parser.add_argument("--feature-selection", choices=("none", "selectk100"), default="selectk100")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--recompute-attributions", action="store_true")
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--smoothgrad-samples", type=int, default=8)
    parser.add_argument("--smoothgrad-noise-std", type=float, default=0.02)
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument("--skip-explainability-summary", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve()
    _ensure_output_dirs(artifact_root)
    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    labels = labels.copy()
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    labels = labels.sort_values("subject_id").reset_index(drop=True)

    audit = audit_swa_checkpoints(source_root, artifact_root)
    if not args.skip_tabular:
        metrics, predictions = run_feature_subset_ablation(
            source_root=source_root,
            artifact_root=artifact_root,
            labels=labels,
            models=args.models,
            feature_selection=args.feature_selection,
            random_state=args.random_state,
        )
        write_ablation_tables_docs_and_figures(artifact_root, metrics, predictions)
    if not args.skip_occlusion:
        occlusion, occlusion_summary, explained = run_final_model_occlusion(
            source_root=source_root,
            artifact_root=artifact_root,
            path_config=path_config,
            labels=labels,
            device_name=args.device,
        )
        write_occlusion_figure(artifact_root, occlusion_summary)
    if not args.skip_explainability_summary:
        if args.recompute_attributions:
            recompute_explainability_attributions(
                source_root=source_root,
                artifact_root=artifact_root,
                path_config=path_config,
                labels=labels,
                device_name=args.device,
                ig_steps=args.ig_steps,
                smoothgrad_samples=args.smoothgrad_samples,
                smoothgrad_noise_std=args.smoothgrad_noise_std,
                n_permutations=args.n_permutations,
            )
        else:
            summarize_existing_explainability(
                source_root=source_root,
                artifact_root=artifact_root,
                path_config=path_config,
                labels=labels,
                n_permutations=args.n_permutations,
            )
        write_explainability_figures_and_docs(artifact_root)
    _write_run_manifest(artifact_root, audit)


def audit_swa_checkpoints(source_root: Path, artifact_root: Path) -> pd.DataFrame:
    checkpoint_dir = source_root / "results" / "checkpoints" / "supervised" / MODEL_GROUP
    rows = []
    for path in sorted(checkpoint_dir.glob(f"{MODEL_GROUP}_seed*_fold*_test_*.pt")):
        payload = _torch_load(path, torch.device("cpu"))
        rows.append(checkpoint_audit_row(path, payload))
    audit = pd.DataFrame(rows).sort_values(["seed", "fold_index"]).reset_index(drop=True)
    expected = {(seed, fold) for seed in SEEDS_10 for fold in range(19)}
    observed = {(int(row.seed), int(row.fold_index)) for row in audit.itertuples()} if not audit.empty else set()
    missing = sorted(expected - observed)
    if missing:
        audit = pd.concat(
            [
                audit,
                pd.DataFrame(
                    [
                        {
                            "checkpoint_path": "",
                            "model_group": MODEL_GROUP,
                            "seed": seed,
                            "fold_index": fold,
                            "uses_swa_confirmed": False,
                            "residual_alpha_is_1": False,
                            "classification_head_available": False,
                            "missing_or_uncertain_items": "checkpoint_missing",
                        }
                        for seed, fold in missing
                    ]
                ),
            ],
            ignore_index=True,
        )
    metrics_path = artifact_root / "results" / "metrics" / "final_model_swa_checkpoint_audit.csv"
    explain_path = artifact_root / "results" / "explainability" / "explainability_swa_loading_audit.csv"
    audit.to_csv(metrics_path, index=False)
    audit.to_csv(explain_path, index=False)
    lines = [
        "# Final Model SWA Checkpoint Audit",
        "",
        f"- Model group: `{MODEL_GROUP}`",
        f"- Checkpoint rows found: {len(rows)}",
        f"- Expected seed/fold rows: {len(expected)}",
        f"- Rows with confirmed SWA: {int(audit['uses_swa_confirmed'].fillna(False).sum())}",
        f"- Rows with residual_alpha=1.0 and selected_alpha=1.0: {int(audit['residual_alpha_is_1'].fillna(False).sum())}",
        f"- Rows with classification head weights: {int(audit['classification_head_available'].fillna(False).sum())}",
        "",
        "The checkpoint payloads store a `state_dict` that is loaded directly for inference. A row is marked as SWA-confirmed when `use_swa=True`, an SWA key is present, or the final model group/path name contains `swa`.",
        "",
        "## Missing Or Uncertain Rows",
        "",
    ]
    uncertain = audit[audit["missing_or_uncertain_items"].fillna("").astype(str) != ""]
    lines.append(_to_markdown(uncertain[["seed", "fold_index", "test_subject_id", "missing_or_uncertain_items", "checkpoint_path"]].head(80)))
    (artifact_root / "docs" / "final_model_swa_checkpoint_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (artifact_root / "docs" / "explainability_swa_loading_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def run_feature_subset_ablation(
    *,
    source_root: Path,
    artifact_root: Path,
    labels: pd.DataFrame,
    models: Sequence[str],
    feature_selection: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = _load_script_module(source_root / "scripts" / "39_train_clinical_and_incremental_baselines.py", "clinical_for_secondary_ablation")
    subjects = labels["subject_id"].tolist()
    psd = build_psd_gamma_band_power_table(source_root / "data" / "features" / "psd", subjects)
    wpli = load_fc_feature_table(source_root / "data" / "features" / "fc", metric="wpli", subject_ids=subjects)
    full = merge_feature_tables(psd, wpli)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for spec in make_ablation_specs():
        columns = select_feature_columns(list(full.columns), spec)
        input_columns = [column for column in columns if column != "subject_id"]
        if not input_columns:
            metric_frames.append(
                pd.DataFrame(
                    [
                        {
                            "ablation_name": spec.name,
                            "ablation_family": spec.family,
                            "input_description": spec.description,
                            "status": "skipped",
                            "skip_reason": "No columns matched this ablation.",
                            "n_input_features": 0,
                        }
                    ]
                )
            )
            continue
        predictions, metrics = clinical.run_loso_tabular_models(
            full.loc[:, columns].copy(),
            label_table=labels[["subject_id", "label"]],
            input_columns=input_columns,
            model_names=models,
            feature_selection=feature_selection,
            random_state=random_state,
            model_family=f"ablation_{spec.name}",
        )
        predictions["ablation_name"] = spec.name
        predictions["ablation_family"] = spec.family
        predictions["ablation_description"] = spec.description
        metrics["ablation_name"] = spec.name
        metrics["ablation_family"] = spec.family
        metrics["input_description"] = spec.description
        metrics["n_input_features"] = len(input_columns)
        metrics["uses_swa"] = False
        metrics["model_or_evaluation_type"] = "feature-subset LOSO tabular logistic_l2"
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    metrics_all = pd.concat(metric_frames, ignore_index=True, sort=False)
    predictions_all = pd.concat(prediction_frames, ignore_index=True, sort=False)
    metrics_all.to_csv(artifact_root / "results" / "metrics" / "modality_state_band_ablation_rerun.csv", index=False)
    predictions_all.to_csv(artifact_root / "results" / "predictions" / "modality_state_band_ablation_predictions_rerun.csv", index=False)
    return metrics_all, predictions_all


def build_psd_gamma_band_power_table(psd_dir: Path, subject_ids: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject_id = normalize_subject_id(subject_id)
        row: dict[str, Any] = {"subject_id": subject_id}
        for state in STATES:
            path = psd_dir / f"{subject_id}_{state}_psd.npz"
            with np.load(path, allow_pickle=False) as payload:
                psd = np.asarray(payload["psd"], dtype=float)
                frequencies = np.asarray(payload["frequency_bins"], dtype=float)
            for band, low, high in PSD_BANDS:
                mask = (frequencies >= low) & (frequencies < high)
                if not mask.any():
                    continue
                values = psd[:, mask].mean(axis=1)
                for channel, value in zip(CANONICAL_CHANNELS_62, values, strict=True):
                    row[f"psd_{state}_{channel}_{_label_slug(band)}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def write_ablation_tables_docs_and_figures(artifact_root: Path, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best = _best_metric_row_per_ablation(metrics)
    full = best[best["ablation_name"] == "full_psd_wpli"].head(1)
    reference = full.iloc[0].to_dict() if not full.empty else {}
    table = best.copy()
    table["uses_swa"] = table.get("uses_swa", False).fillna(False)
    table["model_or_evaluation_type"] = table.get("model_or_evaluation_type", "feature-subset LOSO tabular logistic_l2")
    table["reference_ablation"] = "full_psd_wpli"
    for metric in ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"]:
        table[f"delta_{metric}"] = table[metric] - float(reference.get(metric, np.nan))
    table["delta_brier"] = table["brier_score"] - float(reference.get("brier_score", np.nan))
    table["brier"] = table["brier_score"]
    table["interpretation_for_paper"] = table.apply(_ablation_interpretation, axis=1)
    columns = [
        "ablation_name",
        "ablation_family",
        "input_description",
        "model_or_evaluation_type",
        "uses_swa",
        "n_subjects",
        "n_input_features",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier",
        "reference_ablation",
        "delta_accuracy",
        "delta_balanced_accuracy",
        "delta_roc_auc",
        "delta_pr_auc",
        "delta_brier",
        "interpretation_for_paper",
    ]
    table["n_seeds"] = 1
    columns.insert(6, "n_seeds")
    table.loc[:, columns].to_csv(artifact_root / "results" / "tables" / "table3_feature_state_band_ablation_for_paper.csv", index=False)
    (artifact_root / "results" / "tables" / "table3_feature_state_band_ablation_for_paper.md").write_text(_to_markdown(table.loc[:, columns]), encoding="utf-8")
    delta = table[["ablation_name", "reference_ablation", "delta_accuracy", "delta_balanced_accuracy", "delta_roc_auc", "delta_pr_auc", "delta_brier"]].copy()
    delta.to_csv(artifact_root / "results" / "metrics" / "modality_state_band_ablation_delta_vs_full.csv", index=False)
    _save_figure5c(artifact_root, table)
    _save_figure5d(artifact_root, table)
    _write_ablation_doc(artifact_root, table)


def run_final_model_occlusion(
    *,
    source_root: Path,
    artifact_root: Path,
    path_config: Any,
    labels: pd.DataFrame,
    device_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    explain_mod = _load_script_module(source_root / "scripts" / "31_explain_residual_aware_ssl_cnn.py", "explain_for_secondary_occlusion")
    labels = labels.copy()
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    label_table = labels.set_index("subject_id", drop=False)
    targets = compute_signed_distance_from_label_table(labels, threshold=1.5).set_index("subject_id", drop=False)
    records = load_supervised_feature_records(path_config, labels, feature_kind="psd-fc-wpli")
    record_by_subject = {record.subject_id: record for record in records}
    device = resolve_device(device_name)
    edge_list = build_edge_list(CANONICAL_CHANNELS_62)
    frequency_bins = _load_frequency_bins(source_root, records[0].subject_id)
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    checkpoint_dir = source_root / "results" / "checkpoints" / "supervised" / MODEL_GROUP
    for seed in SEEDS_10:
        for checkpoint_path in sorted(checkpoint_dir.glob(f"{MODEL_GROUP}_seed{seed}_fold*_test_*.pt")):
            payload = _torch_load(checkpoint_path, device)
            metadata = payload["metadata"]
            subject_id = normalize_subject_id(metadata["test_subject_id"])
            model = explain_mod._build_model_from_metadata(metadata, payload["state_dict"], device)
            scaler = explain_mod._scaler_from_payload(payload["state_scaler"])
            batch = _make_batch([record_by_subject[subject_id]], scaler, device, "multimodal")
            y_true = int(label_table.loc[subject_id, "label"])
            residual = float(label_table.loc[subject_id, "Residual"])
            signed_distance = float(targets.loc[subject_id, "signed_distance"])
            with torch.no_grad():
                original_logit = float(classification_logit(model, batch).reshape(-1).item())
                original_prob = float(classification_probability(model, batch).reshape(-1).item())
            y_pred = int(original_prob >= 0.5)
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "fold_index": int(metadata["fold_index"]),
                    "seed": int(metadata["seed"]),
                    "y_true": y_true,
                    "y_score": original_prob,
                    "y_pred": y_pred,
                    "correct": int(y_pred == y_true),
                    "residual": residual,
                    "signed_distance": signed_distance,
                }
            )
            specs = make_occlusion_specs(frequency_bins, edge_list)
            occluded_batches = []
            for _, _, occlude in specs:
                occluded = clone_tensor_batch(batch)
                occlude(occluded)
                occluded_batches.append(occluded)
            stacked = {key: torch.cat([item[key] for item in occluded_batches], dim=0) for key in batch}
            with torch.no_grad():
                logits = classification_logit(model, stacked).reshape(-1).detach().cpu().numpy()
                probs = classification_probability(model, stacked).reshape(-1).detach().cpu().numpy()
            for index, (name, family, _) in enumerate(specs):
                prob = float(probs[index])
                logit = float(logits[index])
                rows.append(
                    {
                        "subject_id": subject_id,
                        "fold_index": int(metadata["fold_index"]),
                        "seed": int(metadata["seed"]),
                        "occlusion_name": name,
                        "occlusion_family": family,
                        "uses_swa": True,
                        "y_true": y_true,
                        "original_prob": original_prob,
                        "occluded_prob": prob,
                        "original_logit": original_logit,
                        "occluded_logit": logit,
                        "original_pred_label": y_pred,
                        "occluded_pred_label": int(prob >= 0.5),
                        "probability_drop": original_prob - prob,
                        "logit_drop": original_logit - logit,
                    }
                )
    occlusion = pd.DataFrame(rows)
    explained = pd.DataFrame(prediction_rows).drop_duplicates(["subject_id", "seed", "fold_index"])
    summary = summarize_occlusion(occlusion)
    occlusion.to_csv(artifact_root / "results" / "explainability" / "final_model_occlusion_modality_state_band.csv", index=False)
    summary.to_csv(artifact_root / "results" / "explainability" / "final_model_occlusion_modality_state_band_summary.csv", index=False)
    explained.to_csv(artifact_root / "results" / "explainability" / "explained_predictions.csv", index=False)
    _write_occlusion_aliases(artifact_root, occlusion)
    return occlusion, summary, explained


def make_occlusion_specs(frequency_bins: np.ndarray, edge_list: Sequence[tuple[str, str]]) -> list[tuple[str, str, Any]]:
    specs: list[tuple[str, str, Any]] = [
        ("occlude_psd_branch", "modality", lambda batch: _zero_keys(batch, ("psd_eo", "psd_ec"))),
        ("occlude_wpli_branch", "modality", lambda batch: _zero_keys(batch, ("wpli_eo", "wpli_ec"))),
        ("occlude_eo_state", "state", lambda batch: _zero_keys(batch, ("psd_eo", "wpli_eo"))),
        ("occlude_ec_state", "state", lambda batch: _zero_keys(batch, ("psd_ec", "wpli_ec"))),
    ]
    for band, _, _ in PSD_BANDS:
        specs.append((f"occlude_psd_{_slug(band)}", "psd_band", lambda batch, band=band: _zero_psd_band(batch, band, frequency_bins)))
    for band in WPLI_BANDS:
        specs.append((f"occlude_wpli_{_slug(band)}", "wpli_band", lambda batch, band=band: _zero_wpli_band(batch, band)))
    specs.extend(
        [
            ("occlude_motor_wpli_edges", "motor_network", lambda batch: _zero_motor_edges(batch, edge_list, want_motor=True)),
            ("occlude_non_motor_wpli_edges", "motor_network", lambda batch: _zero_motor_edges(batch, edge_list, want_motor=False)),
        ]
    )
    return specs


def summarize_occlusion(occlusion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, family), group in occlusion.groupby(["occlusion_name", "occlusion_family"], sort=False):
        subject = (
            group.groupby("subject_id", as_index=False)
            .agg(
                y_true=("y_true", "first"),
                original_prob=("original_prob", "mean"),
                occluded_prob=("occluded_prob", "mean"),
                probability_drop=("probability_drop", "mean"),
                original_logit=("original_logit", "mean"),
                occluded_logit=("occluded_logit", "mean"),
                logit_drop=("logit_drop", "mean"),
            )
            .sort_values("subject_id")
        )
        metrics = binary_classification_metrics(subject["y_true"].to_numpy(int), subject["occluded_prob"].to_numpy(float))
        rows.append(
            {
                "occlusion_name": name,
                "occlusion_family": family,
                "uses_swa": True,
                "n_subjects": int(subject["subject_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "mean_original_prob": float(subject["original_prob"].mean()),
                "mean_occluded_prob": float(subject["occluded_prob"].mean()),
                "mean_probability_drop": float(subject["probability_drop"].mean()),
                "sd_probability_drop": float(subject["probability_drop"].std(ddof=1)),
                "mean_original_logit": float(subject["original_logit"].mean()),
                "mean_occluded_logit": float(subject["occluded_logit"].mean()),
                "mean_logit_drop": float(subject["logit_drop"].mean()),
                "sd_logit_drop": float(subject["logit_drop"].std(ddof=1)),
                "accuracy_after_occlusion": metrics["accuracy"],
                "balanced_accuracy_after_occlusion": metrics["balanced_accuracy"],
                "roc_auc_after_occlusion": metrics["roc_auc"],
                "pr_auc_after_occlusion": metrics["pr_auc"],
                "brier_after_occlusion": metrics["brier_score"],
                "interpretation_for_paper": _occlusion_interpretation(name, float(subject["probability_drop"].mean())),
            }
        )
    return pd.DataFrame(rows)


def summarize_existing_explainability(
    *,
    source_root: Path,
    artifact_root: Path,
    path_config: Any,
    labels: pd.DataFrame,
    n_permutations: int,
) -> None:
    source_explain = source_root / "results" / "explainability"
    target_explain = artifact_root / "results" / "explainability"
    for filename in ("branch_state_gate_weights.csv",):
        source = source_explain / filename
        if source.exists():
            shutil.copy2(source, target_explain / filename)
    psd_long = source_explain / "psd_attribution_long.csv"
    wpli_long = source_explain / "wpli_edge_attribution_long.csv"
    sample_max = _sample_max_abs([psd_long, wpli_long])
    psd_outputs = _summarize_psd_long(psd_long, sample_max)
    wpli_outputs = _summarize_wpli_long(wpli_long, sample_max)
    psd_outputs["channel_band"].to_csv(target_explain / "psd_channel_band_importance.csv", index=False)
    psd_outputs["frequency"].to_csv(target_explain / "psd_frequency_importance.csv", index=False)
    psd_outputs["channel_frequency"].to_csv(target_explain / "psd_channel_frequency_top_features.csv", index=False)
    wpli_outputs["edges"].to_csv(target_explain / "wpli_top_edges.csv", index=False)
    wpli_outputs["band"].to_csv(target_explain / "wpli_band_importance.csv", index=False)
    wpli_outputs["node"].to_csv(target_explain / "wpli_node_importance.csv", index=False)
    wpli_outputs["network"].to_csv(target_explain / "wpli_network_group_importance.csv", index=False)
    stability = _stability_from_seed_tables(psd_outputs["psd_seed"], wpli_outputs["wpli_seed"])
    stability.to_csv(target_explain / "attribution_stability_summary.csv", index=False)
    records = load_supervised_feature_records(path_config, labels, feature_kind="psd-fc-wpli")
    record_by_subject = {record.subject_id: record for record in records}
    frequency_bins = _load_frequency_bins(source_root, records[0].subject_id)
    edge_list = build_edge_list(CANONICAL_CHANNELS_62)
    targets = compute_signed_distance_from_label_table(labels, threshold=1.5).set_index("subject_id", drop=False)
    label_table = labels.set_index("subject_id", drop=False)
    stats = write_feature_statistics(
        artifact_root=artifact_root,
        record_by_subject=record_by_subject,
        label_table=label_table,
        targets=targets,
        frequency_bins=frequency_bins,
        edge_list=edge_list,
        n_permutations=n_permutations,
    )
    write_table4(artifact_root, stats)
    _write_explainability_method_note(artifact_root)


def recompute_explainability_attributions(
    *,
    source_root: Path,
    artifact_root: Path,
    path_config: Any,
    labels: pd.DataFrame,
    device_name: str,
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_noise_std: float,
    n_permutations: int,
) -> None:
    explain_mod = _load_script_module(source_root / "scripts" / "31_explain_residual_aware_ssl_cnn.py", "explain_for_secondary_attribution")
    source_explain = source_root / "results" / "explainability"
    target_explain = artifact_root / "results" / "explainability"
    gate_path = source_explain / "branch_state_gate_weights.csv"
    if gate_path.exists():
        shutil.copy2(gate_path, target_explain / "branch_state_gate_weights.csv")

    labels = labels.copy()
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    records = load_supervised_feature_records(path_config, labels, feature_kind="psd-fc-wpli")
    record_by_subject = {record.subject_id: record for record in records}
    subjects = sorted(record_by_subject)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    seed_index = {seed: index for index, seed in enumerate(SEEDS_10)}
    device = resolve_device(device_name)
    frequency_bins = _load_frequency_bins(source_root, records[0].subject_id)
    edge_list = build_edge_list(CANONICAL_CHANNELS_62)

    meta = _attribution_feature_metadata(frequency_bins, edge_list)
    n_subjects = len(subjects)
    n_seeds = len(SEEDS_10)
    accum = {
        name: _new_accumulator(n_subjects, len(items))
        for name, items in {
            "psd_channel_band": meta["psd_channel_band"],
            "psd_frequency": meta["psd_frequency"],
            "psd_channel_frequency": meta["psd_channel_frequency"],
            "wpli_edges": meta["wpli_edges"],
            "wpli_band": meta["wpli_band"],
        }.items()
    }
    seed_accum = {
        "PSD": _new_accumulator(n_seeds, len(meta["psd_channel_band"])),
        "WPLI": _new_accumulator(n_seeds, len(meta["wpli_edges"])),
    }

    checkpoint_dir = source_root / "results" / "checkpoints" / "supervised" / MODEL_GROUP
    prediction_rows: list[dict[str, Any]] = []
    for seed in SEEDS_10:
        for checkpoint_path in sorted(checkpoint_dir.glob(f"{MODEL_GROUP}_seed{seed}_fold*_test_*.pt")):
            payload = _torch_load(checkpoint_path, device)
            metadata = payload["metadata"]
            subject_id = normalize_subject_id(metadata["test_subject_id"])
            model = explain_mod._build_model_from_metadata(metadata, payload["state_dict"], device)
            scaler = explain_mod._scaler_from_payload(payload["state_scaler"])
            batch = _make_batch([record_by_subject[subject_id]], scaler, device, "multimodal")
            with torch.no_grad():
                prob = float(classification_probability(model, batch).reshape(-1).item())
            label_row = labels[labels["subject_id"] == subject_id].iloc[0]
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "fold_index": int(metadata["fold_index"]),
                    "seed": int(seed),
                    "y_true": int(label_row["label"]),
                    "y_score": prob,
                    "y_pred": int(prob >= 0.5),
                    "correct": int((prob >= 0.5) == bool(int(label_row["label"]))),
                    "residual": float(label_row["Residual"]),
                    "signed_distance": 1.5 - float(label_row["Residual"]),
                }
            )
            attrs_ig = manual_integrated_gradients(model, batch, input_keys=("psd_eo", "psd_ec", "wpli_eo", "wpli_ec"), steps=ig_steps)
            attrs_sg = smoothgrad_integrated_gradients(
                model,
                batch,
                input_keys=("psd_eo", "psd_ec", "wpli_eo", "wpli_ec"),
                steps=ig_steps,
                noise_samples=smoothgrad_samples,
                noise_std=smoothgrad_noise_std,
                seed=int(seed),
            )
            combined = _combined_normalized_attributions(attrs_ig, attrs_sg)
            subject_i = subject_index[subject_id]
            seed_i = seed_index[int(seed)]
            vectors = _attribution_vectors(combined, frequency_bins)
            for name, (signed, absolute) in vectors.items():
                _accumulate(accum[name], subject_i, signed, absolute)
            _accumulate(seed_accum["PSD"], seed_i, vectors["psd_channel_band"][0], vectors["psd_channel_band"][1])
            _accumulate(seed_accum["WPLI"], seed_i, vectors["wpli_edges"][0], vectors["wpli_edges"][1])

    psd_channel_band = _summary_from_accumulator(accum["psd_channel_band"], meta["psd_channel_band"])
    psd_frequency = _summary_from_accumulator(accum["psd_frequency"], meta["psd_frequency"])
    psd_channel_frequency = _summary_from_accumulator(accum["psd_channel_frequency"], meta["psd_channel_frequency"])
    wpli_edges = _summary_from_accumulator(accum["wpli_edges"], meta["wpli_edges"])
    wpli_band = _summary_from_accumulator(accum["wpli_band"], meta["wpli_band"])
    wpli_node = _wpli_node_from_edges(wpli_edges)
    wpli_network = (
        wpli_edges.groupby(["state", "network_group", "band"], as_index=False)
        .agg(
            mean_abs_attribution=("mean_abs_attribution", "mean"),
            mean_signed_attribution=("mean_signed_attribution", "mean"),
            sd_abs_attribution=("mean_abs_attribution", "std"),
            n_edges=("edge_index", "count"),
        )
        .sort_values("mean_abs_attribution", ascending=False)
        .reset_index(drop=True)
    )
    psd_channel_band.to_csv(target_explain / "psd_channel_band_importance.csv", index=False)
    psd_frequency.to_csv(target_explain / "psd_frequency_importance.csv", index=False)
    psd_channel_frequency.to_csv(target_explain / "psd_channel_frequency_top_features.csv", index=False)
    wpli_edges.to_csv(target_explain / "wpli_top_edges.csv", index=False)
    wpli_band.to_csv(target_explain / "wpli_band_importance.csv", index=False)
    wpli_node.to_csv(target_explain / "wpli_node_importance.csv", index=False)
    wpli_network.to_csv(target_explain / "wpli_network_group_importance.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(target_explain / "explained_predictions.csv", index=False)
    psd_seed = _seed_frame_from_accumulator(seed_accum["PSD"], meta["psd_channel_band"], "PSD", SEEDS_10)
    wpli_seed = _seed_frame_from_accumulator(seed_accum["WPLI"], meta["wpli_edges"], "WPLI", SEEDS_10)
    _stability_from_seed_tables(psd_seed, wpli_seed).to_csv(target_explain / "attribution_stability_summary.csv", index=False)

    record_by_subject = {record.subject_id: record for record in records}
    targets = compute_signed_distance_from_label_table(labels, threshold=1.5).set_index("subject_id", drop=False)
    label_table = labels.set_index("subject_id", drop=False)
    stats = write_feature_statistics(
        artifact_root=artifact_root,
        record_by_subject=record_by_subject,
        label_table=label_table,
        targets=targets,
        frequency_bins=frequency_bins,
        edge_list=edge_list,
        n_permutations=n_permutations,
    )
    write_table4(artifact_root, stats)
    _write_recomputed_explainability_method_note(artifact_root, ig_steps, smoothgrad_samples, smoothgrad_noise_std)


def _attribution_feature_metadata(frequency_bins: np.ndarray, edge_list: Sequence[tuple[str, str]]) -> dict[str, list[dict[str, Any]]]:
    psd_channel_band = []
    psd_frequency = []
    psd_channel_frequency = []
    for state in STATES:
        for channel_index, channel in enumerate(CANONICAL_CHANNELS_62):
            for band, _, _ in PSD_BANDS:
                psd_channel_band.append({"state": state, "channel": channel, "channel_index": channel_index, "band": band})
        for frequency_bin, frequency_hz in enumerate(frequency_bins):
            psd_frequency.append({"state": state, "frequency_hz": float(frequency_hz), "frequency_bin": frequency_bin, "band": psd_band_label(float(frequency_hz))})
        for channel_index, channel in enumerate(CANONICAL_CHANNELS_62):
            for frequency_bin, frequency_hz in enumerate(frequency_bins):
                psd_channel_frequency.append(
                    {
                        "state": state,
                        "channel": channel,
                        "channel_index": channel_index,
                        "frequency_hz": float(frequency_hz),
                        "frequency_bin": frequency_bin,
                        "band": psd_band_label(float(frequency_hz)),
                    }
                )
    psd_frequency = [row for row in psd_frequency if row["band"] != "Outside_0p5_45Hz"]
    psd_channel_frequency = [row for row in psd_channel_frequency if row["band"] != "Outside_0p5_45Hz"]
    wpli_edges = []
    wpli_band = []
    for state in STATES:
        for band_index, band in enumerate(WPLI_BANDS):
            wpli_band.append({"state": state, "band": band, "band_index": band_index})
        for edge_index, (first, second) in enumerate(edge_list):
            for band_index, band in enumerate(WPLI_BANDS):
                wpli_edges.append(
                    {
                        "state": state,
                        "edge_index": edge_index,
                        "channel_i": first,
                        "channel_j": second,
                        "channel_i_index": CANONICAL_CHANNELS_62.index(first),
                        "channel_j_index": CANONICAL_CHANNELS_62.index(second),
                        "band": band,
                        "band_index": band_index,
                        "network_group": _edge_network_group(first, second),
                        "interhemispheric": _is_interhemispheric(first, second),
                    }
                )
    return {
        "psd_channel_band": psd_channel_band,
        "psd_frequency": psd_frequency,
        "psd_channel_frequency": psd_channel_frequency,
        "wpli_edges": wpli_edges,
        "wpli_band": wpli_band,
    }


def _new_accumulator(n_rows: int, n_features: int) -> dict[str, np.ndarray]:
    return {
        "sum_signed": np.zeros((n_rows, n_features), dtype=np.float64),
        "sum_abs": np.zeros((n_rows, n_features), dtype=np.float64),
        "count": np.zeros((n_rows, n_features), dtype=np.float64),
    }


def _accumulate(accumulator: dict[str, np.ndarray], row_index: int, signed: np.ndarray, absolute: np.ndarray) -> None:
    accumulator["sum_signed"][row_index, :] += signed
    accumulator["sum_abs"][row_index, :] += absolute
    accumulator["count"][row_index, :] += 1.0


def _summary_from_accumulator(accumulator: dict[str, np.ndarray], metadata: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        subject_signed = np.divide(accumulator["sum_signed"], accumulator["count"], out=np.full_like(accumulator["sum_signed"], np.nan), where=accumulator["count"] > 0)
        subject_abs = np.divide(accumulator["sum_abs"], accumulator["count"], out=np.full_like(accumulator["sum_abs"], np.nan), where=accumulator["count"] > 0)
    rows = []
    for feature_index, meta in enumerate(metadata):
        values_abs = subject_abs[:, feature_index]
        values_signed = subject_signed[:, feature_index]
        rows.append(
            {
                **dict(meta),
                "mean_signed_attribution": float(np.nanmean(values_signed)),
                "mean_abs_attribution": float(np.nanmean(values_abs)),
                "sd_abs_attribution": float(np.nanstd(values_abs, ddof=1)),
                "n_subjects": int(np.isfinite(values_abs).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)


def _seed_frame_from_accumulator(accumulator: dict[str, np.ndarray], metadata: Sequence[Mapping[str, Any]], family: str, seeds: Sequence[int]) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        seed_abs = np.divide(accumulator["sum_abs"], accumulator["count"], out=np.zeros_like(accumulator["sum_abs"]), where=accumulator["count"] > 0)
    rows = []
    for seed_row, seed in enumerate(seeds):
        for feature_index, meta in enumerate(metadata):
            row = {**dict(meta), "seed": int(seed), "feature_family": family, "mean_abs_attribution": float(seed_abs[seed_row, feature_index])}
            row["feature_id"] = _feature_id_from_row(pd.Series(row), family)
            rows.append(row)
    return pd.DataFrame(rows)


def _combined_normalized_attributions(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    first_norm = _normalize_attr_dict(first)
    second_norm = _normalize_attr_dict(second)
    return {key: 0.5 * (first_norm[key] + second_norm[key]) for key in first_norm}


def _normalize_attr_dict(attrs: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    arrays = {key: value.detach().cpu().numpy()[0].astype(np.float64, copy=False) for key, value in attrs.items()}
    max_abs = max(float(np.max(np.abs(value))) for value in arrays.values())
    max_abs = max(max_abs, 1e-12)
    return {key: value / max_abs for key, value in arrays.items()}


def _attribution_vectors(attrs: Mapping[str, np.ndarray], frequency_bins: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    psd_cb_signed = []
    psd_cb_abs = []
    psd_freq_signed = []
    psd_freq_abs = []
    psd_cf_signed = []
    psd_cf_abs = []
    for key in ("psd_eo", "psd_ec"):
        arr = attrs[key]
        for _, low, high in PSD_BANDS:
            mask = (frequency_bins >= low) & (frequency_bins < high)
            psd_cb_signed.extend(arr[:, mask].mean(axis=1).tolist())
            psd_cb_abs.extend(np.abs(arr[:, mask]).mean(axis=1).tolist())
        valid_freq = np.asarray([psd_band_label(freq) != "Outside_0p5_45Hz" for freq in frequency_bins], dtype=bool)
        psd_freq_signed.extend(arr[:, valid_freq].mean(axis=0).tolist())
        psd_freq_abs.extend(np.abs(arr[:, valid_freq]).mean(axis=0).tolist())
        psd_cf_signed.extend(arr[:, valid_freq].reshape(-1).tolist())
        psd_cf_abs.extend(np.abs(arr[:, valid_freq]).reshape(-1).tolist())
    wpli_edge_signed = []
    wpli_edge_abs = []
    wpli_band_signed = []
    wpli_band_abs = []
    for key in ("wpli_eo", "wpli_ec"):
        arr = attrs[key]
        wpli_band_signed.extend(arr.mean(axis=0).tolist())
        wpli_band_abs.extend(np.abs(arr).mean(axis=0).tolist())
        wpli_edge_signed.extend(arr.reshape(-1).tolist())
        wpli_edge_abs.extend(np.abs(arr).reshape(-1).tolist())
    return {
        "psd_channel_band": (np.asarray(psd_cb_signed), np.asarray(psd_cb_abs)),
        "psd_frequency": (np.asarray(psd_freq_signed), np.asarray(psd_freq_abs)),
        "psd_channel_frequency": (np.asarray(psd_cf_signed), np.asarray(psd_cf_abs)),
        "wpli_edges": (np.asarray(wpli_edge_signed), np.asarray(wpli_edge_abs)),
        "wpli_band": (np.asarray(wpli_band_signed), np.asarray(wpli_band_abs)),
    }


def _sample_max_abs(paths: Sequence[Path], chunksize: int = 500_000) -> dict[tuple[str, int, int], float]:
    maxima: dict[tuple[str, int, int], float] = {}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["subject_id", "fold_index", "seed", "abs_attribution"], chunksize=chunksize):
            grouped = chunk.groupby(["subject_id", "fold_index", "seed"])["abs_attribution"].max()
            for key, value in grouped.items():
                key = (normalize_subject_id(key[0]), int(key[1]), int(key[2]))
                maxima[key] = max(maxima.get(key, 0.0), float(value))
    return {key: max(value, 1e-12) for key, value in maxima.items()}


def _summarize_psd_long(path: Path, sample_max: Mapping[tuple[str, int, int], float], chunksize: int = 500_000) -> dict[str, pd.DataFrame]:
    channel_band_parts = []
    frequency_parts = []
    channel_frequency_parts = []
    seed_parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk["subject_id"] = chunk["subject_id"].map(normalize_subject_id)
        chunk["band"] = chunk["frequency_hz"].map(psd_band_label)
        chunk = chunk[chunk["band"] != "Outside_0p5_45Hz"].copy()
        denom = [
            sample_max[(row.subject_id, int(row.fold_index), int(row.seed))]
            for row in chunk[["subject_id", "fold_index", "seed"]].itertuples(index=False)
        ]
        chunk["norm_signed_attribution"] = chunk["signed_attribution"].to_numpy(float) / np.asarray(denom, dtype=float)
        chunk["norm_abs_attribution"] = np.abs(chunk["norm_signed_attribution"])
        channel_band_parts.append(_partial_subject_summary(chunk, ["state", "channel", "channel_index", "band"]))
        frequency_parts.append(_partial_subject_summary(chunk, ["state", "frequency_hz", "frequency_bin", "band"]))
        channel_frequency_parts.append(_partial_subject_summary(chunk, ["state", "channel", "channel_index", "frequency_hz", "frequency_bin", "band"]))
        seed_parts.append(_partial_seed_summary(chunk, ["state", "channel", "channel_index", "band"], "PSD"))
    channel_band = _finish_subject_summary(channel_band_parts, ["state", "channel", "channel_index", "band"])
    frequency = _finish_subject_summary(frequency_parts, ["state", "frequency_hz", "frequency_bin", "band"])
    channel_frequency = _finish_subject_summary(channel_frequency_parts, ["state", "channel", "channel_index", "frequency_hz", "frequency_bin", "band"])
    return {
        "channel_band": channel_band,
        "frequency": frequency,
        "channel_frequency": channel_frequency,
        "psd_seed": _finish_seed_summary(seed_parts, ["state", "channel", "channel_index", "band"], "PSD"),
    }


def _summarize_wpli_long(path: Path, sample_max: Mapping[tuple[str, int, int], float], chunksize: int = 500_000) -> dict[str, pd.DataFrame]:
    edge_parts = []
    band_parts = []
    seed_parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk["subject_id"] = chunk["subject_id"].map(normalize_subject_id)
        denom = [
            sample_max[(row.subject_id, int(row.fold_index), int(row.seed))]
            for row in chunk[["subject_id", "fold_index", "seed"]].itertuples(index=False)
        ]
        chunk["norm_signed_attribution"] = chunk["signed_attribution"].to_numpy(float) / np.asarray(denom, dtype=float)
        chunk["norm_abs_attribution"] = np.abs(chunk["norm_signed_attribution"])
        chunk["network_group"] = [_edge_network_group(a, b) for a, b in zip(chunk["channel_i"], chunk["channel_j"])]
        chunk["interhemispheric"] = [_is_interhemispheric(a, b) for a, b in zip(chunk["channel_i"], chunk["channel_j"])]
        edge_cols = ["state", "edge_index", "channel_i", "channel_j", "channel_i_index", "channel_j_index", "band", "band_index", "network_group", "interhemispheric"]
        edge_parts.append(_partial_subject_summary(chunk, edge_cols))
        band_parts.append(_partial_subject_summary(chunk, ["state", "band", "band_index"]))
        seed_parts.append(_partial_seed_summary(chunk, edge_cols, "WPLI"))
    edges = _finish_subject_summary(edge_parts, ["state", "edge_index", "channel_i", "channel_j", "channel_i_index", "channel_j_index", "band", "band_index", "network_group", "interhemispheric"])
    band = _finish_subject_summary(band_parts, ["state", "band", "band_index"])
    node = _wpli_node_from_edges(edges)
    network = (
        edges.groupby(["state", "network_group", "band"], as_index=False)
        .agg(
            mean_abs_attribution=("mean_abs_attribution", "mean"),
            mean_signed_attribution=("mean_signed_attribution", "mean"),
            sd_abs_attribution=("mean_abs_attribution", "std"),
            n_edges=("edge_index", "count"),
        )
        .sort_values("mean_abs_attribution", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "edges": edges,
        "band": band,
        "node": node,
        "network": network,
        "wpli_seed": _finish_seed_summary(seed_parts, ["state", "edge_index", "channel_i", "channel_j", "band", "band_index", "network_group"], "WPLI"),
    }


def _partial_subject_summary(chunk: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    cols = ["subject_id", *feature_cols]
    return (
        chunk.groupby(cols, as_index=False)
        .agg(
            sum_signed=("norm_signed_attribution", "sum"),
            sum_abs=("norm_abs_attribution", "sum"),
            count=("norm_abs_attribution", "size"),
        )
    )


def _finish_subject_summary(parts: Sequence[pd.DataFrame], feature_cols: Sequence[str]) -> pd.DataFrame:
    partial = pd.concat(parts, ignore_index=True)
    subject = (
        partial.groupby(["subject_id", *feature_cols], as_index=False)
        .agg(sum_signed=("sum_signed", "sum"), sum_abs=("sum_abs", "sum"), count=("count", "sum"))
    )
    subject["subject_mean_signed"] = subject["sum_signed"] / subject["count"].clip(lower=1)
    subject["subject_mean_abs"] = subject["sum_abs"] / subject["count"].clip(lower=1)
    summary = (
        subject.groupby(list(feature_cols), as_index=False)
        .agg(
            mean_signed_attribution=("subject_mean_signed", "mean"),
            mean_abs_attribution=("subject_mean_abs", "mean"),
            sd_abs_attribution=("subject_mean_abs", "std"),
            n_subjects=("subject_id", "nunique"),
        )
        .sort_values("mean_abs_attribution", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def _partial_seed_summary(chunk: pd.DataFrame, feature_cols: Sequence[str], family: str) -> pd.DataFrame:
    out = (
        chunk.groupby(["seed", *feature_cols], as_index=False)
        .agg(sum_abs=("norm_abs_attribution", "sum"), count=("norm_abs_attribution", "size"))
    )
    out["feature_family"] = family
    return out


def _finish_seed_summary(parts: Sequence[pd.DataFrame], feature_cols: Sequence[str], family: str) -> pd.DataFrame:
    partial = pd.concat(parts, ignore_index=True)
    seed = (
        partial.groupby(["seed", *feature_cols], as_index=False)
        .agg(sum_abs=("sum_abs", "sum"), count=("count", "sum"))
    )
    seed["mean_abs_attribution"] = seed["sum_abs"] / seed["count"].clip(lower=1)
    seed["feature_family"] = family
    seed["feature_id"] = seed.apply(lambda row: _feature_id_from_row(row, family), axis=1)
    return seed


def _stability_from_seed_tables(psd_seed: pd.DataFrame, wpli_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, frame in (("PSD", psd_seed), ("WPLI", wpli_seed)):
        for k in (10, 20, 50):
            selected: dict[str, int] = {}
            seeds = sorted(frame["seed"].unique())
            for seed in seeds:
                top = frame[frame["seed"] == seed].sort_values("mean_abs_attribution", ascending=False).head(k)
                for feature_id in top["feature_id"].astype(str):
                    selected[feature_id] = selected.get(feature_id, 0) + 1
            for feature_id, count in selected.items():
                rows.append(
                    {
                        "feature_family": family,
                        "stability_metric": "topk_selection_frequency",
                        "feature_id": feature_id,
                        "k": k,
                        "selection_frequency": count / max(len(seeds), 1),
                        "n_seeds": len(seeds),
                    }
                )
    return pd.DataFrame(rows).sort_values(["k", "selection_frequency"], ascending=[True, False])


def write_feature_statistics(
    *,
    artifact_root: Path,
    record_by_subject: Mapping[str, Any],
    label_table: pd.DataFrame,
    targets: pd.DataFrame,
    frequency_bins: np.ndarray,
    edge_list: Sequence[tuple[str, str]],
    n_permutations: int,
) -> dict[str, pd.DataFrame]:
    explain_root = artifact_root / "results" / "explainability"
    psd_top = pd.read_csv(explain_root / "psd_channel_band_importance.csv").head(20)
    wpli_top = pd.read_csv(explain_root / "wpli_top_edges.csv").head(20)
    network_top = pd.read_csv(explain_root / "wpli_network_group_importance.csv").head(10)
    psd_stats = _stats_for_features(psd_top, "PSD", record_by_subject, label_table, targets, frequency_bins, edge_list, n_permutations)
    wpli_stats = _stats_for_features(wpli_top, "WPLI", record_by_subject, label_table, targets, frequency_bins, edge_list, n_permutations)
    network_stats = _stats_for_features(network_top, "NETWORK", record_by_subject, label_table, targets, frequency_bins, edge_list, n_permutations)
    table_root = artifact_root / "results" / "tables"
    psd_stats.to_csv(table_root / "top_psd_feature_group_statistics.csv", index=False)
    wpli_stats.to_csv(table_root / "top_wpli_feature_group_statistics.csv", index=False)
    network_stats.to_csv(table_root / "top_network_feature_group_statistics.csv", index=False)
    psd_stats.to_csv(explain_root / "psd_biomarker_validation.csv", index=False)
    wpli_stats.to_csv(explain_root / "wpli_biomarker_validation.csv", index=False)
    network_stats.to_csv(explain_root / "network_level_biomarker_validation.csv", index=False)
    return {"PSD": psd_stats, "WPLI": wpli_stats, "NETWORK": network_stats}


def _stats_for_features(
    features: pd.DataFrame,
    feature_type: str,
    record_by_subject: Mapping[str, Any],
    label_table: pd.DataFrame,
    targets: pd.DataFrame,
    frequency_bins: np.ndarray,
    edge_list: Sequence[tuple[str, str]],
    n_permutations: int,
) -> pd.DataFrame:
    rows = []
    for _, feature in features.iterrows():
        values = _raw_feature_values(feature, feature_type, record_by_subject, frequency_bins, edge_list)
        clinical = _clinical_for_values(values.index.tolist(), label_table, targets)
        stat = _feature_stats(values.to_numpy(float), clinical, n_permutations=n_permutations, seed=17 + len(rows))
        rows.append({**feature.to_dict(), "feature_type": feature_type, **stat})
    frame = pd.DataFrame(rows)
    for column in ("mannwhitney_p", "permutation_p", "spearman_residual_p", "spearman_signed_distance_p"):
        frame[f"{column}_fdr"] = _bh_fdr(frame[column].to_numpy(float)) if column in frame.columns else np.nan
    return frame


def _raw_feature_values(
    feature: pd.Series,
    feature_type: str,
    record_by_subject: Mapping[str, Any],
    frequency_bins: np.ndarray,
    edge_list: Sequence[tuple[str, str]],
) -> pd.Series:
    values = {}
    if feature_type == "PSD":
        state_index = 0 if feature["state"] == "EO" else 1
        channel_index = int(feature["channel_index"])
        band = str(feature["band"])
        mask = _psd_mask(band, frequency_bins)
        for subject, record in record_by_subject.items():
            values[subject] = float(np.asarray(record.modalities["psd"][state_index])[channel_index, mask].mean())
    elif feature_type == "WPLI":
        state_index = 0 if feature["state"] == "EO" else 1
        edge_index = int(feature["edge_index"])
        band_index = int(feature["band_index"])
        for subject, record in record_by_subject.items():
            values[subject] = float(np.asarray(record.modalities["wpli"][state_index])[edge_index, band_index])
    else:
        state_index = 0 if feature["state"] == "EO" else 1
        band_index = WPLI_BANDS.index(str(feature["band"]))
        group = str(feature["network_group"])
        edge_mask = np.asarray([_edge_network_group(a, b) == group for a, b in edge_list], dtype=bool)
        for subject, record in record_by_subject.items():
            arr = np.asarray(record.modalities["wpli"][state_index])
            values[subject] = float(arr[edge_mask, band_index].mean())
    return pd.Series(values, dtype=float)


def _clinical_for_values(subjects: Sequence[str], label_table: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": subject,
                "label": int(label_table.loc[subject, "label"]),
                "residual": float(label_table.loc[subject, "Residual"]),
                "signed_distance": float(targets.loc[subject, "signed_distance"]),
            }
            for subject in subjects
        ]
    )


def _feature_stats(values: np.ndarray, clinical: pd.DataFrame, *, n_permutations: int, seed: int) -> dict[str, float]:
    labels = clinical["label"].to_numpy(int)
    positive = values[labels == 1]
    negative = values[labels == 0]
    mw = mannwhitneyu(positive, negative, alternative="two-sided")
    residual_r, residual_p = _safe_spearman(values, clinical["residual"].to_numpy(float))
    distance_r, distance_p = _safe_spearman(values, clinical["signed_distance"].to_numpy(float))
    return {
        "n_subjects": int(len(values)),
        "positive_mean_value": float(np.mean(positive)),
        "negative_mean_value": float(np.mean(negative)),
        "mannwhitney_u": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
        "permutation_p": _permutation_p(values, labels, n_permutations=n_permutations, seed=seed),
        "cliffs_delta": _cliffs_delta(positive, negative),
        "rank_biserial_correlation": 2.0 * float(mw.statistic) / (len(positive) * len(negative)) - 1.0,
        "spearman_residual_r": residual_r,
        "spearman_residual_p": residual_p,
        "spearman_signed_distance_r": distance_r,
        "spearman_signed_distance_p": distance_p,
    }


def write_table4(artifact_root: Path, stats: Mapping[str, pd.DataFrame]) -> None:
    frames = []
    psd = pd.read_csv(artifact_root / "results" / "explainability" / "psd_channel_band_importance.csv").head(20)
    wpli = pd.read_csv(artifact_root / "results" / "explainability" / "wpli_top_edges.csv").head(20)
    network = pd.read_csv(artifact_root / "results" / "explainability" / "wpli_network_group_importance.csv").head(10)
    frames.append(_table4_rows(psd, stats["PSD"], "PSD"))
    frames.append(_table4_rows(wpli, stats["WPLI"], "WPLI"))
    frames.append(_table4_rows(network, stats["NETWORK"], "NETWORK"))
    table = pd.concat(frames, ignore_index=True, sort=False)
    table = table.sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
    table.insert(0, "feature_rank", np.arange(1, len(table) + 1))
    columns = [
        "feature_rank",
        "feature_type",
        "state",
        "band",
        "channel_or_edge",
        "node_1",
        "node_2",
        "network_group",
        "mean_signed_attribution",
        "mean_abs_attribution",
        "sd_abs_attribution",
        "direction",
        "interpretation_for_paper",
        "raw_feature_group_p",
        "raw_feature_group_p_fdr",
        "raw_feature_effect_size",
        "spearman_with_residual",
        "spearman_p",
        "spearman_p_fdr",
    ]
    table = table.loc[:, columns]
    table.to_csv(artifact_root / "results" / "tables" / "table4_explainability_top_features_for_paper.csv", index=False)
    (artifact_root / "results" / "tables" / "table4_explainability_top_features_for_paper.md").write_text(_to_markdown(table), encoding="utf-8")


def _table4_rows(features: pd.DataFrame, stats: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    rows = []
    for index, (_, feature) in enumerate(features.iterrows()):
        stat = stats.iloc[index] if index < len(stats) else pd.Series(dtype=object)
        signed = float(feature.get("mean_signed_attribution", 0.0))
        if feature_type == "PSD":
            channel_or_edge = str(feature["channel"])
            node_1 = str(feature["channel"])
            node_2 = ""
            network_group = ""
        elif feature_type == "WPLI":
            channel_or_edge = f"{feature['channel_i']}-{feature['channel_j']}"
            node_1 = str(feature["channel_i"])
            node_2 = str(feature["channel_j"])
            network_group = str(feature.get("network_group", ""))
        else:
            channel_or_edge = str(feature["network_group"])
            node_1 = ""
            node_2 = ""
            network_group = str(feature["network_group"])
        rows.append(
            {
                "feature_type": feature_type,
                "state": feature.get("state", ""),
                "band": feature.get("band", ""),
                "channel_or_edge": channel_or_edge,
                "node_1": node_1,
                "node_2": node_2,
                "network_group": network_group,
                "mean_signed_attribution": signed,
                "mean_abs_attribution": float(feature.get("mean_abs_attribution", np.nan)),
                "sd_abs_attribution": float(feature.get("sd_abs_attribution", np.nan)),
                "direction": "positive_to_proportional_recovery" if signed > 0 else "negative_to_poor_recovery",
                "interpretation_for_paper": _feature_interpretation(feature_type, signed),
                "raw_feature_group_p": stat.get("permutation_p", np.nan),
                "raw_feature_group_p_fdr": stat.get("permutation_p_fdr", np.nan),
                "raw_feature_effect_size": stat.get("cliffs_delta", np.nan),
                "spearman_with_residual": stat.get("spearman_residual_r", np.nan),
                "spearman_p": stat.get("spearman_residual_p", np.nan),
                "spearman_p_fdr": stat.get("spearman_residual_p_fdr", np.nan),
            }
        )
    return pd.DataFrame(rows)


def write_explainability_figures_and_docs(artifact_root: Path) -> None:
    revised = artifact_root / "results" / "figures" / "revised_initial"
    psd = pd.read_csv(artifact_root / "results" / "explainability" / "psd_channel_band_importance.csv")
    wpli_edges = pd.read_csv(artifact_root / "results" / "explainability" / "wpli_top_edges.csv")
    wpli_node = pd.read_csv(artifact_root / "results" / "explainability" / "wpli_node_importance.csv")
    table4 = pd.read_csv(artifact_root / "results" / "tables" / "table4_explainability_top_features_for_paper.csv")
    stats_psd = pd.read_csv(artifact_root / "results" / "tables" / "top_psd_feature_group_statistics.csv")
    _save_psd_topomap_grid(psd, revised / "figure6a_psd_topomap_bands")
    _save_wpli_connectivity_grid(wpli_edges, revised / "figure6b_wpli_connectivity_bands")
    shutil.copy2(revised / "figure6b_wpli_connectivity_bands.png", revised / "figure6b_wpli_connectivity_contact_sheet.png")
    _save_wpli_node_topomap_grid(wpli_node, revised / "figure6c_wpli_node_topomap")
    _save_figure6_composite(table4, psd, wpli_edges, stats_psd, revised / "figure6_eeg_explainability")
    _write_topomap_manifest(artifact_root)
    _write_connectivity_manifest(artifact_root)
    _write_explainability_docs(artifact_root, table4)


def write_occlusion_figure(artifact_root: Path, summary: pd.DataFrame) -> None:
    _save_figure5e(artifact_root, summary)


def _save_figure5c(artifact_root: Path, table: pd.DataFrame) -> None:
    plot = table.sort_values("balanced_accuracy", ascending=True)
    labels = [_friendly(name) for name in plot["ablation_name"]]
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier", "Brier (lower is better)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, max(5, len(plot) * 0.25)), sharey=True)
    for ax, (column, title) in zip(axes, metrics, strict=True):
        ax.barh(np.arange(len(plot)), plot[column].astype(float), color="#4B79A1")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(column)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        if column != "brier":
            ax.set_xlim(0, 1)
    axes[0].set_yticks(np.arange(len(plot)), labels=labels, fontsize=6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    fig.tight_layout()
    _save_all_formats(fig, artifact_root / "results" / "figures" / "revised_initial" / "figure5c_feature_state_band_ablation_ranking")


def _save_figure5d(artifact_root: Path, table: pd.DataFrame) -> None:
    names = ["full_psd_wpli", "psd_only", "wpli_only", "eo_only", "ec_only", "beta_medium_beta_high", "motor_wpli_edges_only"]
    plot = table[table["ablation_name"].isin(names)].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(plot["n_input_features"], plot["roc_auc"], s=55, color="#2F6B4F", label="ROC-AUC")
    ax.scatter(plot["n_input_features"], plot["balanced_accuracy"], s=55, marker="s", color="#8A4A32", label="Balanced accuracy")
    for _, row in plot.iterrows():
        ax.annotate(_friendly(row["ablation_name"]), (row["n_input_features"], row["roc_auc"]), xytext=(4, 3), textcoords="offset points", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Number of input features")
    ax.set_ylabel("Performance")
    ax.set_ylim(0, 1)
    ax.grid(color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save_all_formats(fig, artifact_root / "results" / "figures" / "revised_initial" / "figure5d_information_efficiency")


def _save_figure5e(artifact_root: Path, summary: pd.DataFrame) -> None:
    order = [
        "occlude_psd_branch",
        "occlude_wpli_branch",
        "occlude_eo_state",
        "occlude_ec_state",
        "occlude_psd_delta",
        "occlude_psd_theta",
        "occlude_psd_alpha",
        "occlude_psd_beta_low",
        "occlude_psd_beta_medium",
        "occlude_psd_beta_high",
        "occlude_psd_gamma",
        "occlude_wpli_delta",
        "occlude_wpli_theta",
        "occlude_wpli_alpha",
        "occlude_wpli_beta_low",
        "occlude_wpli_beta_medium",
        "occlude_wpli_beta_high",
        "occlude_motor_wpli_edges",
    ]
    plot = summary.set_index("occlusion_name").reindex([name for name in order if name in set(summary["occlusion_name"])]).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#8A4A32" if value < 0 else "#2F6B4F" for value in plot["mean_probability_drop"]]
    ax.bar(np.arange(len(plot)), plot["mean_probability_drop"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(plot)), labels=[_friendly(name) for name in plot["occlusion_name"]], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Final-model occlusion ablation")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    _save_all_formats(fig, artifact_root / "results" / "figures" / "revised_initial" / "figure5e_final_model_occlusion_ablation")


def _save_psd_topomap_grid(psd: pd.DataFrame, basename: Path) -> None:
    coords = _channel_coords()
    states = ["EO", "EC"]
    bands = [name for name, _, _ in PSD_BANDS]
    vmax = float(np.nanmax(np.abs(psd["mean_signed_attribution"].to_numpy(float)))) or 1.0
    fig, axes = plt.subplots(len(states), len(bands), figsize=(13, 4.2))
    for i, state in enumerate(states):
        for j, band in enumerate(bands):
            ax = axes[i, j]
            subset = psd[(psd["state"] == state) & (psd["band"] == band)]
            _draw_head_scatter(ax, subset, coords, value_col="mean_signed_attribution", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{state} {band}", fontsize=7)
    cax = fig.add_axes([0.92, 0.18, 0.012, 0.64])
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="coolwarm"), cax=cax, label="positive to proportional recovery / negative to poor recovery")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    _save_all_formats(fig, basename)


def _save_wpli_node_topomap_grid(node: pd.DataFrame, basename: Path) -> None:
    coords = _channel_coords()
    states = ["EO", "EC"]
    bands = list(WPLI_BANDS)
    value_col = "node_importance" if "node_importance" in node.columns else "mean_abs_attribution"
    vmax = float(np.nanmax(node[value_col].to_numpy(float))) or 1.0
    fig, axes = plt.subplots(len(states), len(bands), figsize=(12, 4.2))
    for i, state in enumerate(states):
        for j, band in enumerate(bands):
            ax = axes[i, j]
            subset = node[(node["state"] == state) & (node["band"] == band)].rename(columns={"channel": "channel"})
            _draw_head_scatter(ax, subset, coords, value_col=value_col, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{state} {band}", fontsize=7)
    cax = fig.add_axes([0.92, 0.18, 0.012, 0.64])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, label="Mean absolute attribution")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    _save_all_formats(fig, basename)


def _save_wpli_connectivity_grid(edges: pd.DataFrame, basename: Path) -> None:
    coords = _channel_coords()
    states = ["EO", "EC"]
    bands = list(WPLI_BANDS)
    vmax = float(np.nanmax(np.abs(edges["mean_signed_attribution"].to_numpy(float)))) or 1.0
    fig, axes = plt.subplots(len(states), len(bands), figsize=(13, 4.6))
    for i, state in enumerate(states):
        for j, band in enumerate(bands):
            ax = axes[i, j]
            subset = edges[(edges["state"] == state) & (edges["band"] == band)].sort_values("mean_abs_attribution", ascending=False).head(20)
            _draw_connectivity(ax, subset, coords, vmax)
            ax.set_title(f"{state} {band}", fontsize=7)
    fig.suptitle("WPLI connectivity: line width = mean absolute attribution; color = signed direction", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_all_formats(fig, basename)


def _save_figure6_composite(table4: pd.DataFrame, psd: pd.DataFrame, edges: pd.DataFrame, stats: pd.DataFrame, basename: Path) -> None:
    coords = _channel_coords()
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    top = table4.head(12).iloc[::-1]
    ax1.barh(np.arange(len(top)), top["mean_abs_attribution"], color="#4B79A1")
    ax1.set_yticks(np.arange(len(top)), labels=top["channel_or_edge"], fontsize=6)
    ax1.set_xlabel("Mean absolute attribution")
    ax1.set_title("A. Global importance ranking")
    ax2 = fig.add_subplot(gs[0, 1])
    psd_top = psd.sort_values("mean_abs_attribution", ascending=False).iloc[0]
    subset = psd[(psd["state"] == psd_top["state"]) & (psd["band"] == psd_top["band"])]
    _draw_head_scatter(ax2, subset, coords, value_col="mean_signed_attribution", cmap="coolwarm", vmin=-subset["mean_signed_attribution"].abs().max(), vmax=subset["mean_signed_attribution"].abs().max())
    ax2.set_title(f"B. PSD topography ({psd_top['state']} {psd_top['band']})")
    ax3 = fig.add_subplot(gs[1, 0])
    edge_top = edges.sort_values("mean_abs_attribution", ascending=False).iloc[0]
    edge_subset = edges[(edges["state"] == edge_top["state"]) & (edges["band"] == edge_top["band"])].head(20)
    _draw_connectivity(ax3, edge_subset, coords, float(edge_subset["mean_signed_attribution"].abs().max()) or 1.0)
    ax3.set_title(f"C. WPLI connectivity ({edge_top['state']} {edge_top['band']})")
    ax4 = fig.add_subplot(gs[1, 1])
    stat_plot = stats.head(8).iloc[::-1]
    ax4.barh(np.arange(len(stat_plot)), -np.log10(stat_plot["permutation_p"].astype(float).clip(lower=1e-6)), color="#8A4A32")
    ax4.set_yticks(np.arange(len(stat_plot)), labels=stat_plot["channel"].astype(str) + " " + stat_plot["band"].astype(str), fontsize=6)
    ax4.set_xlabel("-log10 permutation p")
    ax4.set_title("D. Top feature group statistics")
    fig.tight_layout()
    _save_all_formats(fig, basename)


def _draw_head_scatter(ax: plt.Axes, values: pd.DataFrame, coords: Mapping[str, tuple[float, float]], *, value_col: str, cmap: str, vmin: float, vmax: float) -> None:
    xs, ys, cs = [], [], []
    for _, row in values.iterrows():
        channel = str(row.get("channel", ""))
        if channel not in coords:
            continue
        x, y = coords[channel]
        xs.append(x)
        ys.append(y)
        cs.append(float(row[value_col]))
    ax.add_patch(plt.Circle((0, 0), 1.02, fill=False, color="black", linewidth=0.6))
    ax.scatter(xs, ys, c=cs, cmap=cmap, vmin=vmin, vmax=vmax, s=28, edgecolors="black", linewidths=0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_connectivity(ax: plt.Axes, edges: pd.DataFrame, coords: Mapping[str, tuple[float, float]], vmax: float) -> None:
    ax.add_patch(plt.Circle((0, 0), 1.02, fill=False, color="black", linewidth=0.6))
    for channel, (x, y) in coords.items():
        ax.scatter([x], [y], s=5, c="black", alpha=0.6)
    max_abs = float(edges["mean_abs_attribution"].max()) if len(edges) else 1.0
    cmap = plt.get_cmap("coolwarm")
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    for _, row in edges.iterrows():
        if row["channel_i"] not in coords or row["channel_j"] not in coords:
            continue
        x1, y1 = coords[row["channel_i"]]
        x2, y2 = coords[row["channel_j"]]
        width = 0.4 + 3.0 * float(row["mean_abs_attribution"]) / (max_abs + 1e-12)
        ax.plot([x1, x2], [y1, y2], color=cmap(norm(float(row["mean_signed_attribution"]))), linewidth=width, alpha=0.65)
    ax.set_aspect("equal")
    ax.axis("off")


def _channel_coords() -> dict[str, tuple[float, float]]:
    channels = list(CANONICAL_CHANNELS_62)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(channels), endpoint=False)
    return {channel: (0.9 * np.cos(angle), 0.9 * np.sin(angle)) for channel, angle in zip(channels, angles)}


def _write_ablation_doc(artifact_root: Path, table: pd.DataFrame) -> None:
    best = table.sort_values("balanced_accuracy", ascending=False).head(5)
    full = table[table["ablation_name"] == "full_psd_wpli"].iloc[0]
    psd = table[table["ablation_name"] == "psd_only"].iloc[0]
    wpli = table[table["ablation_name"] == "wpli_only"].iloc[0]
    eo = table[table["ablation_name"] == "eo_only"].iloc[0]
    ec = table[table["ablation_name"] == "ec_only"].iloc[0]
    motor = table[table["ablation_name"] == "motor_wpli_edges_only"].iloc[0]
    source = "rerun_ablation_explainability_secondary_20260609/results/tables/table3_feature_state_band_ablation_for_paper.csv"
    lines = [
        "# 3.3 特征、状态、频段消融实验结果摘要草稿",
        "",
        "## 1. 为什么进行特征、状态、频段消融",
        "本分析用于评估最终模型相关 EEG 信息在模态、状态、频段和运动网络连接层面的预测信息保留情况。所有 feature-subset 结果均为 patient-level LOSO，标准化和 SelectK 特征选择仅在训练折拟合。",
        "",
        "## 2. 每类消融组的设置",
        "设置包括 PSD/WPLI 模态消融、EO/EC 状态消融、单频段 only、leave-one-band-out，以及运动相关 WPLI 边保留或删除分析。",
        "",
        "## 3. 模态消融结果",
        f"完整 PSD+WPLI 的 balanced accuracy 为 {full['balanced_accuracy']:.3f}，ROC-AUC 为 {full['roc_auc']:.3f}；PSD only 为 {psd['balanced_accuracy']:.3f}/{psd['roc_auc']:.3f}，WPLI only 为 {wpli['balanced_accuracy']:.3f}/{wpli['roc_auc']:.3f}（来源：`{source}`）。",
        "",
        "## 4. 状态消融结果",
        f"EO only 的 balanced accuracy/ROC-AUC 为 {eo['balanced_accuracy']:.3f}/{eo['roc_auc']:.3f}，EC only 为 {ec['balanced_accuracy']:.3f}/{ec['roc_auc']:.3f}（来源：`{source}`）。",
        "",
        "## 5. 频段消融结果",
        f"按 balanced accuracy 排名前 5 的组合为：{', '.join(best['ablation_name'].astype(str).tolist())}（来源：`{source}`）。",
        "",
        "## 6. 运动网络连接结果",
        f"motor WPLI edges only 在低维输入下的 balanced accuracy/ROC-AUC 为 {motor['balanced_accuracy']:.3f}/{motor['roc_auc']:.3f}（来源：`{source}`）。",
        "",
        "## 7. final-model occlusion 与 feature-subset ablation 的一致性",
        "最终模型 occlusion 摘要见 `rerun_ablation_explainability_secondary_20260609/results/explainability/final_model_occlusion_modality_state_band_summary.csv`。正的 probability drop 表示被遮蔽部分对比例恢复预测有正向贡献。",
        "",
        "## 8. 谨慎结论",
        "这些结果说明模型在当前小样本内部 LOSO 中更依赖哪些 EEG 信息，只能作为模型依赖和关联性证据，不能写成因果机制。",
    ]
    (artifact_root / "docs" / "modality_state_band_ablation_results_for_section3.md").write_text("\n".join(lines), encoding="utf-8")


def _write_explainability_docs(artifact_root: Path, table4: pd.DataFrame) -> None:
    source = "rerun_ablation_explainability_secondary_20260609/results/tables/table4_explainability_top_features_for_paper.csv"
    top = table4.iloc[0]
    fdr = table4[table4["raw_feature_group_p_fdr"].astype(float) < 0.05]
    lines = [
        "# 3.5 可解释性分析结果摘要草稿",
        "",
        "## 1. 可解释性分析目的",
        "本分析用于描述最终 highrank + SWA Residual-aware patient-level Barlow SSL-CNN 在分类 logit 上依赖的 PSD、WPLI、状态、频段和网络连接信息。",
        "",
        "## 2. 全局重要性排序",
        f"当前总表中排名第 1 的特征为 {top['feature_type']} {top['state']} {top['band']} {top['channel_or_edge']}，mean_abs_attribution={top['mean_abs_attribution']:.4f}（来源：`{source}`）。",
        "",
        "## 3. PSD topomap",
        "PSD topomap 使用 mean_signed_attribution，正值表示推动比例恢复预测，负值表示推动恢复不良预测。图件来源：`rerun_ablation_explainability_secondary_20260609/results/figures/revised_initial/figure6a_psd_topomap_bands.*`。",
        "",
        "## 4. WPLI connectivity",
        "WPLI connectivity 图中线条粗细表示 mean_abs_attribution，颜色表示 mean_signed_attribution 的方向。图件来源：`rerun_ablation_explainability_secondary_20260609/results/figures/revised_initial/figure6b_wpli_connectivity_bands.*`。",
        "",
        "## 5. Top 特征统计验证",
        f"FDR 后 raw feature group p < 0.05 的条目数为 {len(fdr)}（来源：`{source}` 及 top_*_feature_group_statistics.csv）。",
        "",
        "## 6. 谨慎结论",
        "所有解释性结果均为模型依赖、关联性、hypothesis-generating 证据，不应写成因果机制或已验证临床 biomarker。",
    ]
    (artifact_root / "docs" / "explainability_results_for_section3.md").write_text("\n".join(lines), encoding="utf-8")
    claims = [
        "# Explainability Numeric Claims",
        "",
        f"- 数字内容：Top feature mean_abs_attribution={top['mean_abs_attribution']:.4f}；来源文件：`{source}`；对应图或表：Table 4 / Figure 6A-D；是否经过 FDR 校正：归因排序本身不适用，统计列包含 FDR；是否 exploratory：是。",
        f"- 数字内容：FDR 后 raw feature group p < 0.05 条目数={len(fdr)}；来源文件：`{source}`；对应图或表：Table 4；是否经过 FDR 校正：是；是否 exploratory：是。",
    ]
    (artifact_root / "docs" / "explainability_numeric_claims.md").write_text("\n".join(claims), encoding="utf-8")


def _write_topomap_manifest(artifact_root: Path) -> None:
    rows = []
    for figure in ["figure6a_psd_topomap_bands", "figure6c_wpli_node_topomap"]:
        for ext in ["png", "svg", "pdf", "tiff"]:
            rows.append({"figure": figure, "path": f"results/figures/revised_initial/{figure}.{ext}", "render_type": "channel-coordinate head map"})
    pd.DataFrame(rows).to_csv(artifact_root / "results" / "explainability" / "mne_topomap_manifest.csv", index=False)


def _write_connectivity_manifest(artifact_root: Path) -> None:
    rows = []
    for ext in ["png", "svg", "pdf", "tiff"]:
        rows.append({"figure": "figure6b_wpli_connectivity_bands", "path": f"results/figures/revised_initial/figure6b_wpli_connectivity_bands.{ext}", "top_edges_per_state_band": 20})
    rows.append({"figure": "figure6b_wpli_connectivity_contact_sheet", "path": "results/figures/revised_initial/figure6b_wpli_connectivity_contact_sheet.png", "top_edges_per_state_band": 20})
    pd.DataFrame(rows).to_csv(artifact_root / "results" / "explainability" / "mne_wpli_connectivity_manifest.csv", index=False)


def _write_occlusion_aliases(artifact_root: Path, occlusion: pd.DataFrame) -> None:
    branch = occlusion[occlusion["occlusion_family"].isin(["modality", "state"])].copy()
    branch = branch.rename(
        columns={
            "occlusion_name": "group_name",
            "occlusion_family": "group_type",
            "original_prob": "original_probability",
            "occluded_prob": "occluded_probability",
            "probability_drop": "delta_probability",
            "logit_drop": "delta_logit",
        }
    )
    branch.to_csv(artifact_root / "results" / "explainability" / "occlusion_branch_state.csv", index=False)
    psd = occlusion[occlusion["occlusion_family"] == "psd_band"].copy()
    psd = psd.rename(columns={"occlusion_name": "group_name", "occlusion_family": "group_type", "original_prob": "original_probability", "occluded_prob": "occluded_probability", "probability_drop": "delta_probability", "logit_drop": "delta_logit"})
    psd.to_csv(artifact_root / "results" / "explainability" / "occlusion_psd_band_channel.csv", index=False)
    wpli = occlusion[occlusion["occlusion_family"].isin(["wpli_band", "motor_network"])].copy()
    wpli = wpli.rename(columns={"occlusion_name": "group_name", "occlusion_family": "group_type", "original_prob": "original_probability", "occluded_prob": "occluded_probability", "probability_drop": "delta_probability", "logit_drop": "delta_logit"})
    wpli.to_csv(artifact_root / "results" / "explainability" / "occlusion_wpli_edge_node_band.csv", index=False)


def _write_explainability_method_note(artifact_root: Path) -> None:
    rows = [
        {
            "method": "existing_raw_attribution_long_reaggregation",
            "target": "classification_logit",
            "baseline": "fold-local standardized zero",
            "patient_level_normalization": "max absolute attribution per subject/fold/seed across PSD and WPLI before group summary",
            "note": "Raw attribution long tables from the prior final-model explainability run were reaggregated with current Gamma band mapping in this secondary output folder.",
        }
    ]
    pd.DataFrame(rows).to_csv(artifact_root / "results" / "explainability" / "explainability_method_manifest.csv", index=False)


def _write_recomputed_explainability_method_note(artifact_root: Path, ig_steps: int, smoothgrad_samples: int, smoothgrad_noise_std: float) -> None:
    rows = [
        {
            "method": "integrated_gradients_and_smoothgrad_integrated_gradients_recomputed",
            "target": "classification_logit",
            "ig_steps": int(ig_steps),
            "smoothgrad_samples": int(smoothgrad_samples),
            "smoothgrad_noise_std": float(smoothgrad_noise_std),
            "baseline": "fold-local standardized zero",
            "patient_level_normalization": "IG and SmoothGrad attributions normalized within each subject/fold/seed, averaged within sample, then summarized at patient level",
            "note": "This secondary output folder recomputed final-model attributions from SWA checkpoints; no training was rerun.",
        }
    ]
    pd.DataFrame(rows).to_csv(artifact_root / "results" / "explainability" / "explainability_method_manifest.csv", index=False)


def _write_run_manifest(artifact_root: Path, audit: pd.DataFrame) -> None:
    rows = [
        {"item": "artifact_root", "value": str(artifact_root)},
        {"item": "model_group", "value": MODEL_GROUP},
        {"item": "n_checkpoint_rows", "value": str(len(audit))},
        {"item": "seeds", "value": ",".join(map(str, SEEDS_10))},
    ]
    pd.DataFrame(rows).to_csv(artifact_root / "run_manifest.csv", index=False)


def _ensure_output_dirs(root: Path) -> None:
    for relative in [
        "scripts",
        "tests",
        "logs",
        "docs",
        "results/metrics",
        "results/predictions",
        "results/tables",
        "results/explainability",
        "results/figures/revised_initial",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def _best_metric_row_per_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    trained = metrics[metrics.get("status", "trained").fillna("trained") == "trained"].copy()
    if trained.empty:
        return metrics.copy()
    return (
        trained.sort_values(["ablation_name", "roc_auc", "brier_score"], ascending=[True, False, True])
        .groupby("ablation_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _ablation_interpretation(row: pd.Series) -> str:
    delta_auc = row.get("delta_roc_auc", np.nan)
    if pd.isna(delta_auc):
        return "Reference row for feature-subset comparison."
    if float(delta_auc) >= 0:
        return "This subset retained or exceeded the full-input ROC-AUC in internal LOSO; interpret cautiously."
    return "This subset lost ROC-AUC relative to the full input, suggesting reduced retained predictive information."


def _occlusion_interpretation(name: str, mean_drop: float) -> str:
    if mean_drop > 0:
        return f"Masking {name} reduced proportional-recovery probability on average, consistent with positive model contribution."
    if mean_drop < 0:
        return f"Masking {name} increased proportional-recovery probability on average, consistent with negative or poor-recovery-oriented contribution."
    return f"Masking {name} produced negligible mean probability change."


def _feature_interpretation(feature_type: str, signed: float) -> str:
    direction = "proportional recovery" if signed > 0 else "poor recovery"
    return f"{feature_type} feature with signed attribution oriented toward {direction}; exploratory model-dependence evidence."


def _load_script_module(path: Path, name: str) -> Any:
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_frequency_bins(source_root: Path, subject_id: str) -> np.ndarray:
    with np.load(source_root / "data" / "features" / "psd" / f"{subject_id}_EO_psd.npz", allow_pickle=False) as payload:
        return np.asarray(payload["frequency_bins"], dtype=float)


def _zero_keys(batch: dict[str, torch.Tensor], keys: Iterable[str]) -> None:
    for key in keys:
        batch[key].zero_()


def _zero_psd_band(batch: dict[str, torch.Tensor], band: str, frequency_bins: np.ndarray) -> None:
    mask = torch.as_tensor(_psd_mask(band, frequency_bins), device=batch["psd_eo"].device)
    for key in ("psd_eo", "psd_ec"):
        batch[key][:, :, mask] = 0.0


def _zero_wpli_band(batch: dict[str, torch.Tensor], band: str) -> None:
    index = WPLI_BANDS.index(band)
    for key in ("wpli_eo", "wpli_ec"):
        batch[key][:, :, index] = 0.0


def _zero_motor_edges(batch: dict[str, torch.Tensor], edge_list: Sequence[tuple[str, str]], *, want_motor: bool) -> None:
    mask = torch.as_tensor([((a in MOTOR_CHANNELS) or (b in MOTOR_CHANNELS)) == want_motor for a, b in edge_list], device=batch["wpli_eo"].device)
    for key in ("wpli_eo", "wpli_ec"):
        batch[key][:, mask, :] = 0.0


def _psd_mask(band: str, frequency_bins: np.ndarray) -> np.ndarray:
    for name, low, high in PSD_BANDS:
        if name == band:
            return (frequency_bins >= low) & (frequency_bins < high)
    raise ValueError(f"Unknown PSD band: {band}")


def _wpli_node_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    first = edges.rename(columns={"channel_i": "channel"})[["state", "band", "channel", "mean_abs_attribution", "mean_signed_attribution"]]
    second = edges.rename(columns={"channel_j": "channel"})[["state", "band", "channel", "mean_abs_attribution", "mean_signed_attribution"]]
    both = pd.concat([first, second], ignore_index=True)
    return (
        both.groupby(["state", "band", "channel"], as_index=False)
        .agg(node_importance=("mean_abs_attribution", "mean"), mean_signed_attribution=("mean_signed_attribution", "mean"))
        .sort_values("node_importance", ascending=False)
        .reset_index(drop=True)
    )


def _feature_id_from_row(row: pd.Series, family: str) -> str:
    if family == "PSD":
        return f"{row['state']}|{row['channel']}|{row['band']}"
    return f"{row['state']}|{row['channel_i']}-{row['channel_j']}|{row['band']}"


def _edge_network_group(first: str, second: str) -> str:
    a = _channel_region(first)
    b = _channel_region(second)
    return "|".join(sorted((a, b)))


def _channel_region(channel: str) -> str:
    channel = str(channel).upper()
    if channel.startswith(("FP", "AF", "F", "FT")):
        return "frontal"
    if channel.startswith(("FC", "C", "CP")):
        return "central"
    if channel.startswith(("T", "TP")):
        return "temporal"
    if channel.startswith(("P", "PO")):
        return "parietal"
    if channel.startswith(("O", "CB")):
        return "occipital"
    return "other"


def _is_interhemispheric(first: str, second: str) -> bool:
    return _hemi(first) * _hemi(second) < 0


def _hemi(channel: str) -> int:
    digits = "".join(ch for ch in str(channel) if ch.isdigit())
    if not digits:
        return 0
    return -1 if int(digits[-1]) % 2 else 1


def _permutation_p(values: np.ndarray, labels: np.ndarray, *, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values[labels == 1].mean() - values[labels == 0].mean()))
    count = 1
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(labels)
        diff = abs(float(values[shuffled == 1].mean() - values[shuffled == 0].mean()))
        if diff >= observed:
            count += 1
    return count / float(n_permutations + 1)


def _cliffs_delta(positive: np.ndarray, negative: np.ndarray) -> float:
    greater = 0
    less = 0
    for x in positive:
        greater += int(np.sum(x > negative))
        less += int(np.sum(x < negative))
    return float((greater - less) / (len(positive) * len(negative)))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    r = 0.0 if not np.isfinite(result.statistic) else float(result.statistic)
    p = 1.0 if not np.isfinite(result.pvalue) else float(result.pvalue)
    return r, p


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.ones_like(p)
    running = 1.0
    m = len(p)
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _save_all_formats(fig: plt.Figure, basename: Path) -> None:
    basename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(basename.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(basename.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(basename.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(basename.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def _friendly(name: str) -> str:
    name = str(name)
    if name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[name]
    if name.startswith("occlude_psd_"):
        return "PSD " + _band_from_slug(name.replace("occlude_psd_", "")).lower()
    if name.startswith("occlude_wpli_"):
        return "WPLI " + _band_from_slug(name.replace("occlude_wpli_", "")).lower()
    return name.replace("_", " ")


def _label_slug(value: str) -> str:
    return "_".join(str(value).split())


def _slug(value: str) -> str:
    return "_".join(str(value).lower().split())


def _band_from_slug(value: str) -> str:
    return str(value).replace("_", " ").title().replace("Psd ", "PSD ").replace("Wpli ", "WPLI ")


def _float_is_close(value: Any, target: float) -> bool:
    try:
        return bool(np.isclose(float(value), target))
    except Exception:
        return False


if __name__ == "__main__":
    main()
