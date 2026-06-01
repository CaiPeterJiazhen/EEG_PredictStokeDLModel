from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.explainability.attribution import (
    attribution_correlation,
    classification_logit,
    classification_probability,
    manual_integrated_gradients,
    randomize_classifier_weights,
    smoothgrad_integrated_gradients,
)
from eeg_recovery.explainability.occlusion import clone_tensor_batch
from eeg_recovery.explainability.stability import compute_topk_selection_frequency
from eeg_recovery.explainability.tables import (
    PSD_BANDS,
    make_psd_attribution_long_table,
    make_wpli_edge_attribution_long_table,
)
from eeg_recovery.features.connectivity import build_edge_list, connectivity_bands
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.residual_targets import compute_signed_distance_from_label_table
from eeg_recovery.training.train_supervised import (
    _make_batch,
    load_supervised_feature_records,
    resolve_device,
)


SEEDS_10 = (0, 1, 2, 3, 4, 5, 7, 13, 21, 42)
INPUT_KEYS = ("psd_eo", "psd_ec", "wpli_eo", "wpli_ec")
WATCHED_SUBJECTS = ("sub09", "sub14")


@dataclass(frozen=True)
class ExplainedSample:
    subject_id: str
    fold_index: int
    seed: int
    y_true: int
    y_score: float
    y_pred: int
    residual: float
    signed_distance: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain residual-aware Patient-level Barlow SSL-CNN with input-level attributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-group", default="residualaware_highrank_swa_clsalpha1")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_10))
    parser.add_argument("--method", nargs="+", default=["integrated_gradients", "smoothgrad", "occlusion"])
    parser.add_argument("--target", default="classification_logit")
    parser.add_argument("--output-tag", default="residualaware_10seed")
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--smoothgrad-samples", type=int, default=4)
    parser.add_argument("--smoothgrad-noise-std", type=float, default=0.02)
    parser.add_argument("--occlusion-top-k", type=int, default=20)
    parser.add_argument("--biomarker-top-k", type=int, default=50)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    if args.target != "classification_logit":
        raise ValueError("Only classification_logit is supported for this locked model.")

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    explain_root = output_root / "results" / "explainability"
    figure_root = output_root / "results" / "figures" / "explainability"
    explain_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    label_table = load_supervised_label_table(path_config)
    label_table = label_table.copy()
    label_table["subject_id"] = label_table["subject_id"].map(normalize_subject_id)
    label_table = label_table.set_index("subject_id", drop=False)
    targets = compute_signed_distance_from_label_table(label_table.reset_index(drop=True), threshold=1.5)
    records = load_supervised_feature_records(path_config, label_table.reset_index(drop=True), feature_kind="psd-fc-wpli")
    record_by_subject = {record.subject_id: record for record in records}
    frequency_bins = _load_frequency_bins(output_root, records[0].subject_id)
    edge_list = build_edge_list(CANONICAL_CHANNELS_62)
    wpli_band_names = tuple(name for name, _ in connectivity_bands())
    device = resolve_device(args.device)

    psd_long_path = explain_root / "psd_attribution_long.csv"
    wpli_long_path = explain_root / "wpli_edge_attribution_long.csv"
    _remove_if_exists(psd_long_path)
    _remove_if_exists(wpli_long_path)

    gate_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    psd_seed_rows: list[pd.DataFrame] = []
    wpli_seed_rows: list[pd.DataFrame] = []
    processed = 0

    for seed in args.seeds:
        checkpoint_paths = sorted(
            _checkpoint_dir(output_root, args.model_group).glob(f"{args.model_group}_seed{seed}_fold*_test_*.pt")
        )
        if len(checkpoint_paths) != 19:
            raise FileNotFoundError(
                f"Expected 19 supervised fold checkpoints for seed {seed}, found {len(checkpoint_paths)}."
            )
        for checkpoint_path in checkpoint_paths:
            if args.max_folds and processed >= args.max_folds:
                break
            payload = _load_checkpoint(checkpoint_path, device)
            metadata = payload["metadata"]
            sample = _sample_from_metadata(metadata, label_table, targets, output_root=output_root)
            model = _build_model_from_metadata(metadata, payload["state_dict"], device)
            scaler = _scaler_from_payload(payload["state_scaler"])
            batch = _make_batch([record_by_subject[sample.subject_id]], scaler, device, "multimodal")
            with torch.no_grad():
                outputs = model(batch)
                probability = float(outputs["classification_probability"].detach().cpu().numpy()[0, 0])
                y_pred = int(probability >= 0.5)
                _, aux = model.backbone.extract_embedding(batch, return_aux=True)
            if abs(probability - sample.y_score) > 1e-5 or y_pred != sample.y_pred:
                raise ValueError(
                    f"Checkpoint prediction mismatch for {checkpoint_path}: got {probability}, expected {sample.y_score}."
                )
            prediction_rows.append(
                {
                    "subject_id": sample.subject_id,
                    "fold_index": sample.fold_index,
                    "seed": sample.seed,
                    "y_true": sample.y_true,
                    "y_score": probability,
                    "y_pred": y_pred,
                    "correct": int(y_pred == sample.y_true),
                    "residual": sample.residual,
                    "signed_distance": sample.signed_distance,
                }
            )
            gate_rows.extend(_gate_rows(sample, aux))
            attributions = _compute_attribution_for_methods(model, batch, args, seed=seed)
            _append_sample_attributions(
                psd_long_path=psd_long_path,
                wpli_long_path=wpli_long_path,
                sample=sample,
                attributions=attributions,
                channel_names=CANONICAL_CHANNELS_62,
                frequency_bins=frequency_bins,
                edge_list=edge_list,
                wpli_band_names=wpli_band_names,
            )
            psd_seed_rows.append(_sample_psd_feature_scores(sample, attributions, frequency_bins))
            wpli_seed_rows.append(_sample_wpli_feature_scores(sample, attributions, edge_list, wpli_band_names))
            if processed < 8:
                sanity_rows.extend(_sanity_check_rows(model, batch, sample, steps=max(8, args.ig_steps // 4)))
            processed += 1
        if args.max_folds and processed >= args.max_folds:
            break

    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(explain_root / "explained_predictions.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(explain_root / "branch_state_gate_weights.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(explain_root / "sanity_check_summary.csv", index=False)

    psd_long = pd.read_csv(psd_long_path)
    wpli_long = pd.read_csv(wpli_long_path)
    psd_outputs = _write_psd_aggregates(psd_long, explain_root)
    wpli_outputs = _write_wpli_aggregates(wpli_long, explain_root)
    _write_stability_outputs(psd_seed_rows, wpli_seed_rows, explain_root)
    _write_occlusion_outputs(
        output_root=output_root,
        model_group=args.model_group,
        seeds=args.seeds,
        label_table=label_table,
        targets=targets,
        record_by_subject=record_by_subject,
        device=device,
        frequency_bins=frequency_bins,
        edge_list=edge_list,
        top_psd_features=psd_outputs["top_channel_band"],
        top_wpli_features=wpli_outputs["top_edge_band"],
        occlusion_top_k=args.occlusion_top_k,
        max_folds=args.max_folds,
    )
    _append_branch_gate_stability(explain_root)
    _write_biomarker_validation(
        record_by_subject=record_by_subject,
        label_table=label_table,
        targets=targets,
        top_psd_features=psd_outputs["top_features"].head(args.biomarker_top_k),
        top_wpli_features=wpli_outputs["top_edges"].head(args.biomarker_top_k),
        frequency_bins=frequency_bins,
        edge_list=edge_list,
        wpli_band_names=wpli_band_names,
        explain_root=explain_root,
    )
    _make_figures(explain_root, figure_root)
    _write_document(explain_root, figure_root, output_root)
    _write_task_status(explain_root, output_root, processed)
    print(f"Explained {processed} seed/fold samples.")
    print(f"Wrote explainability outputs under {explain_root}")
    print(f"Wrote figures under {figure_root}")


def _load_residual_training_module() -> Any:
    script_path = PROJECT_ROOT / "scripts" / "30_train_residual_aware_patient_barlow.py"
    spec = spec_from_file_location("residual_aware_patient_barlow_for_explainability", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_model_from_metadata(metadata: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    module = _load_residual_training_module()
    model = module.ResidualAwarePatientBarlowModel(
        embedding_dim=int(metadata.get("embedding_dim", 32)),
        dropout=float(metadata.get("dropout", 0.0)),
        residual_alpha=float(metadata.get("selected_alpha", metadata.get("residual_alpha", 1.0))),
        residual_probability_scale=float(metadata.get("residual_probability_scale", 1.0)),
    ).to(device)
    model.load_state_dict({key: value.to(device) for key, value in state_dict.items()}, strict=True)
    model.eval()
    return model


def _checkpoint_dir(output_root: Path, model_group: str) -> Path:
    return output_root / "results" / "checkpoints" / "supervised" / model_group


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("checkpoint_type") != "residual_aware_supervised_fold":
        raise ValueError(f"Unexpected supervised checkpoint type in {path}")
    return payload


def _sample_from_metadata(
    metadata: Mapping[str, Any],
    label_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    output_root: Path,
) -> ExplainedSample:
    subject_id = normalize_subject_id(str(metadata["test_subject_id"]))
    y_true = int(label_table.loc[subject_id, "label"])
    prediction_path = (
        Path(output_root)
        / "results"
        / "predictions"
        / f"dl_loso_predictions_patient_barlow_residualaware_highrank_swa_clsalpha1_seed{int(metadata['seed'])}.csv"
    )
    prediction = pd.read_csv(prediction_path)
    row = prediction[prediction["subject_id"].map(normalize_subject_id) == subject_id].iloc[0]
    return ExplainedSample(
        subject_id=subject_id,
        fold_index=int(metadata["fold_index"]),
        seed=int(metadata["seed"]),
        y_true=y_true,
        y_score=float(row["y_score"]),
        y_pred=int(row["y_pred"]),
        residual=float(label_table.loc[subject_id, "Residual"]),
        signed_distance=float(targets.loc[subject_id, "signed_distance"]),
    )


def _scaler_from_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
        return {
            key: (
                np.asarray(value["mean"].detach().cpu().numpy(), dtype=np.float32),
                np.asarray(value["std"].detach().cpu().numpy(), dtype=np.float32),
            )
            for key, value in payload.items()
        }
    return (
        np.asarray(payload["mean"].detach().cpu().numpy(), dtype=np.float32),
        np.asarray(payload["std"].detach().cpu().numpy(), dtype=np.float32),
    )


def _load_frequency_bins(output_root: Path, subject_id: str) -> np.ndarray:
    path = output_root / "data" / "features" / "psd" / f"{subject_id}_EO_psd.npz"
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["frequency_bins"], dtype=float)


def _compute_attribution_for_methods(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], args: argparse.Namespace, *, seed: int) -> dict[str, torch.Tensor]:
    if "smoothgrad" in args.method:
        return smoothgrad_integrated_gradients(
            model,
            batch,
            input_keys=INPUT_KEYS,
            steps=args.ig_steps,
            noise_samples=args.smoothgrad_samples,
            noise_std=args.smoothgrad_noise_std,
            seed=seed,
        )
    return manual_integrated_gradients(model, batch, input_keys=INPUT_KEYS, steps=args.ig_steps)


def _append_sample_attributions(
    *,
    psd_long_path: Path,
    wpli_long_path: Path,
    sample: ExplainedSample,
    attributions: Mapping[str, torch.Tensor],
    channel_names: tuple[str, ...],
    frequency_bins: np.ndarray,
    edge_list: tuple[tuple[str, str], ...],
    wpli_band_names: tuple[str, ...],
) -> None:
    psd_frames = []
    for state, key in (("EO", "psd_eo"), ("EC", "psd_ec")):
        psd_frames.append(
            make_psd_attribution_long_table(
                subject_id=sample.subject_id,
                fold_index=sample.fold_index,
                seed=sample.seed,
                state=state,
                attribution=attributions[key].detach().cpu().numpy()[0],
                channel_names=channel_names,
                frequency_bins=frequency_bins,
                y_true=sample.y_true,
                y_score=sample.y_score,
                y_pred=sample.y_pred,
                residual=sample.residual,
                signed_distance=sample.signed_distance,
            )
        )
    wpli_frames = []
    for state, key in (("EO", "wpli_eo"), ("EC", "wpli_ec")):
        wpli_frames.append(
            make_wpli_edge_attribution_long_table(
                subject_id=sample.subject_id,
                fold_index=sample.fold_index,
                seed=sample.seed,
                state=state,
                attribution=attributions[key].detach().cpu().numpy()[0],
                edge_list=edge_list,
                band_names=wpli_band_names,
                y_true=sample.y_true,
                y_score=sample.y_score,
                y_pred=sample.y_pred,
                residual=sample.residual,
                signed_distance=sample.signed_distance,
            )
        )
    _append_frame(pd.concat(psd_frames, ignore_index=True), psd_long_path)
    _append_frame(pd.concat(wpli_frames, ignore_index=True), wpli_long_path)


def _append_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _sample_psd_feature_scores(sample: ExplainedSample, attributions: Mapping[str, torch.Tensor], frequency_bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for state, key in (("EO", "psd_eo"), ("EC", "psd_ec")):
        values = attributions[key].detach().cpu().numpy()[0]
        for channel_index, channel in enumerate(CANONICAL_CHANNELS_62):
            for band_name, low, high in PSD_BANDS:
                mask = (frequency_bins >= low) & (frequency_bins < high)
                if not mask.any():
                    continue
                rows.append(
                    {
                        "seed": sample.seed,
                        "subject_id": sample.subject_id,
                        "feature_id": f"{state}|{channel}|{band_name}",
                        "state": state,
                        "channel": channel,
                        "band": band_name,
                        "mean_abs_attribution": float(np.mean(np.abs(values[channel_index, mask]))),
                        "mean_signed_attribution": float(np.mean(values[channel_index, mask])),
                    }
                )
    return pd.DataFrame(rows)


def _sample_wpli_feature_scores(
    sample: ExplainedSample,
    attributions: Mapping[str, torch.Tensor],
    edge_list: tuple[tuple[str, str], ...],
    band_names: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for state, key in (("EO", "wpli_eo"), ("EC", "wpli_ec")):
        values = attributions[key].detach().cpu().numpy()[0]
        for edge_index, (channel_i, channel_j) in enumerate(edge_list):
            for band_index, band in enumerate(band_names):
                rows.append(
                    {
                        "seed": sample.seed,
                        "subject_id": sample.subject_id,
                        "feature_id": f"{state}|{channel_i}-{channel_j}|{band}",
                        "state": state,
                        "edge_index": edge_index,
                        "channel_i": channel_i,
                        "channel_j": channel_j,
                        "band": band,
                        "mean_abs_attribution": float(abs(values[edge_index, band_index])),
                        "mean_signed_attribution": float(values[edge_index, band_index]),
                    }
                )
    return pd.DataFrame(rows)


def _write_psd_aggregates(psd_long: pd.DataFrame, explain_root: Path) -> dict[str, pd.DataFrame]:
    group_cols = ["state", "channel", "channel_index", "band"]
    channel_band = _aggregate_attribution(psd_long, group_cols)
    channel_band.to_csv(explain_root / "psd_channel_band_importance.csv", index=False)
    frequency = _aggregate_attribution(psd_long, ["state", "frequency_hz", "frequency_bin", "band"])
    frequency.to_csv(explain_root / "psd_frequency_importance.csv", index=False)
    channel_frequency = _aggregate_attribution(
        psd_long,
        ["state", "channel", "channel_index", "frequency_hz", "frequency_bin", "band"],
    ).sort_values("mean_abs_attribution", ascending=False)
    channel_frequency.to_csv(explain_root / "psd_channel_frequency_top_features.csv", index=False)
    return {
        "top_channel_band": channel_band.sort_values("mean_abs_attribution", ascending=False),
        "top_features": channel_frequency,
    }


def _write_wpli_aggregates(wpli_long: pd.DataFrame, explain_root: Path) -> dict[str, pd.DataFrame]:
    edges = _aggregate_attribution(
        wpli_long,
        ["state", "edge_index", "channel_i", "channel_j", "channel_i_index", "channel_j_index", "band", "band_index"],
    ).sort_values("mean_abs_attribution", ascending=False)
    edges["network_group"] = [
        _edge_network_group(first, second) for first, second in zip(edges["channel_i"], edges["channel_j"])
    ]
    edges["interhemispheric"] = [
        _is_interhemispheric(first, second) for first, second in zip(edges["channel_i"], edges["channel_j"])
    ]
    edges.to_csv(explain_root / "wpli_top_edges.csv", index=False)
    band = _aggregate_attribution(wpli_long, ["state", "band", "band_index"])
    band.to_csv(explain_root / "wpli_band_importance.csv", index=False)
    node = _wpli_node_importance(wpli_long)
    node.to_csv(explain_root / "wpli_node_importance.csv", index=False)
    network = edges.groupby(["network_group", "band"], as_index=False).agg(
        mean_abs_attribution=("mean_abs_attribution", "mean"),
        mean_signed_attribution=("mean_signed_attribution", "mean"),
        n_edges=("edge_index", "count"),
    )
    network.to_csv(explain_root / "wpli_network_group_importance.csv", index=False)
    return {"top_edge_band": edges, "top_edges": edges}


def _aggregate_attribution(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = frame.groupby(group_cols, as_index=False).agg(
        mean_signed_attribution=("signed_attribution", "mean"),
        mean_abs_attribution=("abs_attribution", "mean"),
        std_abs_attribution=("abs_attribution", "std"),
        n_samples=("abs_attribution", "size"),
    )
    positive = frame[frame["y_true"] == 1].groupby(group_cols, as_index=False)["abs_attribution"].mean()
    positive = positive.rename(columns={"abs_attribution": "positive_mean_abs_attribution"})
    negative = frame[frame["y_true"] == 0].groupby(group_cols, as_index=False)["abs_attribution"].mean()
    negative = negative.rename(columns={"abs_attribution": "negative_mean_abs_attribution"})
    correct = frame[frame["correct"] == 1].groupby(group_cols, as_index=False)["abs_attribution"].mean()
    correct = correct.rename(columns={"abs_attribution": "correct_only_mean_abs_attribution"})
    rows = rows.merge(positive, on=group_cols, how="left").merge(negative, on=group_cols, how="left").merge(correct, on=group_cols, how="left")
    return rows


def _wpli_node_importance(wpli_long: pd.DataFrame) -> pd.DataFrame:
    first = wpli_long.rename(columns={"channel_i": "channel"})[["state", "band", "channel", "abs_attribution", "signed_attribution"]]
    second = wpli_long.rename(columns={"channel_j": "channel"})[["state", "band", "channel", "abs_attribution", "signed_attribution"]]
    nodes = pd.concat([first, second], ignore_index=True)
    return nodes.groupby(["state", "band", "channel"], as_index=False).agg(
        node_importance=("abs_attribution", "mean"),
        mean_signed_attribution=("signed_attribution", "mean"),
    ).sort_values("node_importance", ascending=False)


def _write_stability_outputs(psd_seed_rows: list[pd.DataFrame], wpli_seed_rows: list[pd.DataFrame], explain_root: Path) -> None:
    psd_seed = pd.concat(psd_seed_rows, ignore_index=True)
    wpli_seed = pd.concat(wpli_seed_rows, ignore_index=True)
    psd_by_seed = psd_seed.groupby(["seed", "feature_id", "state", "channel", "band"], as_index=False).agg(
        mean_abs_attribution=("mean_abs_attribution", "mean"),
        mean_signed_attribution=("mean_signed_attribution", "mean"),
    )
    wpli_by_seed = wpli_seed.groupby(["seed", "feature_id", "state", "edge_index", "channel_i", "channel_j", "band"], as_index=False).agg(
        mean_abs_attribution=("mean_abs_attribution", "mean"),
        mean_signed_attribution=("mean_signed_attribution", "mean"),
    )
    psd_stability = compute_topk_selection_frequency(psd_by_seed, k_values=(10, 20, 50))
    wpli_stability = compute_topk_selection_frequency(wpli_by_seed, k_values=(10, 20, 50))
    psd_stability.to_csv(explain_root / "psd_seed_stability.csv", index=False)
    wpli_stability.to_csv(explain_root / "wpli_seed_stability.csv", index=False)
    stability = pd.concat(
        [
            psd_stability.assign(feature_family="PSD", stability_metric="topk_selection_frequency"),
            wpli_stability.assign(feature_family="WPLI", stability_metric="topk_selection_frequency"),
            _seed_pair_stability_rows(psd_by_seed, feature_family="PSD"),
            _seed_pair_stability_rows(wpli_by_seed, feature_family="WPLI"),
            _top_feature_bootstrap_rows(psd_seed, feature_family="PSD", top_n=20),
            _top_feature_bootstrap_rows(wpli_seed, feature_family="WPLI", top_n=20),
        ],
        ignore_index=True,
    )
    stability.to_csv(explain_root / "attribution_stability_summary.csv", index=False)


def _seed_pair_stability_rows(feature_table: pd.DataFrame, *, feature_family: str, k: int = 20) -> pd.DataFrame:
    seeds = sorted(feature_table["seed"].drop_duplicates().tolist())
    rows = []
    for i, first in enumerate(seeds):
        first_frame = feature_table[feature_table["seed"] == first].set_index("feature_id")
        first_scores = first_frame["mean_abs_attribution"]
        first_top = set(first_scores.sort_values(ascending=False).head(k).index.astype(str))
        for second in seeds[i + 1 :]:
            second_frame = feature_table[feature_table["seed"] == second].set_index("feature_id")
            joined = pd.concat(
                [first_scores.rename("first"), second_frame["mean_abs_attribution"].rename("second")],
                axis=1,
                join="inner",
            ).fillna(0.0)
            corr, _ = _safe_spearman(joined["first"].to_numpy(float), joined["second"].to_numpy(float))
            second_top = set(second_frame["mean_abs_attribution"].sort_values(ascending=False).head(k).index.astype(str))
            union = first_top | second_top
            rows.append(
                {
                    "feature_family": feature_family,
                    "stability_metric": "seed_pair_spearman_jaccard",
                    "feature_id": f"seed_pair:{first}-{second}",
                    "k": int(k),
                    "seed_a": int(first),
                    "seed_b": int(second),
                    "spearman_corr": corr,
                    "jaccard_similarity": float(len(first_top & second_top) / len(union)) if union else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _top_feature_bootstrap_rows(feature_table: pd.DataFrame, *, feature_family: str, top_n: int, n_bootstrap: int = 500) -> pd.DataFrame:
    if "subject_id" not in feature_table.columns:
        return pd.DataFrame()
    top_features = (
        feature_table.groupby("feature_id", as_index=False)["mean_abs_attribution"].mean()
        .sort_values("mean_abs_attribution", ascending=False)
        .head(top_n)
    )
    rows = []
    for _, feature in top_features.iterrows():
        values = (
            feature_table[feature_table["feature_id"] == feature["feature_id"]]
            .groupby("subject_id")["mean_abs_attribution"]
            .mean()
            .to_numpy(float)
        )
        mean, low, high = _bootstrap_mean_ci(values, seed=29)
        rows.append(
            {
                "feature_family": feature_family,
                "stability_metric": "subject_bootstrap_ci",
                "feature_id": str(feature["feature_id"]),
                "mean_abs_attribution": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "n_subjects": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _write_occlusion_outputs(
    *,
    output_root: Path,
    model_group: str,
    seeds: Iterable[int],
    label_table: pd.DataFrame,
    targets: pd.DataFrame,
    record_by_subject: Mapping[str, Any],
    device: torch.device,
    frequency_bins: np.ndarray,
    edge_list: tuple[tuple[str, str], ...],
    top_psd_features: pd.DataFrame,
    top_wpli_features: pd.DataFrame,
    occlusion_top_k: int,
    max_folds: int = 0,
) -> None:
    explain_root = output_root / "results" / "explainability"
    branch_rows = []
    psd_rows = []
    wpli_rows = []
    top_psd = top_psd_features.head(occlusion_top_k)
    top_wpli = top_wpli_features.head(occlusion_top_k)
    processed = 0
    for seed in seeds:
        for checkpoint_path in sorted(_checkpoint_dir(output_root, model_group).glob(f"{model_group}_seed{seed}_fold*_test_*.pt")):
            if max_folds and processed >= max_folds:
                break
            payload = _load_checkpoint(checkpoint_path, device)
            metadata = payload["metadata"]
            sample = _sample_from_metadata(metadata, label_table, targets, output_root=output_root)
            model = _build_model_from_metadata(metadata, payload["state_dict"], device)
            scaler = _scaler_from_payload(payload["state_scaler"])
            batch = _make_batch([record_by_subject[sample.subject_id]], scaler, device, "multimodal")
            occlusion_specs: list[tuple[str, str, str, Any]] = []
            for group_name, keys in {
                "PSD branch": ("psd_eo", "psd_ec"),
                "WPLI branch": ("wpli_eo", "wpli_ec"),
                "EO state": ("psd_eo", "wpli_eo"),
                "EC state": ("psd_ec", "wpli_ec"),
            }.items():
                occlusion_specs.append(("branch", "branch_state", group_name, lambda occluded, keys=keys: _zero_keys(occluded, keys)))
            for _, feature in top_psd.iterrows():
                occlusion_specs.append(
                    (
                        "psd",
                        "psd_channel_band",
                        f"{feature['state']}|{feature['channel']}|{feature['band']}",
                        lambda occluded, feature=feature: _zero_psd_channel_band(occluded, feature, frequency_bins),
                    )
                )
            for band_name, _, _ in PSD_BANDS:
                occlusion_specs.append(
                    (
                        "psd",
                        "psd_band",
                        band_name,
                        lambda occluded, band_name=band_name: _zero_psd_band(occluded, band_name, frequency_bins),
                    )
                )
            for channel_index, channel in enumerate(CANONICAL_CHANNELS_62):
                occlusion_specs.append(
                    (
                        "psd",
                        "psd_channel",
                        str(channel),
                        lambda occluded, channel_index=channel_index: _zero_psd_channel(occluded, channel_index),
                    )
                )
            for band in tuple(name for name, _ in connectivity_bands()):
                occlusion_specs.append(
                    (
                        "wpli",
                        "wpli_band",
                        band,
                        lambda occluded, band=band: _zero_wpli_band(occluded, band),
                    )
                )
            for _, feature in top_wpli.iterrows():
                occlusion_specs.append(
                    (
                        "wpli",
                        "wpli_edge_band",
                        f"{feature['state']}|{feature['channel_i']}-{feature['channel_j']}|{feature['band']}",
                        lambda occluded, feature=feature: _zero_wpli_edge_band(occluded, feature),
                    )
                )
            for node_index, channel in enumerate(CANONICAL_CHANNELS_62):
                occlusion_specs.append(
                    (
                        "wpli",
                        "wpli_node",
                        str(channel),
                        lambda occluded, node_index=node_index: _zero_wpli_node(occluded, node_index, edge_list),
                    )
                )
            for group_name in ("interhemispheric", "intrahemispheric"):
                occlusion_specs.append(
                    (
                        "wpli",
                        "wpli_hemisphere_group",
                        group_name,
                        lambda occluded, group_name=group_name: _zero_wpli_hemisphere_group(occluded, group_name, edge_list),
                    )
                )
            for network_group in sorted({_edge_network_group(first, second) for first, second in edge_list}):
                occlusion_specs.append(
                    (
                        "wpli",
                        "wpli_network_group",
                        network_group,
                        lambda occluded, network_group=network_group: _zero_wpli_network_group(occluded, network_group, edge_list),
                    )
                )
            occluded_rows = _batched_occlusion_rows(model, batch, sample, occlusion_specs)
            branch_rows.extend(occluded_rows["branch"])
            psd_rows.extend(occluded_rows["psd"])
            wpli_rows.extend(occluded_rows["wpli"])
            processed += 1
        if max_folds and processed >= max_folds:
            break
    pd.DataFrame(branch_rows).to_csv(explain_root / "occlusion_branch_state.csv", index=False)
    pd.DataFrame(psd_rows).to_csv(explain_root / "occlusion_psd_band_channel.csv", index=False)
    pd.DataFrame(wpli_rows).to_csv(explain_root / "occlusion_wpli_edge_node_band.csv", index=False)


def _append_branch_gate_stability(explain_root: Path) -> None:
    path = explain_root / "attribution_stability_summary.csv"
    if path.exists():
        stability = pd.read_csv(path)
    else:
        stability = pd.DataFrame()
    rows = []
    gate_path = explain_root / "branch_state_gate_weights.csv"
    if gate_path.exists():
        gates = pd.read_csv(gate_path)
        for (branch, state), frame in gates.groupby(["branch", "state"]):
            values = frame.groupby("subject_id")["gate_weight"].mean().to_numpy(float)
            mean, low, high = _bootstrap_mean_ci(values, seed=31)
            rows.append(
                {
                    "feature_family": "gate",
                    "stability_metric": "subject_bootstrap_ci",
                    "feature_id": f"{branch}|{state}|gate_weight",
                    "mean_abs_attribution": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "n_subjects": int(len(values)),
                }
            )
    branch_path = explain_root / "occlusion_branch_state.csv"
    if branch_path.exists():
        branch = pd.read_csv(branch_path)
        for group_name, frame in branch.groupby("group_name"):
            values = frame.groupby("subject_id")["delta_probability"].mean().to_numpy(float)
            mean, low, high = _bootstrap_mean_ci(values, seed=37)
            rows.append(
                {
                    "feature_family": "occlusion",
                    "stability_metric": "subject_bootstrap_ci",
                    "feature_id": f"{group_name}|delta_probability",
                    "mean_abs_attribution": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "n_subjects": int(len(values)),
                }
            )
    if rows:
        stability = pd.concat([stability, pd.DataFrame(rows)], ignore_index=True)
        stability.to_csv(path, index=False)


def _batched_occlusion_rows(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    sample: ExplainedSample,
    specs: list[tuple[str, str, str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows = {"branch": [], "psd": [], "wpli": []}
    if not specs:
        return rows
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        original_logit = float(classification_logit(model, batch).reshape(-1).mean().detach().cpu().item())
        original_probability = float(classification_probability(model, batch).reshape(-1).mean().detach().cpu().item())
        occluded_batches = []
        for _, _, _, occlude in specs:
            occluded = clone_tensor_batch(batch)
            occlude(occluded)
            occluded_batches.append(occluded)
        stacked = {
            key: torch.cat([occluded[key] for occluded in occluded_batches], dim=0)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        occluded_logits = classification_logit(model, stacked).reshape(-1).detach().cpu().numpy()
        occluded_probabilities = classification_probability(model, stacked).reshape(-1).detach().cpu().numpy()
    model.train(model_was_training)
    for index, (family, group_type, group_name, _) in enumerate(specs):
        result = {
            "original_logit": original_logit,
            "occluded_logit": float(occluded_logits[index]),
            "delta_logit": float(original_logit - float(occluded_logits[index])),
            "original_probability": original_probability,
            "occluded_probability": float(occluded_probabilities[index]),
            "delta_probability": float(original_probability - float(occluded_probabilities[index])),
        }
        rows[family].append(_occlusion_row(sample, group_type, group_name, result))
    return rows


def _occlusion_row(sample: ExplainedSample, group_type: str, group_name: str, result: Mapping[str, float]) -> dict[str, Any]:
    original_loss = _binary_loss(result["original_probability"], sample.y_true)
    occluded_loss = _binary_loss(result["occluded_probability"], sample.y_true)
    return {
        "subject_id": sample.subject_id,
        "fold_index": sample.fold_index,
        "seed": sample.seed,
        "group_type": group_type,
        "group_name": group_name,
        "y_true": sample.y_true,
        "y_score": sample.y_score,
        "y_pred": sample.y_pred,
        "correct": int(sample.y_true == sample.y_pred),
        "original_loss": original_loss,
        "occluded_loss": occluded_loss,
        "delta_loss": occluded_loss - original_loss,
        **result,
    }


def _binary_loss(probability: float, target: int) -> float:
    probability = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    target = int(target)
    return float(-(target * np.log(probability) + (1 - target) * np.log(1.0 - probability)))


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_bootstrap: int = 500) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = float(np.mean(sample))
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _zero_keys(batch: dict[str, torch.Tensor], keys: Iterable[str]) -> None:
    for key in keys:
        batch[key].zero_()


def _zero_psd_band(batch: dict[str, torch.Tensor], band_name: str, frequency_bins: np.ndarray) -> None:
    mask = _psd_frequency_mask(str(band_name), frequency_bins)
    for key in ("psd_eo", "psd_ec"):
        batch[key][:, :, torch.as_tensor(mask, device=batch[key].device)] = 0.0


def _zero_psd_channel_band(batch: dict[str, torch.Tensor], feature: pd.Series, frequency_bins: np.ndarray) -> None:
    key = "psd_eo" if feature["state"] == "EO" else "psd_ec"
    channel_index = int(feature["channel_index"])
    mask = torch.as_tensor(_psd_frequency_mask(str(feature["band"]), frequency_bins), device=batch[key].device)
    batch[key][:, channel_index, mask] = 0.0


def _psd_frequency_mask(band_name: str, frequency_bins: np.ndarray) -> np.ndarray:
    for name, low, high in PSD_BANDS:
        if band_name == name:
            return (frequency_bins >= low) & (frequency_bins < high)
    if band_name == "Other":
        known = np.zeros_like(frequency_bins, dtype=bool)
        for _, low, high in PSD_BANDS:
            known |= (frequency_bins >= low) & (frequency_bins < high)
        return ~known
    raise ValueError(f"Unknown PSD band {band_name!r}")


def _zero_psd_channel(batch: dict[str, torch.Tensor], channel_index: int) -> None:
    for key in ("psd_eo", "psd_ec"):
        batch[key][:, int(channel_index), :] = 0.0


def _zero_wpli_band(batch: dict[str, torch.Tensor], band_name: str) -> None:
    band_names = tuple(name for name, _ in connectivity_bands())
    band_index = band_names.index(band_name)
    for key in ("wpli_eo", "wpli_ec"):
        batch[key][:, :, band_index] = 0.0


def _zero_wpli_edge_band(batch: dict[str, torch.Tensor], feature: pd.Series) -> None:
    key = "wpli_eo" if feature["state"] == "EO" else "wpli_ec"
    batch[key][:, int(feature["edge_index"]), int(feature["band_index"])] = 0.0


def _zero_wpli_node(batch: dict[str, torch.Tensor], node_index: int, edge_list: tuple[tuple[str, str], ...]) -> None:
    channel = CANONICAL_CHANNELS_62[int(node_index)]
    mask = torch.as_tensor(
        [first == channel or second == channel for first, second in edge_list],
        device=batch["wpli_eo"].device,
    )
    for key in ("wpli_eo", "wpli_ec"):
        batch[key][:, mask, :] = 0.0


def _zero_wpli_hemisphere_group(batch: dict[str, torch.Tensor], group_name: str, edge_list: tuple[tuple[str, str], ...]) -> None:
    want_inter = group_name == "interhemispheric"
    mask = torch.as_tensor(
        [_is_interhemispheric(first, second) == want_inter for first, second in edge_list],
        device=batch["wpli_eo"].device,
    )
    for key in ("wpli_eo", "wpli_ec"):
        batch[key][:, mask, :] = 0.0


def _zero_wpli_network_group(batch: dict[str, torch.Tensor], network_group: str, edge_list: tuple[tuple[str, str], ...]) -> None:
    mask = torch.as_tensor(
        [_edge_network_group(first, second) == network_group for first, second in edge_list],
        device=batch["wpli_eo"].device,
    )
    for key in ("wpli_eo", "wpli_ec"):
        batch[key][:, mask, :] = 0.0


def _write_biomarker_validation(
    *,
    record_by_subject: Mapping[str, Any],
    label_table: pd.DataFrame,
    targets: pd.DataFrame,
    top_psd_features: pd.DataFrame,
    top_wpli_features: pd.DataFrame,
    frequency_bins: np.ndarray,
    edge_list: tuple[tuple[str, str], ...],
    wpli_band_names: tuple[str, ...],
    explain_root: Path,
) -> None:
    psd_rows = []
    for _, feature in top_psd_features.iterrows():
        values, clinical = _raw_psd_feature_values(record_by_subject, label_table, targets, feature, frequency_bins)
        psd_rows.append({**feature.to_dict(), **_clinical_association(values, clinical)})
    pd.DataFrame(_fdr_correct(psd_rows)).to_csv(explain_root / "psd_biomarker_validation.csv", index=False)
    wpli_rows = []
    for _, feature in top_wpli_features.iterrows():
        values, clinical = _raw_wpli_feature_values(record_by_subject, label_table, targets, feature, edge_list, wpli_band_names)
        wpli_rows.append({**feature.to_dict(), **_clinical_association(values, clinical)})
    pd.DataFrame(_fdr_correct(wpli_rows)).to_csv(explain_root / "wpli_biomarker_validation.csv", index=False)


def _raw_psd_feature_values(record_by_subject: Mapping[str, Any], label_table: pd.DataFrame, targets: pd.DataFrame, feature: pd.Series, frequency_bins: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    state_index = 0 if feature["state"] == "EO" else 1
    channel_index = int(feature["channel_index"])
    frequency_bin = int(feature["frequency_bin"])
    values = []
    subjects = []
    for subject_id, record in record_by_subject.items():
        array = record.modalities["psd"][state_index]
        values.append(float(array[channel_index, frequency_bin]))
        subjects.append(subject_id)
    clinical = _clinical_frame(subjects, label_table, targets)
    return np.asarray(values, dtype=float), clinical


def _raw_wpli_feature_values(record_by_subject: Mapping[str, Any], label_table: pd.DataFrame, targets: pd.DataFrame, feature: pd.Series, edge_list: tuple[tuple[str, str], ...], band_names: tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    state_index = 0 if feature["state"] == "EO" else 1
    edge_index = int(feature["edge_index"])
    band_index = int(feature["band_index"])
    values = []
    subjects = []
    for subject_id, record in record_by_subject.items():
        array = record.modalities["wpli"][state_index]
        values.append(float(array[edge_index, band_index]))
        subjects.append(subject_id)
    clinical = _clinical_frame(subjects, label_table, targets)
    return np.asarray(values, dtype=float), clinical


def _clinical_frame(subjects: list[str], label_table: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id in subjects:
        row = label_table.loc[subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "label": int(row["label"]),
                "residual": float(row["Residual"]),
                "signed_distance": float(targets.loc[subject_id, "signed_distance"]),
                "FMA_pre": float(row["FMA_pre"]),
                "FMA_post": float(row["FMA_post"]),
                "observed_delta": float(row["Delta_FMA_obs"]),
                "predicted_delta": float(row["Delta_FMA_pred"]),
            }
        )
    return pd.DataFrame(rows)


def _clinical_association(values: np.ndarray, clinical: pd.DataFrame) -> dict[str, float]:
    residual_r, residual_p = _safe_spearman(values, clinical["residual"].to_numpy(float))
    distance_r, distance_p = _safe_spearman(values, clinical["signed_distance"].to_numpy(float))
    group_p = _permutation_group_pvalue(values, clinical["label"].to_numpy(int), seed=13)
    return {
        "spearman_residual_r": residual_r,
        "spearman_residual_p": residual_p,
        "spearman_signed_distance_r": distance_r,
        "spearman_signed_distance_p": distance_p,
        "label_group_permutation_p": group_p,
        "positive_mean_value": float(np.mean(values[clinical["label"].to_numpy(int) == 1])),
        "negative_mean_value": float(np.mean(values[clinical["label"].to_numpy(int) == 0])),
    }


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    r = 0.0 if not np.isfinite(result.statistic) else float(result.statistic)
    p = 1.0 if not np.isfinite(result.pvalue) else float(result.pvalue)
    return r, p


def _permutation_group_pvalue(values: np.ndarray, labels: np.ndarray, *, seed: int, n_perm: int = 1000) -> float:
    rng = np.random.default_rng(seed)
    labels = labels.astype(int)
    observed = abs(float(np.mean(values[labels == 1]) - np.mean(values[labels == 0])))
    count = 1
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        diff = abs(float(np.mean(values[shuffled == 1]) - np.mean(values[shuffled == 0])))
        if diff >= observed:
            count += 1
    return float(count / (n_perm + 1))


def _fdr_correct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    p_values = np.asarray([row.get("spearman_signed_distance_p", 1.0) for row in rows], dtype=float)
    order = np.argsort(p_values)
    adjusted = np.ones_like(p_values)
    running = 1.0
    m = len(p_values)
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["spearman_signed_distance_fdr_p"] = float(min(1.0, value))
    return rows


def _make_figures(explain_root: Path, figure_root: Path) -> None:
    psd_long = pd.read_csv(explain_root / "psd_attribution_long.csv")
    wpli_edges = pd.read_csv(explain_root / "wpli_top_edges.csv")
    node = pd.read_csv(explain_root / "wpli_node_importance.csv")
    occlusion = pd.read_csv(explain_root / "occlusion_branch_state.csv")
    stability = pd.read_csv(explain_root / "attribution_stability_summary.csv")
    for label, subset in {
        "all": psd_long,
        "positive": psd_long[psd_long["y_true"] == 1],
        "negative": psd_long[psd_long["y_true"] == 0],
    }.items():
        heatmap = subset.groupby(["channel_index", "frequency_bin"], as_index=False)["signed_attribution"].mean()
        matrix = heatmap.pivot(index="channel_index", columns="frequency_bin", values="signed_attribution").fillna(0.0).to_numpy()
        _save_heatmap(matrix, figure_root / f"psd_channel_frequency_heatmap_{label}.png", title=f"PSD attribution heatmap ({label})")
    psd_band = pd.read_csv(explain_root / "psd_channel_band_importance.csv")
    for band in [name for name, _, _ in PSD_BANDS]:
        subset = psd_band[psd_band["band"] == band].sort_values("mean_abs_attribution", ascending=False).head(20)
        _save_bar(subset["channel"] + "-" + subset["state"], subset["mean_abs_attribution"], figure_root / f"psd_band_{_safe_name(band)}_top_channels.png", f"PSD {band} top channels")
    for band in ("Alpha", "Beta Medium", "Theta"):
        subset = wpli_edges[wpli_edges["band"] == band].head(20)
        _save_connectome(subset, figure_root / f"wpli_connectome_top20_{_safe_name(band)}.png", f"WPLI top 20 {band} edges")
    node_top = node.groupby("channel", as_index=False)["node_importance"].mean().sort_values("node_importance", ascending=False).head(25)
    _save_bar(node_top["channel"], node_top["node_importance"], figure_root / "wpli_node_importance_top25.png", "WPLI node importance")
    occ = occlusion.groupby("group_name", as_index=False)["delta_probability"].mean().sort_values("delta_probability", ascending=False)
    _save_bar(occ["group_name"], occ["delta_probability"], figure_root / "branch_state_occlusion_barplot.png", "Branch/state occlusion")
    stab = stability[stability["k"] == 20].sort_values("selection_frequency", ascending=False).head(30)
    _save_bar(stab["feature_family"] + ":" + stab["feature_id"], stab["selection_frequency"], figure_root / "stability_topk_selection_frequency.png", "Top-20 selection frequency")
    corr_values = _seed_correlation_summary(psd_long)
    if corr_values.empty:
        _save_bar(["insufficient seeds"], [0.0], figure_root / "stability_seed_to_seed_correlation.png", "Seed-to-seed attribution correlation")
    else:
        _save_bar(corr_values["pair"], corr_values["spearman_corr"], figure_root / "stability_seed_to_seed_correlation.png", "Seed-to-seed attribution correlation")
    _make_error_subject_figure(psd_long, figure_root)


def _save_heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 7))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Mean signed attribution")
    plt.title(title)
    plt.xlabel("Frequency bin")
    plt.ylabel("Channel index")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_bar(labels: Iterable[Any], values: Iterable[float], path: Path, title: str) -> None:
    labels = [str(label) for label in labels]
    values = list(values)
    plt.figure(figsize=(max(8, min(18, len(labels) * 0.35)), 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=75, ha="right", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_connectome(edges: pd.DataFrame, path: Path, title: str) -> None:
    channels = list(CANONICAL_CHANNELS_62)
    angles = np.linspace(0, 2 * np.pi, len(channels), endpoint=False)
    coords = {channel: (np.cos(angle), np.sin(angle)) for channel, angle in zip(channels, angles)}
    max_abs = float(edges["mean_abs_attribution"].max()) if len(edges) else 1.0
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for channel, (x, y) in coords.items():
        ax.scatter([x], [y], c="black", s=15)
        ax.text(x * 1.08, y * 1.08, channel, fontsize=6, ha="center", va="center")
    for _, edge in edges.iterrows():
        x1, y1 = coords[edge["channel_i"]]
        x2, y2 = coords[edge["channel_j"]]
        color = "red" if edge["mean_signed_attribution"] >= 0 else "blue"
        width = 0.5 + 4.0 * float(edge["mean_abs_attribution"]) / (max_abs + 1e-12)
        ax.plot([x1, x2], [y1, y2], color=color, alpha=0.55, linewidth=width)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _seed_correlation_summary(psd_long: pd.DataFrame) -> pd.DataFrame:
    pivot = psd_long.groupby(["seed", "state", "channel", "band"], as_index=False)["abs_attribution"].mean()
    seeds = sorted(pivot["seed"].unique())
    rows = []
    for i, first in enumerate(seeds):
        first_values = pivot[pivot["seed"] == first].sort_values(["state", "channel", "band"])["abs_attribution"].to_numpy()
        for second in seeds[i + 1 :]:
            second_values = pivot[pivot["seed"] == second].sort_values(["state", "channel", "band"])["abs_attribution"].to_numpy()
            corr, _ = _safe_spearman(first_values, second_values)
            rows.append({"pair": f"{first}-{second}", "spearman_corr": corr})
    return pd.DataFrame(rows, columns=["pair", "spearman_corr"])


def _make_error_subject_figure(psd_long: pd.DataFrame, figure_root: Path) -> None:
    watched = psd_long[psd_long["subject_id"].isin(WATCHED_SUBJECTS)]
    correct_negative = psd_long[(psd_long["y_true"] == 0) & (psd_long["correct"] == 1)]
    rows = []
    for label, frame in [("sub09/sub14", watched), ("correct negatives", correct_negative)]:
        if frame.empty:
            continue
        grouped = frame.groupby("band", as_index=False)["abs_attribution"].mean()
        grouped["group"] = label
        rows.append(grouped)
    if not rows:
        plot = pd.DataFrame(
            {
                "band": [name for name, _, _ in PSD_BANDS],
                "abs_attribution": [0.0 for _ in PSD_BANDS],
                "group": ["not available" for _ in PSD_BANDS],
            }
        )
    else:
        plot = pd.concat(rows, ignore_index=True)
    pivot = plot.pivot(index="band", columns="group", values="abs_attribution").fillna(0.0)
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Mean absolute PSD attribution")
    plt.title("sub09/sub14 attribution vs correctly classified negatives")
    plt.tight_layout()
    plt.savefig(figure_root / "sub09_sub14_error_attribution_comparison.png", dpi=200)
    plt.close()


def _write_document(explain_root: Path, figure_root: Path, output_root: Path) -> None:
    psd_top = pd.read_csv(explain_root / "psd_channel_frequency_top_features.csv").head(10)
    wpli_top = pd.read_csv(explain_root / "wpli_top_edges.csv").head(10)
    occlusion = pd.read_csv(explain_root / "occlusion_branch_state.csv").groupby("group_name", as_index=False)["delta_probability"].mean().sort_values("delta_probability", ascending=False)
    sanity = pd.read_csv(explain_root / "sanity_check_summary.csv")
    stability = pd.read_csv(explain_root / "attribution_stability_summary.csv")
    doc = output_root / "docs" / "explainability_residual_aware_ssl_cnn_results.md"
    lines = [
        "# Explainability: Residual-aware Patient-level Barlow SSL-CNN",
        "",
        "Explained model: `residualaware_highrank_swa_clsalpha1`, the Patient-level Barlow SSL-CNN with residual-aware auxiliary fine-tuning, SWA, and classification-head inference.",
        "",
        "The attribution target is the binary classification logit because residual regression, ranking, and soft-label heads are auxiliary training objectives only. The residual head is not interpreted as the main prediction target.",
        "",
        "Integrated Gradients used a zero baseline after fold-local scaling and targeted the classification logit. SmoothGrad averaged noisy IG samples. Occlusion zeroed branch, state, PSD band/channel-band, and WPLI band/top edge-band groups.",
        "",
        "## Top PSD Features",
        "",
        psd_top[["state", "channel", "frequency_hz", "band", "mean_signed_attribution", "mean_abs_attribution"]].to_markdown(index=False),
        "",
        "## Top WPLI Edges",
        "",
        wpli_top[["state", "channel_i", "channel_j", "band", "mean_signed_attribution", "mean_abs_attribution", "network_group"]].to_markdown(index=False),
        "",
        "## Branch and State Occlusion",
        "",
        occlusion.to_markdown(index=False),
        "",
        "## Stability",
        "",
        stability.sort_values("selection_frequency", ascending=False).head(20).to_markdown(index=False),
        "",
        "## Sanity Checks",
        "",
        sanity.to_markdown(index=False),
        "",
        "## Figures",
        "",
        f"Figures were written under `{figure_root.relative_to(output_root).as_posix()}/`, including PSD heatmaps, PSD band barplots, WPLI circular connectomes, node importance, occlusion, stability, and sub09/sub14 error attribution comparison.",
        "",
        "## Biomarker Validation",
        "",
        "Top attributed PSD and WPLI raw feature values were tested against label, residual, signed_distance, and FMA-derived variables using Spearman association and label-group permutation tests with FDR correction.",
        "",
        "## Interpretation and Limitations",
        "",
        "The outputs identify stable PSD channel-frequency and WPLI edge-band candidates for the selected model. Signed attribution is interpreted as positive when it pushes toward proportional recovery and negative when it pushes toward poor recovery. These findings are model explanations, not causal evidence. The cohort has n=19, seed/fold variability remains, and there is no external validation. Topomap figures are represented as channel barplots/circular connectomes because full electrode coordinate topomap layout is not implemented in this repository.",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")


def _write_task_status(explain_root: Path, output_root: Path, processed: int) -> None:
    path = output_root / "docs" / "task_status.md"
    lines = [
        "# Task Status",
        "",
        "## Explainability residual-aware SSL-CNN",
        "",
        f"- Explained seed/fold samples: {processed}",
        f"- Output directory: `{explain_root.relative_to(output_root).as_posix()}`",
        "- Model: `residualaware_highrank_swa_clsalpha1`",
        "- Attribution target: classification logit",
        "- Checkpoints: supervised fold checkpoints are required locally and are not intended for commit.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _gate_rows(sample: ExplainedSample, aux: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for branch in ("psd", "wpli"):
        key = f"{branch}_state_weights"
        if key not in aux:
            continue
        weights = aux[key].detach().cpu().numpy()[0]
        for state, weight in zip(("EO", "EC"), weights):
            rows.append(
                {
                    "subject_id": sample.subject_id,
                    "fold_index": sample.fold_index,
                    "seed": sample.seed,
                    "branch": branch,
                    "state": state,
                    "gate_weight": float(weight),
                    "y_true": sample.y_true,
                    "y_score": sample.y_score,
                    "y_pred": sample.y_pred,
                    "correct": int(sample.y_true == sample.y_pred),
                }
            )
    return rows


def _sanity_check_rows(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], sample: ExplainedSample, *, steps: int) -> list[dict[str, Any]]:
    trained = manual_integrated_gradients(model, batch, input_keys=("psd_eo", "wpli_eo"), steps=steps)
    permuted = dict(batch)
    permuted["psd_eo"] = batch["psd_eo"].flip(dims=(-1,))
    permuted["wpli_eo"] = batch["wpli_eo"].flip(dims=(1,))

    variants = [
        (
            "classifier_only_randomization",
            randomize_classifier_weights(model, seed=sample.seed + sample.fold_index + 17),
            batch,
        ),
        (
            "classifier_final_projection_randomization",
            _randomize_named_modules(
                model,
                seed=sample.seed + sample.fold_index + 23,
                predicate=lambda name, module: (
                    name.startswith("classifier")
                    or name.endswith("projection")
                    or ".projection." in name
                    or name.endswith("network.11")
                    or ".network.11" in name
                ),
            ),
            batch,
        ),
        (
            "full_psd_encoder_randomization",
            _randomize_named_modules(
                model,
                seed=sample.seed + sample.fold_index + 31,
                predicate=lambda name, module: name.startswith("branch_models.psd"),
            ),
            batch,
        ),
        (
            "full_wpli_encoder_randomization",
            _randomize_named_modules(
                model,
                seed=sample.seed + sample.fold_index + 37,
                predicate=lambda name, module: name.startswith("branch_models.wpli"),
            ),
            batch,
        ),
        ("input_permutation", model, permuted),
    ]
    rows = []
    for check_name, variant_model, variant_batch in variants:
        variant_attr = manual_integrated_gradients(variant_model, variant_batch, input_keys=("psd_eo", "wpli_eo"), steps=steps)
        row = _sanity_row_from_attribution(sample, check_name, trained, variant_attr)
        rows.append(row)
    return rows


def _sanity_row_from_attribution(
    sample: ExplainedSample,
    check_name: str,
    trained: Mapping[str, torch.Tensor],
    variant_attr: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    psd_signed = attribution_correlation(trained["psd_eo"], variant_attr["psd_eo"])
    wpli_signed = attribution_correlation(trained["wpli_eo"], variant_attr["wpli_eo"])
    psd_abs = attribution_correlation(trained["psd_eo"].abs(), variant_attr["psd_eo"].abs())
    wpli_abs = attribution_correlation(trained["wpli_eo"].abs(), variant_attr["wpli_eo"].abs())
    return {
        "subject_id": sample.subject_id,
        "seed": sample.seed,
        "fold_index": sample.fold_index,
        "sanity_check": check_name,
        "psd_eo_signed_correlation": psd_signed,
        "wpli_eo_signed_correlation": wpli_signed,
        "psd_eo_abs_correlation": psd_abs,
        "wpli_eo_abs_correlation": wpli_abs,
        # Backward-compatible columns used by the existing documentation summary.
        "psd_eo_correlation": psd_signed,
        "wpli_eo_correlation": wpli_signed,
    }


def _randomize_named_modules(
    model: torch.nn.Module,
    *,
    seed: int,
    predicate: Any,
) -> torch.nn.Module:
    randomized = copy.deepcopy(model)
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        for name, module in randomized.named_modules():
            if not name or not predicate(name, module):
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    finally:
        torch.random.set_rng_state(old_state)
    return randomized


def _edge_network_group(first: str, second: str) -> str:
    group_first = _channel_group(first)
    group_second = _channel_group(second)
    if group_first == group_second:
        return group_first
    return "|".join(sorted((group_first, group_second)))


def _channel_group(channel: str) -> str:
    token = "".join(ch for ch in channel.upper() if ch.isalpha())
    if token.startswith(("FP", "AF", "F", "FT", "FC")):
        return "frontal"
    if token.startswith(("C",)):
        return "central"
    if token.startswith(("CP", "P", "PO")):
        return "parietal"
    if token.startswith(("O", "CB")):
        return "occipital"
    if token.startswith(("T", "TP")):
        return "temporal"
    return "other"


def _is_interhemispheric(first: str, second: str) -> bool:
    return _hemisphere(first) != _hemisphere(second) and "midline" not in {_hemisphere(first), _hemisphere(second)}


def _hemisphere(channel: str) -> str:
    digits = "".join(ch for ch in channel if ch.isdigit())
    if not digits:
        return "midline"
    return "left" if int(digits[-1]) % 2 == 1 else "right"


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
