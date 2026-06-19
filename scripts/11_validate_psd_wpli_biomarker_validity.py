from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.validation.feature_overlap import (
    compute_feature_overlap,
    load_model_attribution_features,
    significant_feature_set,
)
from eeg_recovery.validation.healthy_reference import (
    LoadedFeature,
    aggregate_feature_groups,
    build_healthy_reference,
    compute_health_distance,
    compute_normalization_index,
    feature_band_distances,
    flatten_feature_values,
    load_feature,
    transform_to_healthy_zscore,
)
from eeg_recovery.validation.statistics import (
    add_fdr_column,
    paired_wilcoxon_test,
    partial_spearman_association,
    permutation_group_p_value,
    spearman_association,
    two_sample_test,
)
from eeg_recovery.visualization.biomarker_validity_plots import (
    make_biomarker_validity_summary_figure,
    plot_baseline_health_distance_group_boxplot,
    plot_feature_evidence_overlap,
    plot_health_distance_slopeplot,
    plot_normalization_scatter,
    plot_psd_topomap_normalization,
    plot_wpli_connectome_normalization,
)


META_COLUMN_MAP = {
    "编号": "subject_id",
    "受试者编号": "subject_id",
    "患者ID": "subject_id",
    "年龄": "age",
    "病程": "disease_duration",
    "性别": "sex",
    "患病侧（手）": "affected_hand",
    "患病侧": "affected_hand",
    "治疗前FMA": "FMA_pre",
    "治疗后FMA": "FMA_post",
    "治疗前MBI": "MBI_pre",
    "治疗后MBI": "MBI_post",
    "缺少数据": "missing_data",
    "脱落原因": "dropout_reason",
    "核磁次数": "mri_count",
}

SAFE_PATIENT_OUTPUT_COLUMNS = [
    "subject_id",
    "age",
    "sex",
    "disease_duration",
    "affected_hand",
    "FMA_pre",
    "FMA_post",
    "Delta_FMA",
    "proportional_recovery_residual",
    "recovery_label",
    "recovery_group",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PSD/wPLI biomarker validity against healthy templates and recovery outcomes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True, help="Clinical/healthy metadata Excel workbook.")
    parser.add_argument("--features-dir", required=True, help="Existing PSD/FC feature root.")
    parser.add_argument("--explainability-dir", default=None, help="Existing model explainability output directory.")
    parser.add_argument("--out-dir", default="results/biomarker_validity")
    parser.add_argument("--states", nargs="+", default=["EO", "EC"])
    parser.add_argument("--modalities", nargs="+", default=["psd", "wpli"])
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--robust-reference", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    patients, healthy, skipped = load_validation_metadata(args.metadata)
    model_features = (
        load_model_attribution_features(args.explainability_dir)
        if args.explainability_dir
        else set()
    )
    tables = build_validation_tables(
        patients=patients,
        healthy=healthy,
        features_dir=Path(args.features_dir),
        states=tuple(args.states),
        modalities=tuple(args.modalities),
        model_attribution_features=model_features,
        robust_reference=bool(args.robust_reference),
    )
    skipped_all = pd.concat([skipped, tables["skipped_subjects"]], ignore_index=True, sort=False)

    distances = tables["distances"]
    normalization = tables["normalization"]
    feature_measurements = tables["feature_measurements"]
    feature_normalization = tables["feature_normalization"]

    abnormality = baseline_abnormality_stats(distances, feature_measurements)
    baseline_assoc = baseline_outcome_association_stats(distances, feature_measurements, patients)
    pre_post = pre_post_normalization_stats(
        normalization,
        feature_normalization,
        n_bootstrap=args.n_bootstrap,
    )
    norm_assoc = normalization_outcome_correlation_stats(normalization, feature_normalization, patients)
    group_diff = normalization_group_difference_stats(
        normalization,
        feature_normalization,
        n_permutation=args.n_permutation,
    )
    overlap = feature_overlap_stats(
        model_features=model_features,
        abnormality=abnormality,
        pre_post=pre_post,
        baseline_assoc=baseline_assoc,
        norm_assoc=norm_assoc,
        feature_measurements=feature_measurements,
        n_permutation=args.n_permutation,
    )

    _write_csv(distances, out_dir / "healthy_reference_distances.csv")
    _write_csv(normalization, out_dir / "normalization_indices.csv")
    _write_csv(abnormality, out_dir / "baseline_abnormality_stats.csv")
    _write_csv(baseline_assoc, out_dir / "baseline_outcome_association_stats.csv")
    _write_csv(pre_post, out_dir / "pre_post_normalization_stats.csv")
    _write_csv(norm_assoc, out_dir / "normalization_outcome_correlation_stats.csv")
    _write_csv(group_diff, out_dir / "normalization_group_difference_stats.csv")
    _write_csv(overlap, out_dir / "feature_evidence_overlap_stats.csv")
    _write_csv(feature_normalization, out_dir / "feature_normalization_indices.csv")
    _write_csv(skipped_all, out_dir / "skipped_subjects.csv")

    figure_paths = make_figures(distances, normalization, feature_normalization, overlap, figure_dir)
    write_report(
        out_dir / "biomarker_validity_report.md",
        patients=patients,
        healthy=healthy,
        skipped=skipped_all,
        distances=distances,
        normalization=normalization,
        abnormality=abnormality,
        baseline_assoc=baseline_assoc,
        pre_post=pre_post,
        norm_assoc=norm_assoc,
        group_diff=group_diff,
        overlap=overlap,
        figure_paths=figure_paths,
    )
    print(f"Wrote biomarker validity outputs to {out_dir}")


def load_validation_metadata(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load patient/healthy metadata, strip identifiers, and derive recovery outcomes."""

    frame = _read_metadata_excel(path)
    frame = frame.rename(columns=_canonical_metadata_column)
    if "subject_id" not in frame.columns:
        raise ValueError(f"Metadata workbook is missing subject ID column: {path}")
    frame = frame.dropna(subset=["subject_id"]).copy()
    keep_columns = [column for column in dict.fromkeys(META_COLUMN_MAP.values()) if column in frame.columns]
    frame = frame.loc[:, keep_columns].copy()
    for column in ["age", "disease_duration", "FMA_pre", "FMA_post", "MBI_pre", "MBI_post"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column].map(_extract_numeric_token), errors="coerce")
        else:
            frame[column] = np.nan
    for column in ["sex", "affected_hand", "missing_data", "dropout_reason", "mri_count"]:
        if column not in frame.columns:
            frame[column] = np.nan

    skipped_rows: list[dict[str, object]] = []
    patient_mask = (
        frame["affected_hand"].notna()
        | frame["FMA_pre"].notna()
        | frame["FMA_post"].notna()
        | frame["MBI_pre"].notna()
    )
    patients = frame.loc[patient_mask].copy()
    healthy = frame.loc[~patient_mask].copy()

    normalized_ids = []
    for raw_id in patients["subject_id"]:
        try:
            normalized_ids.append(normalize_subject_id(raw_id))
        except ValueError as exc:
            skipped_rows.append(
                {"subject_id": str(raw_id), "analysis": "metadata", "reason": str(exc)}
            )
            normalized_ids.append(np.nan)
    patients["subject_id"] = normalized_ids
    patients = patients.dropna(subset=["subject_id"]).copy()
    healthy["subject_id"] = healthy["subject_id"].map(_sanitize_healthy_subject_id)
    healthy = healthy[
        healthy["subject_id"].astype(str).str.contains(r"sub\s*\d+|hc\s*\d+|healthy\s*\d+", case=False, regex=True)
    ].copy()

    patients["Delta_FMA"] = patients["FMA_post"] - patients["FMA_pre"]
    patients["Delta_FMA_pred"] = 0.7 * (66.0 - patients["FMA_pre"])
    patients["proportional_recovery_residual"] = patients["Delta_FMA_pred"] - patients["Delta_FMA"]
    valid_residual = patients["proportional_recovery_residual"].dropna()
    if valid_residual.empty:
        patients["recovery_label"] = np.nan
        patients["recovery_group"] = "unknown"
    else:
        median_residual = float(valid_residual.median())
        patients["recovery_label"] = (
            patients["proportional_recovery_residual"] <= median_residual
        ).astype("Int64")
        patients["recovery_group"] = np.where(
            patients["recovery_label"].eq(1),
            "proportional_recovery",
            "poor_recovery",
        )
        patients.loc[patients["recovery_label"].isna(), "recovery_group"] = "unknown"

    for row in patients[patients["FMA_post"].isna() | patients["FMA_pre"].isna()].itertuples(index=False):
        skipped_rows.append(
            {
                "subject_id": row.subject_id,
                "analysis": "outcome_association",
                "reason": "missing FMA_pre or FMA_post; skipped outcome-dependent statistics",
            }
        )

    patients = patients[[column for column in SAFE_PATIENT_OUTPUT_COLUMNS if column in patients.columns]].copy()
    healthy = healthy[["subject_id", "age", "sex", "mri_count"]].copy()
    skipped = pd.DataFrame(skipped_rows, columns=["subject_id", "analysis", "reason"])
    return patients.reset_index(drop=True), healthy.reset_index(drop=True), skipped


def build_validation_tables(
    *,
    patients: pd.DataFrame,
    healthy: pd.DataFrame,
    features_dir: Path,
    states: Sequence[str],
    modalities: Sequence[str],
    model_attribution_features: set[str],
    robust_reference: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load features, build healthy references, and compute z-distance tables."""

    distance_rows: list[dict[str, object]] = []
    feature_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []
    patients_by_id = patients.set_index("subject_id", drop=False)
    healthy_id_map = {
        subject_id: f"healthy_{index:03d}"
        for index, subject_id in enumerate(healthy["subject_id"].dropna().astype(str), start=1)
    }

    for modality in modalities:
        if modality not in {"psd", "wpli"}:
            skipped_rows.append({"subject_id": "", "analysis": modality, "reason": "unsupported modality"})
            continue
        for state in states:
            healthy_features = []
            for subject_id in healthy["subject_id"].dropna().astype(str):
                try:
                    feature = load_feature(subject_id, "healthy", "baseline", state, modality, features_dir)
                    healthy_features.append(
                        replace(feature, subject_id=healthy_id_map.get(subject_id, "healthy_unmapped"))
                    )
                except Exception as exc:  # noqa: BLE001 - recorded for skipped_subjects.csv
                    skipped_rows.append(
                        {
                            "subject_id": healthy_id_map.get(subject_id, "healthy_unmapped"),
                            "analysis": f"{modality}_{state}_healthy",
                            "reason": "missing or unreadable healthy feature",
                        }
                    )
            if not healthy_features:
                skipped_rows.append({"subject_id": "", "analysis": f"{modality}_{state}", "reason": "no healthy feature files"})
                continue
            try:
                reference = build_healthy_reference(
                    np.stack([feature.values for feature in healthy_features], axis=0),
                    robust=robust_reference,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append({"subject_id": "", "analysis": f"{modality}_{state}", "reason": str(exc)})
                continue

            for feature in healthy_features:
                z = transform_to_healthy_zscore(feature.values, reference)
                distance_rows.extend(_distance_rows(feature, z, model_attribution_features, patients_by_id))
                feature_frames.append(_aggregate_feature_frame(feature, z, patients_by_id))

            patient_features_by_timepoint: dict[tuple[str, str], tuple[LoadedFeature, np.ndarray]] = {}
            for subject_id in patients["subject_id"].dropna().astype(str):
                for timepoint in ("baseline", "post14"):
                    try:
                        feature = load_feature(subject_id, "patient", timepoint, state, modality, features_dir)
                    except Exception as exc:  # noqa: BLE001
                        skipped_rows.append(
                            {
                                "subject_id": subject_id,
                                "analysis": f"{modality}_{state}_{timepoint}",
                                "reason": str(exc),
                            }
                        )
                        continue
                    z = transform_to_healthy_zscore(feature.values, reference)
                    patient_features_by_timepoint[(subject_id, timepoint)] = (feature, z)
                    distance_rows.extend(_distance_rows(feature, z, model_attribution_features, patients_by_id))
                    feature_frames.append(_aggregate_feature_frame(feature, z, patients_by_id))

    distances = pd.DataFrame(distance_rows)
    feature_measurements = (
        pd.concat(feature_frames, ignore_index=True, sort=False)
        if feature_frames
        else pd.DataFrame()
    )
    normalization = _normalization_from_distances(distances)
    feature_normalization = _normalization_from_feature_measurements(feature_measurements)
    return {
        "distances": distances,
        "normalization": normalization,
        "feature_measurements": feature_measurements,
        "feature_normalization": feature_normalization,
        "skipped_subjects": pd.DataFrame(skipped_rows, columns=["subject_id", "analysis", "reason"]),
    }


def baseline_abnormality_stats(distances: pd.DataFrame, feature_measurements: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline patient health-distance or feature deviation against healthy controls."""

    rows: list[dict[str, object]] = []
    if not distances.empty:
        baseline = distances[distances["timepoint"].eq("baseline")]
        keys = ["modality", "state", "level", "band", "network_group"]
        for values, group in baseline.groupby(keys, dropna=False):
            patient = group[group["group"].eq("patient")]["distance_to_health"]
            healthy = group[group["group"].eq("healthy")]["distance_to_health"]
            result = two_sample_test(patient, healthy, method="mannwhitney")
            rows.append(
                {
                    **_key_dict(keys, values),
                    **result,
                    "feature_scope": "distance",
                    "feature_id": _distance_feature_id(_key_dict(keys, values)),
                    "comparison": "baseline_patient_vs_healthy",
                }
            )
    if not feature_measurements.empty:
        baseline_features = feature_measurements[feature_measurements["timepoint"].eq("baseline")]
        for feature_id, group in baseline_features.groupby("feature_id", dropna=False):
            patient = group[group["group"].eq("patient")]["abs_z"]
            healthy = group[group["group"].eq("healthy")]["abs_z"]
            result = two_sample_test(patient, healthy, method="mannwhitney")
            rows.append(
                {
                    **_feature_id_parts(str(feature_id)),
                    **result,
                    "feature_scope": "feature",
                    "feature_id": str(feature_id),
                    "comparison": "baseline_patient_vs_healthy",
                }
            )
    return add_fdr_column(pd.DataFrame(rows))


def baseline_outcome_association_stats(
    distances: pd.DataFrame,
    feature_measurements: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Associate baseline PSD/wPLI health deviation with later functional outcome at group level."""

    rows: list[dict[str, object]] = []
    covariates = _available_covariates(patients)
    if not distances.empty:
        baseline = distances[
            distances["group"].eq("patient") & distances["timepoint"].eq("baseline")
        ]
        keys = ["modality", "state", "level", "band", "network_group"]
        for values, group in baseline.groupby(keys, dropna=False):
            key_values = _key_dict(keys, values)
            for outcome in ("Delta_FMA", "proportional_recovery_residual"):
                rows.extend(
                    _association_rows(
                        group,
                        x_column="distance_to_health",
                        outcome=outcome,
                        covariates=covariates,
                        include_partial=True,
                        base={
                            **key_values,
                            "feature_scope": "distance",
                            "feature_id": _distance_feature_id(key_values),
                            "association_type": "baseline_outcome",
                        },
                    )
                )
            rows.append(
                {
                    **key_values,
                    **two_sample_test(
                        group[group["recovery_label"].eq(1)]["distance_to_health"],
                        group[group["recovery_label"].eq(0)]["distance_to_health"],
                        method="mannwhitney",
                    ),
                    "feature_scope": "distance",
                    "feature_id": _distance_feature_id(key_values),
                    "outcome": "recovery_group",
                    "association_type": "baseline_outcome_group_difference",
                }
            )
    if not feature_measurements.empty:
        baseline_features = feature_measurements[
            feature_measurements["group"].eq("patient") & feature_measurements["timepoint"].eq("baseline")
        ]
        for feature_id, group in baseline_features.groupby("feature_id", dropna=False):
            for outcome in ("Delta_FMA", "proportional_recovery_residual"):
                rows.extend(
                    _association_rows(
                        group,
                        x_column="abs_z",
                        outcome=outcome,
                        covariates=covariates,
                        include_partial=False,
                        base={
                            **_feature_id_parts(str(feature_id)),
                            "feature_scope": "feature",
                            "feature_id": str(feature_id),
                            "association_type": "baseline_outcome",
                        },
                    )
                )
    return add_fdr_column(pd.DataFrame(rows))


def pre_post_normalization_stats(
    normalization: pd.DataFrame,
    feature_normalization: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> pd.DataFrame:
    """Test whether post14 PSD/wPLI distances are closer to the healthy template."""

    rows: list[dict[str, object]] = []
    if not normalization.empty:
        keys = ["modality", "state", "level", "band", "network_group"]
        for values, group in normalization.groupby(keys, dropna=False):
            key_values = _key_dict(keys, values)
            rows.append(
                {
                    **key_values,
                    **paired_wilcoxon_test(group["D_pre"], group["D_post"], n_bootstrap=n_bootstrap),
                    "feature_scope": "distance",
                    "feature_id": _distance_feature_id(key_values),
                    "comparison": "pre_post_distance_to_health",
                }
            )
    if not feature_normalization.empty:
        for feature_id, group in feature_normalization.groupby("feature_id", dropna=False):
            rows.append(
                {
                    **_feature_id_parts(str(feature_id)),
                    **paired_wilcoxon_test(group["abs_z_pre"], group["abs_z_post"], n_bootstrap=0),
                    "feature_scope": "feature",
                    "feature_id": str(feature_id),
                    "comparison": "pre_post_feature_abs_z",
                }
            )
    return add_fdr_column(pd.DataFrame(rows))


def normalization_outcome_correlation_stats(
    normalization: pd.DataFrame,
    feature_normalization: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Associate normalization index with Delta_FMA and residual outcomes."""

    rows: list[dict[str, object]] = []
    covariates = _available_covariates(patients)
    if not normalization.empty:
        keys = ["modality", "state", "level", "band", "network_group"]
        for values, group in normalization.groupby(keys, dropna=False):
            key_values = _key_dict(keys, values)
            for outcome in ("Delta_FMA", "proportional_recovery_residual", "negative_residual"):
                rows.extend(
                    _association_rows(
                        group,
                        x_column="NI",
                        outcome=outcome,
                        covariates=covariates,
                        include_partial=True,
                        base={
                            **key_values,
                            "feature_scope": "distance",
                            "feature_id": _distance_feature_id(key_values),
                            "association_type": "normalization_outcome",
                        },
                    )
                )
    if not feature_normalization.empty:
        for feature_id, group in feature_normalization.groupby("feature_id", dropna=False):
            for outcome in ("Delta_FMA", "proportional_recovery_residual", "negative_residual"):
                rows.extend(
                    _association_rows(
                        group,
                        x_column="NI_feature",
                        outcome=outcome,
                        covariates=covariates,
                        include_partial=False,
                        base={
                            **_feature_id_parts(str(feature_id)),
                            "feature_scope": "feature",
                            "feature_id": str(feature_id),
                            "association_type": "normalization_outcome",
                        },
                    )
                )
    return add_fdr_column(pd.DataFrame(rows))


def normalization_group_difference_stats(
    normalization: pd.DataFrame,
    feature_normalization: pd.DataFrame,
    *,
    n_permutation: int,
) -> pd.DataFrame:
    """Compare NI between proportional-recovery and poor-recovery groups."""

    rows: list[dict[str, object]] = []
    if not normalization.empty:
        keys = ["modality", "state", "level", "band", "network_group"]
        for values, group in normalization.groupby(keys, dropna=False):
            key_values = _key_dict(keys, values)
            result = two_sample_test(
                group[group["recovery_label"].eq(1)]["NI"],
                group[group["recovery_label"].eq(0)]["NI"],
                method="mannwhitney",
            )
            rows.append(
                {
                    **key_values,
                    **result,
                    "permutation_p_value": permutation_group_p_value(
                        group["NI"],
                        group["recovery_label"],
                        n_permutation=n_permutation,
                    ),
                    "feature_scope": "distance",
                    "feature_id": _distance_feature_id(key_values),
                    "comparison": "normalization_group_difference",
                }
            )
    if not feature_normalization.empty:
        for feature_id, group in feature_normalization.groupby("feature_id", dropna=False):
            result = two_sample_test(
                group[group["recovery_label"].eq(1)]["NI_feature"],
                group[group["recovery_label"].eq(0)]["NI_feature"],
                method="mannwhitney",
            )
            rows.append(
                {
                    **_feature_id_parts(str(feature_id)),
                    **result,
                    "permutation_p_value": np.nan,
                    "feature_scope": "feature",
                    "feature_id": str(feature_id),
                    "comparison": "normalization_group_difference",
                }
            )
    return add_fdr_column(pd.DataFrame(rows))


def feature_overlap_stats(
    *,
    model_features: set[str],
    abnormality: pd.DataFrame,
    pre_post: pd.DataFrame,
    baseline_assoc: pd.DataFrame,
    norm_assoc: pd.DataFrame,
    feature_measurements: pd.DataFrame,
    n_permutation: int,
) -> pd.DataFrame:
    """Compute overlap between model high-attribution features and statistical evidence sets."""

    abnormal_features = significant_feature_set(abnormality[abnormality.get("feature_scope").eq("feature")] if not abnormality.empty else abnormality)
    normalization_features = significant_feature_set(
        pre_post[
            pre_post.get("feature_scope").eq("feature")
            & (pd.to_numeric(pre_post.get("mean_change"), errors="coerce") > 0)
        ] if not pre_post.empty and "mean_change" in pre_post.columns else pd.DataFrame()
    )
    outcome_frames = []
    if not baseline_assoc.empty:
        outcome_frames.append(baseline_assoc[baseline_assoc.get("feature_scope").eq("feature")])
    if not norm_assoc.empty:
        outcome_frames.append(norm_assoc[norm_assoc.get("feature_scope").eq("feature")])
    outcome = pd.concat(outcome_frames, ignore_index=True, sort=False) if outcome_frames else pd.DataFrame()
    outcome_features = significant_feature_set(outcome)
    universe = set(feature_measurements["feature_id"].dropna().astype(str)) if not feature_measurements.empty else set()
    universe.update(model_features)
    return compute_feature_overlap(
        model_attribution_features=model_features,
        evidence_sets={
            "abnormal_features": abnormal_features,
            "normalization_features": normalization_features,
            "outcome_associated_features": outcome_features,
        },
        universe_features=universe,
        n_permutation=n_permutation,
        random_state=0,
    )


def make_figures(
    distances: pd.DataFrame,
    normalization: pd.DataFrame,
    feature_normalization: pd.DataFrame,
    overlap: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    """Create all required biomarker validity figures."""

    paths = [
        plot_baseline_health_distance_group_boxplot(
            distances,
            figure_dir / "baseline_health_distance_group_boxplot.png",
        ),
        plot_health_distance_slopeplot(
            normalization,
            figure_dir / "health_distance_slopeplot.png",
        ),
        plot_normalization_scatter(
            normalization,
            figure_dir / "normalization_vs_delta_fma_scatter.png",
            outcome_column="Delta_FMA",
            y_label="Delta FMA-UE",
        ),
        plot_normalization_scatter(
            normalization,
            figure_dir / "normalization_vs_residual_scatter.png",
            outcome_column="negative_residual",
            y_label="-residual (higher is better)",
        ),
        plot_psd_topomap_normalization(
            feature_normalization,
            figure_dir / "psd_topomap_normalization.png",
        ),
        plot_wpli_connectome_normalization(
            feature_normalization,
            figure_dir / "wpli_connectome_normalization.png",
        ),
        plot_feature_evidence_overlap(
            overlap,
            figure_dir / "feature_evidence_overlap.png",
        ),
    ]
    summary = make_biomarker_validity_summary_figure(
        paths,
        figure_dir / "biomarker_validity_summary_figure.png",
    )
    return [*paths, summary]


def write_report(
    path: Path,
    *,
    patients: pd.DataFrame,
    healthy: pd.DataFrame,
    skipped: pd.DataFrame,
    distances: pd.DataFrame,
    normalization: pd.DataFrame,
    abnormality: pd.DataFrame,
    baseline_assoc: pd.DataFrame,
    pre_post: pd.DataFrame,
    norm_assoc: pd.DataFrame,
    group_diff: pd.DataFrame,
    overlap: pd.DataFrame,
    figure_paths: Sequence[Path],
) -> None:
    """Write the markdown report with leakage-control statements."""

    included_patients = distances.loc[distances.get("group").eq("patient"), "subject_id"].nunique() if not distances.empty else 0
    included_healthy = distances.loc[distances.get("group").eq("healthy"), "subject_id"].nunique() if not distances.empty else 0
    lines = [
        "# PSD/wPLI Biomarker Validity Report",
        "",
        "This module evaluates PSD/wPLI feature validity as group-level association, not individual prediction.",
        "",
        "## Data Inclusion",
        "",
        f"- Metadata patient rows: {len(patients)}",
        f"- Metadata healthy rows: {len(healthy)}",
        f"- Patients with at least one loaded EEG feature: {included_patients}",
        f"- Healthy subjects with at least one loaded EEG feature: {included_healthy}",
        f"- Skipped records: {len(skipped)}",
        "",
        "## Healthy Template",
        "",
        "Healthy PSD/wPLI tensors were used to estimate per-feature center and scale. Patient baseline and post14 tensors were transformed into healthy-reference z-scores, and RMS z-distance was used as distance-to-health.",
        "",
        "## Feature Computation",
        "",
        "PSD and wPLI features were not recomputed in this validation step. The module reads the existing feature files generated by the repository's feature pipeline under `data/features_all_stages`, then validates those saved tensors against the healthy reference template.",
        "",
        "PSD features were computed from EO/EC EEG after affected-hand hemisphere alignment. The saved PSD tensors use 62 canonical channels and Welch power spectral density estimation with a Hann window, density scaling, constant detrending, 50% overlap, and a segment length selected to give 0.5 Hz frequency resolution. The retained frequency bins span 0.5-45 Hz at 0.5 Hz resolution, giving a channel-frequency tensor of shape 62 x 90. For validation summaries, PSD bins were also aggregated into Delta, Theta, Alpha, Beta Low, Beta Medium, and Beta High bands.",
        "",
        "wPLI features were computed from the same EO/EC, hemisphere-aligned EEG using STFT spectra. The STFT uses a Hann window, a 2-second segment length, 50% overlap, and constant detrending. Connectivity is calculated for the deterministic upper-triangle channel pairs among the 62 canonical channels, giving 1891 edges. For each edge and band, wPLI is computed as `abs(mean(imaginary cross-spectrum)) / mean(abs(imaginary cross-spectrum))`. The six saved bands are Delta 1-3 Hz, Theta 4-7 Hz, Alpha 8-13 Hz, Beta Low 13-18 Hz, Beta Medium 18-21 Hz, and Beta High 21-30 Hz, producing a wPLI edge-band tensor of shape 1891 x 6.",
        "",
        "## Baseline Abnormality",
        "",
        _top_table(abnormality, "baseline_patient_vs_healthy"),
        "",
        "## Baseline Outcome Association",
        "",
        "Baseline PSD/wPLI health-distance and feature-level deviations were associated with later Delta_FMA and proportional recovery residual only at the cohort level.",
        _top_table(baseline_assoc, "baseline_outcome"),
        "",
        "## Pre/Post Normalization",
        "",
        "Paired baseline versus post14 analyses test whether PSD/wPLI moved toward the healthy reference template.",
        _top_table(pre_post, "pre_post"),
        "",
        "## Normalization Outcome Association",
        "",
        "Normalization index was associated with Delta_FMA and residual at the group level.",
        _top_table(norm_assoc, "normalization_outcome"),
        "",
        "## Outcome-Group Normalization Difference",
        "",
        _top_table(group_diff, "normalization_group_difference"),
        "",
        "## Model Explanation Consistency",
        "",
        "Model high-attribution PSD/wPLI features were compared with abnormal, normalization, and outcome-associated feature sets.",
        _top_table(overlap, "feature_evidence_overlap"),
        "",
        "Interpretation: model-attended PSD/wPLI features overlapping independently validated abnormal, normalizing, or recovery-associated features supports physiologic interpretability. It does not mean the statistical features themselves directly predict individual outcome.",
        "",
        "## Figures",
        "",
        *[f"- `{_relative_or_name(figure, path.parent)}`" for figure in figure_paths],
        "",
        "## Data Leakage Control",
        "",
        "- post14 PSD/wPLI are used only for longitudinal validation.",
        "- post14 PSD/wPLI are not marked as baseline prediction input.",
        "- post14 FMA is used only to compute Delta_FMA and proportional recovery residual.",
        "- This module does not train a baseline-only healthy-distance classifier.",
        "- This module does not output individual prediction scores or subject-level prediction probabilities.",
        "- Individualized prediction remains the role of the existing residual-aware SSL-CNN.",
        "",
        "## Conclusion",
        "",
        "These results support PSD and wPLI as physiologically meaningful inputs that reflect post-stroke brain-function abnormalities and recovery-related changes. The statistical associations are evidence for feature validity, but they are group-level association, not individual prediction; final patient-level inference remains with the residual-aware SSL-CNN.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _distance_rows(
    feature: LoadedFeature,
    z: np.ndarray,
    model_attribution_features: set[str],
    patients_by_id: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = _subject_context(feature, patients_by_id)
    rows.append(
        {
            **base,
            "level": "overall",
            "band": "",
            "network_group": "",
            "distance_to_health": compute_health_distance(z),
            "baseline_prediction_input": False,
            "analysis_role": _analysis_role(feature.timepoint),
        }
    )
    for row in feature_band_distances(feature, z):
        rows.append(
            {
                **base,
                **row,
                "baseline_prediction_input": False,
                "analysis_role": _analysis_role(feature.timepoint),
            }
        )
    subset_indices = []
    for feature_id, indices in aggregate_feature_groups(feature).items():
        if feature_id in model_attribution_features:
            subset_indices.extend(indices.tolist())
    if subset_indices:
        flat_z = z.reshape(-1)
        rows.append(
            {
                **base,
                "level": "model_attribution_top_feature_subset",
                "band": "",
                "network_group": "",
                "distance_to_health": compute_health_distance(flat_z[np.asarray(subset_indices, dtype=int)]),
                "baseline_prediction_input": False,
                "analysis_role": _analysis_role(feature.timepoint),
            }
        )
    return rows


def _aggregate_feature_frame(
    feature: LoadedFeature,
    z: np.ndarray,
    patients_by_id: pd.DataFrame,
) -> pd.DataFrame:
    base = _subject_context(feature, patients_by_id)
    if feature.modality == "psd":
        frame = _psd_feature_frame(feature, z)
    else:
        frame = _wpli_feature_frame(feature, z)
    for column, value in base.items():
        frame[column] = value
    frame["baseline_prediction_input"] = False
    frame["analysis_role"] = _analysis_role(feature.timepoint)
    return frame


def _psd_feature_frame(feature: LoadedFeature, z: np.ndarray) -> pd.DataFrame:
    values = np.asarray(feature.values, dtype=float)
    z_values = np.asarray(z, dtype=float)
    channels = feature.channel_names or tuple(f"ch{index:03d}" for index in range(values.shape[0]))
    frequencies = np.asarray(feature.frequency_bins or tuple(range(values.shape[1])), dtype=float)
    rows: list[dict[str, object]] = []
    band_defs = (
        ("Delta", 0.5, 4.0),
        ("Theta", 4.0, 8.0),
        ("Alpha", 8.0, 13.0),
        ("Beta Low", 13.0, 18.0),
        ("Beta Medium", 18.0, 21.0),
        ("Beta High", 21.0, 30.0),
    )
    for band, low_hz, high_hz in band_defs:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        if not np.any(mask):
            continue
        band_values = values[:, mask].mean(axis=1)
        band_z = z_values[:, mask].mean(axis=1)
        band_abs_z = np.abs(z_values[:, mask]).mean(axis=1)
        for channel, value, signed_z, abs_z in zip(channels, band_values, band_z, band_abs_z, strict=True):
            rows.append(
                {
                    "feature_id": f"psd|{feature.state}|{channel}|{band}",
                    "channel": channel,
                    "band": band,
                    "channel_i": "",
                    "channel_j": "",
                    "network_group": "",
                    "value": float(value),
                    "signed_z": float(signed_z),
                    "abs_z": float(abs_z),
                }
            )
    return pd.DataFrame(rows)


def _wpli_feature_frame(feature: LoadedFeature, z: np.ndarray) -> pd.DataFrame:
    values = np.asarray(feature.values, dtype=float)
    z_values = np.asarray(z, dtype=float)
    edges = feature.edge_list or tuple((f"edge{index:04d}", "") for index in range(values.shape[0]))
    bands = feature.band_names or tuple(f"band{index:02d}" for index in range(values.shape[1]))
    n_edges = len(edges)
    n_bands = len(bands)
    channel_i = np.repeat([edge[0] for edge in edges], n_bands)
    channel_j = np.repeat([edge[1] for edge in edges], n_bands)
    band_values = np.tile(np.asarray(bands, dtype=object), n_edges)
    feature_ids = [
        f"wpli|{feature.state}|{first}-{second}|{band}"
        for first, second in edges
        for band in bands
    ]
    return pd.DataFrame(
        {
            "feature_id": feature_ids,
            "channel": "",
            "band": band_values,
            "channel_i": channel_i,
            "channel_j": channel_j,
            "network_group": [
                _edge_network_group(first, second)
                for first, second in zip(channel_i, channel_j, strict=True)
            ],
            "value": values.reshape(-1),
            "signed_z": z_values.reshape(-1),
            "abs_z": np.abs(z_values).reshape(-1),
        }
    )


def _normalization_from_distances(distances: pd.DataFrame) -> pd.DataFrame:
    if distances.empty:
        return pd.DataFrame()
    patient = distances[distances["group"].eq("patient")].copy()
    keys = ["subject_id", "modality", "state", "level", "band", "network_group"]
    pre = patient[patient["timepoint"].eq("baseline")].rename(columns={"distance_to_health": "D_pre"})
    post = patient[patient["timepoint"].eq("post14")].rename(columns={"distance_to_health": "D_post"})
    merged = pre.merge(
        post[keys + ["D_post"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return merged
    merged["NI"] = compute_normalization_index(merged["D_pre"], merged["D_post"])
    merged["negative_residual"] = -pd.to_numeric(merged["proportional_recovery_residual"], errors="coerce")
    merged["baseline_prediction_input"] = False
    merged["analysis_role"] = "longitudinal_validation_only"
    return merged


def _normalization_from_feature_measurements(feature_measurements: pd.DataFrame) -> pd.DataFrame:
    if feature_measurements.empty:
        return pd.DataFrame()
    patient = feature_measurements[feature_measurements["group"].eq("patient")].copy()
    keys = ["subject_id", "modality", "state", "feature_id"]
    keep = [
        "abs_z",
        "signed_z",
        "value",
        "channel",
        "band",
        "channel_i",
        "channel_j",
        "network_group",
        "Delta_FMA",
        "proportional_recovery_residual",
        "recovery_label",
        "recovery_group",
    ]
    pre = patient[patient["timepoint"].eq("baseline")].rename(
        columns={"abs_z": "abs_z_pre", "signed_z": "signed_z_pre", "value": "value_pre"}
    )
    post = patient[patient["timepoint"].eq("post14")].rename(
        columns={"abs_z": "abs_z_post", "signed_z": "signed_z_post", "value": "value_post"}
    )
    post_columns = keys + ["abs_z_post", "signed_z_post", "value_post"]
    merged = pre.merge(post[post_columns], on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        return merged
    merged["normalization_change"] = merged["abs_z_pre"] - merged["abs_z_post"]
    merged["NI_feature"] = compute_normalization_index(merged["abs_z_pre"], merged["abs_z_post"])
    merged["negative_residual"] = -pd.to_numeric(merged["proportional_recovery_residual"], errors="coerce")
    merged["baseline_prediction_input"] = False
    merged["analysis_role"] = "longitudinal_validation_only"
    return merged[[column for column in [*keys, *keep, "abs_z_pre", "abs_z_post", "signed_z_pre", "signed_z_post", "value_pre", "value_post", "normalization_change", "NI_feature", "negative_residual", "baseline_prediction_input", "analysis_role"] if column in merged.columns]]


def _association_rows(
    group: pd.DataFrame,
    *,
    x_column: str,
    outcome: str,
    covariates: list[str],
    include_partial: bool,
    base: dict[str, object],
) -> list[dict[str, object]]:
    if outcome not in group.columns:
        return []
    rows: list[dict[str, object]] = []
    spearman = spearman_association(group[x_column], group[outcome])
    rows.append(
        {
            **base,
            "outcome": outcome,
            "association_method": "spearman",
            "rho": spearman["rho"],
            "p_value": spearman["p_value"],
            "n": spearman["n"],
        }
    )
    available = [column for column in covariates if column in group.columns]
    if not include_partial:
        return rows
    if available:
        partial = partial_spearman_association(
            group[x_column],
            group[outcome],
            group[available],
        )
        rows.append(
            {
                **base,
                "outcome": outcome,
                "association_method": "partial_spearman",
                "rho": partial["partial_rho"],
                "p_value": partial["p_value"],
                "n": partial["n"],
                "covariates": partial["covariates"],
            }
        )
    return rows


def _subject_context(feature: LoadedFeature, patients_by_id: pd.DataFrame) -> dict[str, object]:
    context: dict[str, object] = {
        "subject_id": feature.subject_id,
        "group": feature.group,
        "timepoint": feature.timepoint,
        "modality": feature.modality,
        "state": feature.state,
        "source_path": "" if feature.group == "healthy" else feature.source_path.as_posix(),
    }
    if feature.group == "patient" and feature.subject_id in patients_by_id.index:
        row = patients_by_id.loc[feature.subject_id]
        for column in [
            "age",
            "disease_duration",
            "FMA_pre",
            "FMA_post",
            "Delta_FMA",
            "proportional_recovery_residual",
            "recovery_label",
            "recovery_group",
        ]:
            context[column] = row.get(column, np.nan)
        context["negative_residual"] = -float(row["proportional_recovery_residual"]) if pd.notna(row.get("proportional_recovery_residual")) else np.nan
    else:
        context.update(
            {
                "age": np.nan,
                "disease_duration": np.nan,
                "FMA_pre": np.nan,
                "FMA_post": np.nan,
                "Delta_FMA": np.nan,
                "proportional_recovery_residual": np.nan,
                "negative_residual": np.nan,
                "recovery_label": np.nan,
                "recovery_group": "",
            }
        )
    return context


def _feature_id_parts(feature_id: str) -> dict[str, object]:
    parts = feature_id.split("|")
    if len(parts) < 4:
        return {"modality": "", "state": "", "channel": "", "band": "", "channel_i": "", "channel_j": "", "network_group": ""}
    modality, state, item, band = parts[:4]
    if modality == "wpli":
        channel_i, channel_j = item.split("-", 1) if "-" in item else (item, "")
        return {
            "modality": modality,
            "state": state,
            "channel": "",
            "band": band,
            "channel_i": channel_i,
            "channel_j": channel_j,
            "network_group": _edge_network_group(channel_i, channel_j),
        }
    return {
        "modality": modality,
        "state": state,
        "channel": item,
        "band": band,
        "channel_i": "",
        "channel_j": "",
        "network_group": "",
    }


def _available_covariates(patients: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("FMA_pre", "age", "disease_duration")
        if column in patients.columns and patients[column].notna().sum() >= 3
    ]


def _distance_feature_id(values: dict[str, object]) -> str:
    tokens = [
        "distance",
        str(values.get("modality", "")),
        str(values.get("state", "")),
        str(values.get("level", "")),
        str(values.get("band", "")),
        str(values.get("network_group", "")),
    ]
    return "|".join(token for token in tokens if token)


def _key_dict(keys: Sequence[str], values: object) -> dict[str, object]:
    if not isinstance(values, tuple):
        values = (values,)
    return {key: value for key, value in zip(keys, values, strict=True)}


def _read_metadata_excel(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_index = 0
    for index, row in raw.iterrows():
        normalized = {_normalize_header(value) for value in row.dropna()}
        if "编号" in normalized or "受试者编号" in normalized or "患者ID" in normalized:
            header_index = int(index)
            break
    return pd.read_excel(path, header=header_index)


def _canonical_metadata_column(column: object) -> object:
    normalized = _normalize_header(column)
    normalized_map = {_normalize_header(source): target for source, target in META_COLUMN_MAP.items()}
    return normalized_map.get(normalized, column)


def _normalize_header(value: object) -> str:
    return "".join(str(value).split())


def _extract_numeric_token(value: object) -> object:
    if pd.isna(value) or isinstance(value, (int, float)):
        return value
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return match.group(0) if match else value


def _sanitize_healthy_subject_id(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def _edge_network_group(first: str, second: str) -> str:
    first_group = _channel_group(first)
    second_group = _channel_group(second)
    if first_group == second_group:
        return first_group
    return "|".join(sorted((first_group, second_group)))


def _channel_group(channel: str) -> str:
    token = "".join(ch for ch in str(channel).upper() if ch.isalpha())
    if token.startswith(("FP", "AF", "F", "FT", "FC")):
        return "frontal"
    if token.startswith("C"):
        return "central"
    if token.startswith(("CP", "P", "PO")):
        return "parietal"
    if token.startswith(("O", "CB")):
        return "occipital"
    if token.startswith(("T", "TP")):
        return "temporal"
    return "other"


def _analysis_role(timepoint: str) -> str:
    return "baseline_group_validation_only" if timepoint == "baseline" else "longitudinal_validation_only"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _top_table(frame: pd.DataFrame, label: str) -> str:
    if frame.empty:
        return f"No rows available for {label}."
    display = frame.copy()
    if "q_value" in display.columns:
        display = display.sort_values("q_value", na_position="last")
    columns = [
        column
        for column in [
            "feature_scope",
            "feature_id",
            "modality",
            "state",
            "level",
            "band",
            "network_group",
            "outcome",
            "rho",
            "effect_size",
            "p_value",
            "q_value",
            "overlap_count",
        ]
        if column in display.columns
    ]
    table = display.loc[:, columns].head(10).copy()
    for column in table.select_dtypes(include=["object"]).columns:
        table[column] = table[column].fillna("").astype(str).str.replace("|", r"\|", regex=False)
    return table.to_markdown(index=False)


def _relative_or_name(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
