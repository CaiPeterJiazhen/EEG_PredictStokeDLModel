from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.io.eeglab import read_eeglab_set_metadata
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table, read_clinical_metadata
from eeg_recovery.metadata.subjects import normalize_subject_id


ERROR_SUBJECTS = ("sub09", "sub14")
OUTLIER_Z = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full-cohort EEG QC and focused sub09/sub14 audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--hash-files", action="store_true", help="Compute full SHA256 hashes for source .set/.fdt files.")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = path_config.output_root
    labels = load_supervised_label_table(path_config)
    clinical = read_clinical_metadata(path_config.patient_info_clinical_xlsx)
    supervised_ids = labels["subject_id"].map(normalize_subject_id).tolist()
    eeg_records = build_eeg_file_index(
        path_config.patient_eeg_root,
        path_config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )

    summary = _build_subject_summary(
        output_root=output_root,
        labels=labels,
        clinical=clinical,
        eeg_records=eeg_records,
        hash_files=args.hash_files,
    )
    outliers = _feature_outlier_table(summary)
    summary_path = output_root / "results" / "metrics" / "eeg_qc_subject_summary.csv"
    outlier_path = output_root / "results" / "metrics" / "eeg_qc_feature_outliers.csv"
    doc_path = output_root / "docs" / "error_subject_qc_sub09_sub14.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    outliers.to_csv(outlier_path, index=False)
    doc_path.write_text(_render_error_subject_doc(summary, outliers), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {outlier_path}")
    print(f"Wrote {doc_path}")


def _build_subject_summary(
    *,
    output_root: Path,
    labels: pd.DataFrame,
    clinical: pd.DataFrame,
    eeg_records: Iterable[Any],
    hash_files: bool,
) -> pd.DataFrame:
    clinical_by_subject = {
        normalize_subject_id(row.subject_id): row
        for row in clinical.itertuples(index=False)
    }
    baseline_records = {
        (record.subject_id, record.state): record
        for record in eeg_records
        if record.group == "patient" and record.stage == "基线"
    }
    feature_rows = []
    loaded_features: dict[str, dict[str, np.ndarray]] = {}
    for row in labels.itertuples(index=False):
        subject_id = normalize_subject_id(row.subject_id)
        psd_eo = _load_npz_array(output_root / "data" / "features" / "psd" / f"{subject_id}_EO_psd.npz", "psd")
        psd_ec = _load_npz_array(output_root / "data" / "features" / "psd" / f"{subject_id}_EC_psd.npz", "psd")
        wpli_eo = _load_npz_array(output_root / "data" / "features" / "fc" / f"{subject_id}_EO_fc.npz", "wpli")
        wpli_ec = _load_npz_array(output_root / "data" / "features" / "fc" / f"{subject_id}_EC_fc.npz", "wpli")
        loaded_features[subject_id] = {
            "psd_eo": psd_eo,
            "psd_ec": psd_ec,
            "wpli_eo": wpli_eo,
            "wpli_ec": wpli_ec,
        }
        clinical_row = clinical_by_subject[subject_id]
        eo_record = baseline_records[(subject_id, "EO")]
        ec_record = baseline_records[(subject_id, "EC")]
        eo_meta = read_eeglab_set_metadata(eo_record.set_path)
        ec_meta = read_eeglab_set_metadata(ec_record.set_path)
        source_meta = _source_metadata(eo_record, ec_record, hash_files=hash_files)
        residual = float(row.Residual)
        feature_rows.append(
            {
                "subject_id": subject_id,
                "FMA_pre": float(row.FMA_pre),
                "FMA_post": float(row.FMA_post),
                "observed_delta": float(row.Delta_FMA_obs),
                "predicted_delta": float(row.Delta_FMA_pred),
                "residual": residual,
                "label": int(row.label),
                "distance_to_threshold": abs(residual - 1.5),
                "baseline_eo_path": str(eo_record.set_path),
                "baseline_ec_path": str(ec_record.set_path),
                "affected_hand": str(clinical_row.affected_hand),
                "hemisphere_flip_applied": bool(_hemisphere_flip_applied(output_root, subject_id)),
                "eo_file_duration_seconds": float(eo_meta.pnts * eo_meta.trials / eo_meta.srate),
                "ec_file_duration_seconds": float(ec_meta.pnts * ec_meta.trials / ec_meta.srate),
                **source_meta,
                **_array_stats("psd", [psd_eo, psd_ec]),
                **_array_stats("wpli", [wpli_eo, wpli_ec]),
                "nan_count": int(
                    sum(np.isnan(array).sum() for array in (psd_eo, psd_ec, wpli_eo, wpli_ec))
                ),
                "zero_count": int(
                    sum(np.count_nonzero(array == 0.0) for array in (psd_eo, psd_ec, wpli_eo, wpli_ec))
                ),
                "psd_eo_ec_l2_distance": float(np.linalg.norm(psd_eo - psd_ec)),
                "wpli_eo_ec_l2_distance": float(np.linalg.norm(wpli_eo - wpli_ec)),
                "psd_eo_ec_mean_abs_distance": float(np.mean(np.abs(psd_eo - psd_ec))),
                "wpli_eo_ec_mean_abs_distance": float(np.mean(np.abs(wpli_eo - wpli_ec))),
            }
        )

    summary = pd.DataFrame(feature_rows).sort_values("subject_id").reset_index(drop=True)
    summary = _add_feature_z_qc(summary, loaded_features)
    return summary


def _add_feature_z_qc(summary: pd.DataFrame, features: Mapping[str, Mapping[str, np.ndarray]]) -> pd.DataFrame:
    subjects = summary["subject_id"].tolist()
    psd_vectors = np.stack([
        np.concatenate([features[subject]["psd_eo"], features[subject]["psd_ec"]], axis=0).reshape(-1)
        for subject in subjects
    ])
    wpli_vectors = np.stack([
        np.concatenate([features[subject]["wpli_eo"], features[subject]["wpli_ec"]], axis=0).reshape(-1)
        for subject in subjects
    ])
    psd_channel = np.stack([
        np.stack([features[subject]["psd_eo"], features[subject]["psd_ec"]], axis=0).mean(axis=(0, 2))
        for subject in subjects
    ])
    wpli_edge_band = np.stack([
        np.stack([features[subject]["wpli_eo"], features[subject]["wpli_ec"]], axis=0).mean(axis=0)
        for subject in subjects
    ])
    summary = summary.copy()
    summary["extreme_z_score_count"] = _extreme_count(np.concatenate([psd_vectors, wpli_vectors], axis=1), threshold=4.0)
    summary["channel_level_psd_outlier_count"] = _extreme_count(psd_channel, threshold=OUTLIER_Z)
    summary["channel_level_psd_max_abs_z"] = _max_abs_z(psd_channel)
    summary["edge_band_wpli_outlier_count"] = _extreme_count(wpli_edge_band.reshape(len(subjects), -1), threshold=OUTLIER_Z)
    summary["edge_band_wpli_max_abs_z"] = _max_abs_z(wpli_edge_band.reshape(len(subjects), -1))
    return summary


def _feature_outlier_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "FMA_pre",
        "FMA_post",
        "observed_delta",
        "predicted_delta",
        "residual",
        "distance_to_threshold",
        "eo_file_duration_seconds",
        "ec_file_duration_seconds",
        "psd_mean",
        "psd_std",
        "psd_min",
        "psd_max",
        "wpli_mean",
        "wpli_std",
        "wpli_min",
        "wpli_max",
        "nan_count",
        "zero_count",
        "extreme_z_score_count",
        "psd_eo_ec_l2_distance",
        "wpli_eo_ec_l2_distance",
        "psd_eo_ec_mean_abs_distance",
        "wpli_eo_ec_mean_abs_distance",
        "channel_level_psd_outlier_count",
        "channel_level_psd_max_abs_z",
        "edge_band_wpli_outlier_count",
        "edge_band_wpli_max_abs_z",
    ]
    rows = []
    for metric in metrics:
        values = pd.to_numeric(summary[metric], errors="coerce").astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        z_scores = (values - mean) / (std if std > 1e-12 else 1.0)
        ranks = values.rank(method="min", ascending=False)
        for subject_id, value, z_score, rank in zip(summary["subject_id"], values, z_scores, ranks):
            rows.append(
                {
                    "metric": metric,
                    "subject_id": subject_id,
                    "value": float(value),
                    "z_score": float(z_score),
                    "rank_desc": int(rank),
                    "is_outlier": bool(abs(float(z_score)) >= OUTLIER_Z),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "rank_desc", "subject_id"]).reset_index(drop=True)


def _render_error_subject_doc(summary: pd.DataFrame, outliers: pd.DataFrame) -> str:
    lines = [
        "# sub09/sub14 Error Subject QC Audit",
        "",
        "This audit compares sub09 and sub14 against the full 19-patient supervised cohort. A preprocessing recommendation is made only when an objective QC metric is an outlier (absolute z-score >= 3).",
        "",
        "## Label Borderline Check",
        "",
        "| subject_id | FMA_pre | FMA_post | observed_delta | predicted_delta | residual | label | distance_to_threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subject_id in ERROR_SUBJECTS:
        row = summary.loc[summary["subject_id"] == subject_id].iloc[0]
        lines.append(
            "| {subject_id} | {FMA_pre:.3f} | {FMA_post:.3f} | {observed_delta:.3f} | "
            "{predicted_delta:.3f} | {residual:.3f} | {label:d} | {distance_to_threshold:.3f} |".format(
                subject_id=subject_id,
                FMA_pre=row.FMA_pre,
                FMA_post=row.FMA_post,
                observed_delta=row.observed_delta,
                predicted_delta=row.predicted_delta,
                residual=row.residual,
                label=int(row.label),
                distance_to_threshold=row.distance_to_threshold,
            )
        )
    lines.extend(["", "## QC Outlier Summary", ""])
    for subject_id in ERROR_SUBJECTS:
        subject_outliers = outliers.loc[
            (outliers["subject_id"] == subject_id)
            & (outliers["is_outlier"])
            & (~outliers["metric"].isin(["residual", "distance_to_threshold", "FMA_pre", "FMA_post"]))
        ].copy()
        if subject_outliers.empty:
            lines.append(f"- {subject_id}: no objective EEG/metadata QC metric exceeded |z| >= {OUTLIER_Z}.")
        else:
            metrics = ", ".join(
                f"{row.metric} (z={row.z_score:.2f}, rank={int(row.rank_desc)})"
                for row in subject_outliers.itertuples(index=False)
            )
            lines.append(f"- {subject_id}: objective QC outliers detected: {metrics}.")
    lines.extend(
        [
            "",
            "## Metric Ranks For sub09/sub14",
            "",
            "| metric | sub09 value/rank/z | sub14 value/rank/z |",
            "|---|---:|---:|",
        ]
    )
    rank_metrics = [
        "residual",
        "distance_to_threshold",
        "psd_mean",
        "psd_std",
        "wpli_mean",
        "wpli_std",
        "zero_count",
        "extreme_z_score_count",
        "psd_eo_ec_mean_abs_distance",
        "wpli_eo_ec_mean_abs_distance",
        "channel_level_psd_outlier_count",
        "edge_band_wpli_outlier_count",
        "channel_level_psd_max_abs_z",
        "edge_band_wpli_max_abs_z",
    ]
    for metric in rank_metrics:
        cells = []
        for subject_id in ERROR_SUBJECTS:
            row = outliers.loc[(outliers["metric"] == metric) & (outliers["subject_id"] == subject_id)].iloc[0]
            cells.append(f"{row.value:.4g} / {int(row.rank_desc)} / {row.z_score:.2f}")
        lines.append(f"| {metric} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "## Interpretation", ""])
    for subject_id in ERROR_SUBJECTS:
        qc_outlier = outliers.loc[
            (outliers["subject_id"] == subject_id)
            & (outliers["is_outlier"])
            & (~outliers["metric"].isin(["residual", "distance_to_threshold", "FMA_pre", "FMA_post"]))
        ]
        row = summary.loc[summary["subject_id"] == subject_id].iloc[0]
        threshold_note = (
            "near the residual threshold"
            if float(row.distance_to_threshold) <= 1.0
            else "not especially close to the residual threshold"
        )
        if qc_outlier.empty:
            lines.append(
                f"- {subject_id}: classify as a hard/borderline case rather than a preprocessing failure; it is {threshold_note} and has no objective QC outlier."
            )
        else:
            lines.append(
                f"- {subject_id}: review preprocessing because objective QC outliers are present; it is {threshold_note}."
            )
    lines.append("")
    return "\n".join(lines)


def _array_stats(prefix: str, arrays: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate([array.reshape(-1) for array in arrays]).astype(float)
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
    }


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise KeyError(f"{path} is missing key {key!r}.")
        return np.asarray(payload[key], dtype=np.float32)


def _hemisphere_flip_applied(output_root: Path, subject_id: str) -> bool:
    path = output_root / "data" / "features" / "psd" / f"{subject_id}_EO_psd.npz"
    with np.load(path, allow_pickle=False) as payload:
        if "hemisphere_aligned" not in payload:
            return False
        return bool(payload["hemisphere_aligned"].item())


def _source_metadata(eo_record: Any, ec_record: Any, *, hash_files: bool) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for state, record in (("eo", eo_record), ("ec", ec_record)):
        for suffix, path in (("set", record.set_path), ("fdt", record.fdt_path)):
            stat = Path(path).stat()
            values[f"{state}_{suffix}_mtime_ns"] = int(stat.st_mtime_ns)
            values[f"{state}_{suffix}_size"] = int(stat.st_size)
            values[f"{state}_{suffix}_sha256"] = _sha256(path) if hash_files else ""
    return values


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extreme_count(matrix: np.ndarray, *, threshold: float) -> np.ndarray:
    z = _z_matrix(matrix)
    return (np.abs(z) >= threshold).sum(axis=1).astype(int)


def _max_abs_z(matrix: np.ndarray) -> np.ndarray:
    return np.nanmax(np.abs(_z_matrix(matrix)), axis=1)


def _z_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (values - mean) / std


if __name__ == "__main__":
    main()
