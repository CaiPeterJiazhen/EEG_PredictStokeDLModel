from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.feature_tables import load_fc_feature_table, load_psd_band_power_table, merge_feature_tables
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id


ERROR_SUBJECTS = ("sub05", "sub14", "sub09", "sub28")
FINAL_SEED_PATTERN = "dl_loso_predictions_patient_barlow_residualaware_highrank_swa_clsalpha1_seed*.csv"
FINAL_SEEDMEAN_FILE = "seedmean_patient_barlow_residualaware_highrank_swa_clsalpha1_10seed.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze repeated error and near-threshold subjects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--subjects", nargs="+", default=list(ERROR_SUBJECTS))
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    subjects = [normalize_subject_id(subject) for subject in args.subjects]
    predictions = load_final_prediction_distribution(output_root)
    feature_table = build_feature_zscore_table(output_root, labels["subject_id"].tolist())
    attribution = load_attribution_summary(output_root)
    summary = build_error_subject_summary(
        labels,
        predictions,
        feature_table,
        attribution,
        subjects=subjects,
        threshold=args.threshold,
    )

    tables_dir = output_root / "results" / "tables"
    figures_dir = output_root / "results" / "figures" / "paper"
    docs_dir = output_root / "docs"
    for directory in (tables_dir, figures_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "error_subject_clinical_eeg_summary.csv", index=False)
    plot_probability_distribution(predictions, subjects, figures_dir / "error_subject_probability_distribution.png")
    plot_feature_outlier_heatmap(feature_table, subjects, figures_dir / "error_subject_feature_outlier_heatmap.png")
    write_error_doc(docs_dir / "error_subject_analysis.md", summary)
    print(f"Wrote {tables_dir / 'error_subject_clinical_eeg_summary.csv'}")


def load_final_prediction_distribution(output_root: Path) -> pd.DataFrame:
    prediction_dir = output_root / "results" / "predictions"
    frames = []
    for path in sorted(prediction_dir.glob(FINAL_SEED_PATTERN)):
        frame = pd.read_csv(path)
        seed = _extract_seed(path.name)
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        path = prediction_dir / FINAL_SEEDMEAN_FILE
        if not path.exists():
            raise FileNotFoundError(f"No final model predictions found under {prediction_dir}")
        frame = pd.read_csv(path)
        frame["seed"] = "seedmean"
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["subject_id"] = predictions["subject_id"].map(normalize_subject_id)
    if "y_pred" not in predictions.columns:
        predictions["y_pred"] = (predictions["y_score"].astype(float) >= 0.5).astype(int)
    return predictions


def build_feature_zscore_table(output_root: Path, subject_ids: list[str]) -> pd.DataFrame:
    psd = load_psd_band_power_table(output_root / "data" / "features" / "psd", subject_ids=subject_ids)
    wpli = load_fc_feature_table(output_root / "data" / "features" / "fc", metric="wpli", subject_ids=subject_ids)
    features = merge_feature_tables(psd, wpli).set_index("subject_id")
    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (features - means) / stds
    return z.fillna(0.0).reset_index()


def load_attribution_summary(output_root: Path) -> pd.DataFrame:
    explain_root = output_root / "results" / "explainability"
    path = explain_root / "error_subject_explainability_summary.csv"
    if path.exists():
        frame = pd.read_csv(path)
        frame["subject_id"] = frame["subject_id"].map(normalize_subject_id)
        return frame
    path = explain_root / "explained_predictions.csv"
    if path.exists():
        frame = pd.read_csv(path)
        frame["subject_id"] = frame["subject_id"].map(normalize_subject_id)
        return (
            frame.groupby("subject_id", as_index=False)
            .agg(mean_explained_score=("y_score", "mean"), explained_error_rate=("correct", lambda value: float(1.0 - np.mean(value))))
        )
    return pd.DataFrame(columns=["subject_id"])


def build_error_subject_summary(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_z: pd.DataFrame,
    attribution: pd.DataFrame,
    *,
    subjects: list[str],
    threshold: float,
) -> pd.DataFrame:
    labels_index = labels.set_index("subject_id", drop=False)
    feature_index = feature_z.set_index("subject_id", drop=False)
    attribution_index = attribution.set_index("subject_id", drop=False) if "subject_id" in attribution.columns else pd.DataFrame()
    rows = []
    for subject_id in subjects:
        if subject_id not in labels_index.index:
            continue
        label = labels_index.loc[subject_id]
        pred = predictions[predictions["subject_id"] == subject_id].copy()
        if pred.empty:
            mean_score = np.nan
            std_score = np.nan
            error_rate = np.nan
            y_true = int(label["label"])
        else:
            y_true = int(pred["y_true"].iloc[0]) if "y_true" in pred.columns else int(label["label"])
            correct = pred["y_pred"].astype(int).to_numpy() == y_true
            mean_score = float(pred["y_score"].astype(float).mean())
            std_score = float(pred["y_score"].astype(float).std(ddof=0))
            error_rate = float(1.0 - np.mean(correct))
        z_summary = summarize_feature_outliers(feature_index.loc[subject_id]) if subject_id in feature_index.index else {}
        attribution_summary = {}
        if not attribution_index.empty and subject_id in attribution_index.index:
            attr = attribution_index.loc[subject_id]
            if isinstance(attr, pd.DataFrame):
                attr = attr.iloc[0]
            attribution_summary = {f"attribution_{column}": attr[column] for column in attr.index if column != "subject_id"}
        residual = float(label["Residual"])
        signed_distance = float(threshold - residual)
        distance_to_threshold = abs(signed_distance)
        rows.append(
            {
                "subject_id": subject_id,
                "y_true": y_true,
                "mean_score": mean_score,
                "std_score": std_score,
                "error_rate": error_rate,
                "FMA_pre": float(label["FMA_pre"]),
                "FMA_post": float(label["FMA_post"]),
                "Residual": residual,
                "signed_distance": signed_distance,
                "distance_to_threshold": distance_to_threshold,
                "age": float(label["age"]),
                "sex": label["sex"],
                "duration": float(label["duration"]),
                "affected_hand": label["affected_hand"],
                "MBI_pre": float(label["MBI_pre"]),
                "near_threshold_margin_0.5": bool(distance_to_threshold <= 0.5),
                "near_threshold_margin_1.0": bool(distance_to_threshold <= 1.0),
                "possible_clinical_heterogeneity": bool(distance_to_threshold <= 1.0 or error_rate >= 0.5 or z_summary.get("n_abs_z_ge_3", 0) > 0),
                **z_summary,
                **attribution_summary,
            }
        )
    return pd.DataFrame(rows)


def summarize_feature_outliers(row: pd.Series) -> dict[str, object]:
    values = row.drop(labels=["subject_id"], errors="ignore").astype(float)
    abs_values = values.abs()
    top = abs_values.sort_values(ascending=False).head(10)
    return {
        "max_abs_feature_z": float(top.iloc[0]) if len(top) else np.nan,
        "n_abs_z_ge_2": int((abs_values >= 2.0).sum()),
        "n_abs_z_ge_3": int((abs_values >= 3.0).sum()),
        "top_outlier_features": ";".join(top.index.tolist()),
        "top_outlier_z": ";".join(f"{values[index]:.2f}" for index in top.index),
    }


def plot_probability_distribution(predictions: pd.DataFrame, subjects: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [predictions.loc[predictions["subject_id"] == subject, "y_score"].astype(float).to_numpy() for subject in subjects]
    ax.boxplot(data, labels=subjects, showmeans=True)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability")
    ax.set_title("Error subject probability distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_feature_outlier_heatmap(feature_z: pd.DataFrame, subjects: list[str], path: Path) -> None:
    frame = feature_z[feature_z["subject_id"].isin(subjects)].set_index("subject_id")
    if frame.empty:
        matrix = np.zeros((len(subjects), 1))
        labels = ["not_available"]
    else:
        top_features = frame.abs().max(axis=0).sort_values(ascending=False).head(30).index.tolist()
        matrix = frame.loc[[subject for subject in subjects if subject in frame.index], top_features].to_numpy(float)
        labels = top_features
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([subject for subject in subjects if subject in frame.index] or subjects)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_title("Top EEG feature z-score outliers")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_error_doc(path: Path, summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Error Subject Analysis",
        "",
        "This is a post-hoc audit of repeatedly misclassified or watched subjects. It combines final-model probability distributions, baseline clinical variables, residual distance to the locked threshold, EEG feature z-score outliers, and available attribution summaries.",
        "",
        _to_markdown(summary),
        "",
        "Subjects close to the residual threshold or with high EEG outlier burden should be discussed as potential clinical heterogeneity rather than as simple model failures.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_seed(filename: str) -> int | str:
    token = filename.split("_seed")[-1].split(".csv")[0]
    try:
        return int(token)
    except ValueError:
        return token


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
