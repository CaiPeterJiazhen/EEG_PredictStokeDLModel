from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.patient_level_comparison import (
    paired_correctness_significance_ceiling,
    paired_patient_prediction_comparison,
)
from eeg_recovery.training.train_feature_ssl import _safe_run_name
from eeg_recovery.training.train_supervised import (
    aggregate_patient_probabilities,
    load_supervised_feature_records,
)


SEEDS = (0, 1, 2, 3, 4, 5, 7, 13, 21, 42)
QEEG_FEATURE = "qeeg_ec_global_slow_fast_bsi"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize qEEG-guided graph-smoothed masked-VICReg SSL-CNN outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    metrics_dir = output_root / "results" / "metrics"
    predictions_dir = output_root / "results" / "predictions"
    docs_dir = output_root / "docs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    qeeg_guided_predictions, qeeg_guided_metrics = _collect_qeeg_guided_outputs(metrics_dir, predictions_dir)
    qeeg_guided_seedmean = _seedmean_predictions(qeeg_guided_predictions)
    qeeg_guided_seedmean_path = predictions_dir / "seedmean_qeeg_guided_graphsmooth_mvicreg.csv"
    qeeg_guided_seedmean.to_csv(qeeg_guided_seedmean_path, index=False)

    labels = load_supervised_label_table(path_config)
    qeeg_only_predictions, qeeg_only_metrics = _run_qeeg_only_baseline(path_config, labels)
    qeeg_only_path = predictions_dir / "dl_loso_predictions_qeeg_only_ec_slowfastbsi.csv"
    qeeg_only_predictions.to_csv(qeeg_only_path, index=False)

    no_ssl_qeeg_predictions, no_ssl_qeeg_metrics = _collect_seeded_outputs(
        metrics_dir,
        predictions_dir,
        metric_pattern="dl_model_comparison_qeeg_guided_no_ssl_psdfcwpli_seed{seed}.csv",
        prediction_pattern="dl_loso_predictions_qeeg_guided_no_ssl_psdfcwpli_seed{seed}.csv",
        model_group="no_ssl_cnn_qeeg_branch",
    )
    no_ssl_qeeg_seedmean = _seedmean_predictions(no_ssl_qeeg_predictions)

    model_selection = _model_selection_summary(
        qeeg_guided_metrics,
        qeeg_guided_seedmean,
        no_ssl_qeeg_metrics,
        no_ssl_qeeg_seedmean,
        qeeg_only_metrics,
        qeeg_only_predictions,
    )
    model_selection.to_csv(metrics_dir / "qeeg_guided_ssl_cnn_10seed_model_selection_summary.csv", index=False)

    stability = _seed_stability_summary(qeeg_guided_metrics, no_ssl_qeeg_metrics)
    stability.to_csv(metrics_dir / "qeeg_guided_ssl_cnn_seed_stability_summary.csv", index=False)

    subject_errors = _subject_error_frequency(qeeg_guided_predictions)
    subject_errors.to_csv(metrics_dir / "qeeg_guided_ssl_cnn_subject_error_frequency.csv", index=False)

    watched = _watched_subject_scores(qeeg_guided_predictions, no_ssl_qeeg_predictions)
    watched.to_csv(metrics_dir / "sub09_sub14_qeeg_guided_scores_summary.csv", index=False)

    comparison = _statistical_comparison(
        predictions_dir,
        qeeg_guided_seedmean,
        no_ssl_qeeg_seedmean,
        qeeg_only_predictions,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )
    comparison.to_csv(metrics_dir / "qeeg_guided_ssl_cnn_statistical_comparison.csv", index=False)

    _write_results_doc(
        docs_dir / "qeeg_guided_ssl_cnn_results.md",
        model_selection,
        stability,
        subject_errors,
        watched,
        comparison,
        qeeg_guided_seedmean_path,
    )
    print(f"Wrote {metrics_dir / 'qeeg_guided_ssl_cnn_10seed_model_selection_summary.csv'}")
    print(f"Wrote {metrics_dir / 'qeeg_guided_ssl_cnn_seed_stability_summary.csv'}")
    print(f"Wrote {metrics_dir / 'qeeg_guided_ssl_cnn_statistical_comparison.csv'}")
    print(f"Wrote {metrics_dir / 'qeeg_guided_ssl_cnn_subject_error_frequency.csv'}")
    print(f"Wrote {metrics_dir / 'sub09_sub14_qeeg_guided_scores_summary.csv'}")
    print(f"Wrote {docs_dir / 'qeeg_guided_ssl_cnn_results.md'}")


def _collect_qeeg_guided_outputs(
    metrics_dir: Path,
    predictions_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    prediction_frames = []
    for seed in SEEDS:
        summary_path = metrics_dir / f"qeeg_guided_ssl_cnn_seed{seed}_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing qEEG-guided seed summary: {summary_path}")
        metrics = pd.read_csv(summary_path)
        if metrics.empty or "run_name" not in metrics.columns:
            raise ValueError(f"qEEG-guided summary is missing run_name: {summary_path}")
        run_name = str(metrics.loc[0, "run_name"])
        prediction_path = predictions_dir / f"dl_loso_predictions_{_safe_run_name(run_name)}.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing qEEG-guided prediction file: {prediction_path}")
        predictions = pd.read_csv(prediction_path)
        predictions["seed"] = seed
        predictions["model_group"] = "qeeg_guided_graphsmooth_mvicreg"
        canonical = predictions_dir / f"dl_loso_predictions_qeeg_guided_graphsmooth_mvicreg_seed{seed}.csv"
        predictions.to_csv(canonical, index=False)
        metrics["seed"] = seed
        metrics["model_group"] = "qeeg_guided_graphsmooth_mvicreg"
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)
    return pd.concat(prediction_frames, ignore_index=True), pd.concat(metrics_frames, ignore_index=True)


def _collect_seeded_outputs(
    metrics_dir: Path,
    predictions_dir: Path,
    *,
    metric_pattern: str,
    prediction_pattern: str,
    model_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    prediction_frames = []
    for seed in SEEDS:
        metric_path = metrics_dir / metric_pattern.format(seed=seed)
        prediction_path = predictions_dir / prediction_pattern.format(seed=seed)
        if not metric_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"Missing {model_group} seed {seed} output.")
        metrics = pd.read_csv(metric_path)
        predictions = pd.read_csv(prediction_path)
        metrics["seed"] = seed
        predictions["seed"] = seed
        metrics["model_group"] = model_group
        predictions["model_group"] = model_group
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)
    return pd.concat(prediction_frames, ignore_index=True), pd.concat(metrics_frames, ignore_index=True)


def _seedmean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("subject_id", as_index=False)
        .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
        .sort_values("subject_id")
        .assign(y_pred=lambda frame: (frame["y_score"] >= 0.5).astype(int))
        .reset_index(drop=True)
    )


def _run_qeeg_only_baseline(
    path_config,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    records = load_supervised_feature_records(
        path_config,
        labels,
        feature_kind="psd-fc-wpli",
        qeeg_features_enabled=True,
        qeeg_feature_names=(QEEG_FEATURE,),
    )
    by_subject = {record.subject_id: record for record in records}
    rows = []
    for fold in make_loso_folds([record.subject_id for record in records]):
        train_records = [by_subject[subject_id] for subject_id in fold.train_subject_ids]
        test_record = by_subject[fold.test_subject_id]
        train_x = np.stack([record.qeeg_features for record in train_records]).astype(np.float32)
        train_y = np.asarray([record.label for record in train_records], dtype=int)
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        train_x = (train_x - mean) / std
        test_x = (np.asarray(test_record.qeeg_features, dtype=np.float32).reshape(1, -1) - mean) / std
        model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", random_state=0)
        model.fit(train_x, train_y)
        score = float(model.predict_proba(test_x)[0, 1])
        rows.append(
            {
                "subject_id": test_record.subject_id,
                "y_true": int(test_record.label),
                "y_score": score,
                "y_pred": int(score >= 0.5),
                "fold_index": fold.fold_index,
                "model_group": "qeeg_only_logistic",
                "qeeg_feature_name": QEEG_FEATURE,
            }
        )
    predictions = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    metrics = binary_classification_metrics(predictions["y_true"].to_numpy(int), predictions["y_score"].to_numpy(float))
    return predictions, metrics


def _model_selection_summary(
    qeeg_guided_metrics: pd.DataFrame,
    qeeg_guided_seedmean: pd.DataFrame,
    no_ssl_qeeg_metrics: pd.DataFrame,
    no_ssl_qeeg_seedmean: pd.DataFrame,
    qeeg_only_metrics: dict[str, float],
    qeeg_only_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        _seeded_summary_row("qeeg_guided_graphsmooth_mvicreg", qeeg_guided_metrics, qeeg_guided_seedmean),
        _seeded_summary_row("no_ssl_cnn_qeeg_branch", no_ssl_qeeg_metrics, no_ssl_qeeg_seedmean),
        _single_prediction_row("qeeg_only_logistic", qeeg_only_metrics, qeeg_only_predictions),
        {
            "model_group": "no_ssl_cnn_reference_from_prior_locked_result",
            "n_seeds": 10,
            "mean_accuracy": 0.794736842105263,
            "std_accuracy": np.nan,
            "min_accuracy": np.nan,
            "max_accuracy": np.nan,
            "mean_balanced_accuracy": np.nan,
            "mean_roc_auc": np.nan,
            "mean_pr_auc": np.nan,
            "mean_brier_score": np.nan,
            "min_roc_auc": np.nan,
            "min_pr_auc": np.nan,
            "seedmean_accuracy": 0.8421052631578947,
            "seedmean_balanced_accuracy": 0.8333333333333333,
            "seedmean_roc_auc": 0.8111111111111111,
            "seedmean_pr_auc": 0.7824,
            "seedmean_brier_score": 0.1714,
            "notes": "Reference values from prior locked no-SSL 10-seed result document.",
        },
        {
            "model_group": "graphsmooth_mvicreg_no_qeeg_reference_from_prior_locked_result",
            "n_seeds": 10,
            "mean_accuracy": np.nan,
            "std_accuracy": np.nan,
            "min_accuracy": np.nan,
            "max_accuracy": np.nan,
            "mean_balanced_accuracy": np.nan,
            "mean_roc_auc": np.nan,
            "mean_pr_auc": np.nan,
            "mean_brier_score": np.nan,
            "min_roc_auc": np.nan,
            "min_pr_auc": np.nan,
            "seedmean_accuracy": 0.8421052631578947,
            "seedmean_balanced_accuracy": np.nan,
            "seedmean_roc_auc": 0.7777777777777778,
            "seedmean_pr_auc": 0.7422,
            "seedmean_brier_score": 0.1812,
            "notes": "Reference values from prior graph-smoothed masked-VICReg 10-seed result document.",
        },
    ]
    return pd.DataFrame(rows)


def _seeded_summary_row(model_group: str, metrics: pd.DataFrame, seedmean: pd.DataFrame) -> dict[str, object]:
    seedmean_metrics = binary_classification_metrics(seedmean["y_true"].to_numpy(int), seedmean["y_score"].to_numpy(float))
    return {
        "model_group": model_group,
        "n_seeds": int(metrics["seed"].nunique()),
        "mean_accuracy": float(metrics["accuracy"].mean()),
        "std_accuracy": float(metrics["accuracy"].std(ddof=0)),
        "min_accuracy": float(metrics["accuracy"].min()),
        "max_accuracy": float(metrics["accuracy"].max()),
        "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
        "mean_roc_auc": float(metrics["roc_auc"].mean()),
        "mean_pr_auc": float(metrics["pr_auc"].mean()),
        "mean_brier_score": float(metrics["brier_score"].mean()),
        "min_roc_auc": float(metrics["roc_auc"].min()),
        "min_pr_auc": float(metrics["pr_auc"].min()),
        "seedmean_accuracy": seedmean_metrics["accuracy"],
        "seedmean_balanced_accuracy": seedmean_metrics["balanced_accuracy"],
        "seedmean_roc_auc": seedmean_metrics["roc_auc"],
        "seedmean_pr_auc": seedmean_metrics["pr_auc"],
        "seedmean_brier_score": seedmean_metrics["brier_score"],
        "notes": "",
    }


def _single_prediction_row(model_group: str, metrics: dict[str, float], predictions: pd.DataFrame) -> dict[str, object]:
    return {
        "model_group": model_group,
        "n_seeds": 1,
        "mean_accuracy": metrics["accuracy"],
        "std_accuracy": 0.0,
        "min_accuracy": metrics["accuracy"],
        "max_accuracy": metrics["accuracy"],
        "mean_balanced_accuracy": metrics["balanced_accuracy"],
        "mean_roc_auc": metrics["roc_auc"],
        "mean_pr_auc": metrics["pr_auc"],
        "mean_brier_score": metrics["brier_score"],
        "min_roc_auc": metrics["roc_auc"],
        "min_pr_auc": metrics["pr_auc"],
        "seedmean_accuracy": metrics["accuracy"],
        "seedmean_balanced_accuracy": metrics["balanced_accuracy"],
        "seedmean_roc_auc": metrics["roc_auc"],
        "seedmean_pr_auc": metrics["pr_auc"],
        "seedmean_brier_score": metrics["brier_score"],
        "notes": f"Deterministic one-feature ridge logistic baseline; n_patients={len(predictions)}.",
    }


def _seed_stability_summary(qeeg_guided_metrics: pd.DataFrame, no_ssl_qeeg_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in (
        ("qeeg_guided_graphsmooth_mvicreg", qeeg_guided_metrics),
        ("no_ssl_cnn_qeeg_branch", no_ssl_qeeg_metrics),
    ):
        rows.append(
            {
                "model_group": name,
                "n_seeds": int(frame["seed"].nunique()),
                "mean_accuracy": float(frame["accuracy"].mean()),
                "std_accuracy": float(frame["accuracy"].std(ddof=0)),
                "min_accuracy": float(frame["accuracy"].min()),
                "max_accuracy": float(frame["accuracy"].max()),
                "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
                "mean_roc_auc": float(frame["roc_auc"].mean()),
                "mean_pr_auc": float(frame["pr_auc"].mean()),
                "mean_brier_score": float(frame["brier_score"].mean()),
                "min_roc_auc": float(frame["roc_auc"].min()),
                "min_pr_auc": float(frame["pr_auc"].min()),
            }
        )
    return pd.DataFrame(rows)


def _subject_error_frequency(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["y_pred"] = (frame["y_score"] >= 0.5).astype(int)
    frame["error"] = frame["y_pred"] != frame["y_true"].astype(int)
    return (
        frame.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            error_count=("error", "sum"),
            n_seeds=("seed", "nunique"),
            mean_y_score=("y_score", "mean"),
            min_y_score=("y_score", "min"),
            max_y_score=("y_score", "max"),
        )
        .assign(error_frequency=lambda data: data["error_count"] / data["n_seeds"])
        .sort_values(["error_count", "subject_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _watched_subject_scores(
    qeeg_guided_predictions: pd.DataFrame,
    no_ssl_qeeg_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, frame in (
        ("qeeg_guided_graphsmooth_mvicreg", qeeg_guided_predictions),
        ("no_ssl_cnn_qeeg_branch", no_ssl_qeeg_predictions),
    ):
        subset = frame.loc[frame["subject_id"].isin(["sub05", "sub09", "sub13", "sub14"])].copy()
        subset["y_pred"] = (subset["y_score"] >= 0.5).astype(int)
        subset["error"] = subset["y_pred"] != subset["y_true"].astype(int)
        grouped = subset.groupby("subject_id", as_index=False).agg(
            y_true=("y_true", "first"),
            mean_y_score=("y_score", "mean"),
            min_y_score=("y_score", "min"),
            max_y_score=("y_score", "max"),
            error_count=("error", "sum"),
            n_seeds=("seed", "nunique"),
        )
        grouped["model_group"] = name
        grouped["error_frequency"] = grouped["error_count"] / grouped["n_seeds"]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True).sort_values(["subject_id", "model_group"]).reset_index(drop=True)


def _statistical_comparison(
    predictions_dir: Path,
    qeeg_guided_seedmean: pd.DataFrame,
    no_ssl_qeeg_seedmean: pd.DataFrame,
    qeeg_only_predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
) -> pd.DataFrame:
    comparisons = [
        ("no_ssl_cnn_qeeg_branch_seedmean", no_ssl_qeeg_seedmean, "qeeg_guided_graphsmooth_mvicreg_seedmean", qeeg_guided_seedmean),
        ("qeeg_only_logistic", qeeg_only_predictions, "qeeg_guided_graphsmooth_mvicreg_seedmean", qeeg_guided_seedmean),
    ]
    no_ssl_reference = predictions_dir / "dl_loso_predictions_no_ssl_stability_schemeA_t8_lr2e3_wd1e5_d0_seedensemble.csv"
    if no_ssl_reference.exists():
        comparisons.append(
            (
                "available_no_ssl_schemeA_t8_seedensemble6",
                pd.read_csv(no_ssl_reference),
                "qeeg_guided_graphsmooth_mvicreg_seedmean",
                qeeg_guided_seedmean,
            )
        )

    rows = []
    for reference_name, reference, candidate_name, candidate in comparisons:
        stats = paired_patient_prediction_comparison(
            reference,
            candidate,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            random_state=7,
        )
        stats.insert(0, "reference_model", reference_name)
        stats.insert(1, "candidate_model", candidate_name)
        ceiling = paired_correctness_significance_ceiling(reference, candidate)
        stats["paired_correctness_reference_errors"] = ceiling["reference_errors"]
        stats["paired_correctness_candidate_errors"] = ceiling["candidate_errors"]
        stats["paired_correctness_favorable_discordant"] = ceiling["favorable_discordant"]
        stats["paired_correctness_adverse_discordant"] = ceiling["adverse_discordant"]
        stats["paired_correctness_exact_two_sided_p"] = ceiling["observed_exact_two_sided_p"]
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)


def _write_results_doc(
    path: Path,
    model_selection: pd.DataFrame,
    stability: pd.DataFrame,
    subject_errors: pd.DataFrame,
    watched: pd.DataFrame,
    comparison: pd.DataFrame,
    seedmean_path: Path,
) -> None:
    qg = model_selection.loc[model_selection["model_group"] == "qeeg_guided_graphsmooth_mvicreg"].iloc[0]
    no_ssl_q = model_selection.loc[model_selection["model_group"] == "no_ssl_cnn_qeeg_branch"].iloc[0]
    qeeg_only = model_selection.loc[model_selection["model_group"] == "qeeg_only_logistic"].iloc[0]
    reference = model_selection.loc[model_selection["model_group"] == "no_ssl_cnn_reference_from_prior_locked_result"].iloc[0]
    lines = [
        "# qEEG-guided SSL-CNN Results",
        "",
        "## Locked Candidate",
        "",
        "Model: qEEG-guided graph-smoothed masked-VICReg SSL-CNN.",
        "",
        "Primary qEEG feature: `qeeg_ec_global_slow_fast_bsi` only.",
        "",
        "CNN input features: PSD EO/EC `62 x 90` and WPLI EO/EC `1891 x 6`.",
        "",
        "Supervised branch: fold-local z-scored qEEG scalar clipped to `[-3, 3]`, `Linear(1,4) -> ReLU -> Linear(4,4) -> ReLU`, concatenated with the 64-dim PSD+WPLI CNN embedding.",
        "",
        "SSL objective: masked reconstruction + VICReg + WPLI graph smoothness + `0.05 * Huber(q_pred, qEEG_target)`.",
        "",
        "## Main 10-seed Result",
        "",
        f"- qEEG-guided SSL-CNN mean accuracy: `{qg.mean_accuracy:.4f}`; min accuracy: `{qg.min_accuracy:.4f}`; std accuracy: `{qg.std_accuracy:.4f}`.",
        f"- qEEG-guided SSL-CNN mean ROC AUC: `{qg.mean_roc_auc:.4f}`; mean PR AUC: `{qg.mean_pr_auc:.4f}`; mean Brier: `{qg.mean_brier_score:.4f}`.",
        f"- qEEG-guided SSL-CNN seedmean10 accuracy: `{qg.seedmean_accuracy:.4f}`; ROC AUC: `{qg.seedmean_roc_auc:.4f}`; PR AUC: `{qg.seedmean_pr_auc:.4f}`; Brier: `{qg.seedmean_brier_score:.4f}`.",
        f"- Prior locked no-SSL reference seedmean10 accuracy: `{reference.seedmean_accuracy:.4f}`; ROC AUC: `{reference.seedmean_roc_auc:.4f}`; PR AUC: `{reference.seedmean_pr_auc:.4f}`; Brier: `{reference.seedmean_brier_score:.4f}`.",
        "",
        "## Required Questions",
        "",
        "1. What is qEEG EC global slow/fast BSI?",
        "",
        "It is an eyes-closed, global hemispheric Brain Symmetry Index computed from PSD slow-band burden relative to fast-band preservation. It summarizes left-right asymmetry in slow/fast EEG balance.",
        "",
        "2. Why is it biologically plausible?",
        "",
        "Post-stroke recovery is plausibly related to asymmetric slowing, impaired alpha/beta preservation, and hemispheric imbalance. The feature is low-dimensional, EEG-derived, and does not use labels.",
        "",
        "3. Does qEEG branch improve no-SSL CNN?",
        "",
        f"No. no-SSL CNN + qEEG branch mean accuracy was `{no_ssl_q.mean_accuracy:.4f}` with min accuracy `{no_ssl_q.min_accuracy:.4f}`, worse than the prior no-SSL 10-seed mean accuracy `{reference.mean_accuracy:.4f}`.",
        "",
        "4. Does qEEG branch improve SSL-CNN?",
        "",
        "No clear improvement. The qEEG-guided SSL-CNN seedmean accuracy matched the prior no-SSL seedmean accuracy, but seed-level mean accuracy and Brier were worse than the no-SSL reference.",
        "",
        "5. Does qEEG auxiliary SSL improve over qEEG branch alone?",
        "",
        f"Only partially. It improved over no-SSL+qEEG branch in mean accuracy (`{qg.mean_accuracy:.4f}` vs `{no_ssl_q.mean_accuracy:.4f}`), but did not outperform the original no-SSL reference.",
        "",
        "6. Does it improve 10-seed mean accuracy?",
        "",
        f"No. qEEG-guided mean accuracy `{qg.mean_accuracy:.4f}` is below the prior locked no-SSL mean accuracy `{reference.mean_accuracy:.4f}`.",
        "",
        "7. Does it improve min accuracy?",
        "",
        f"No clear claim. qEEG-guided min accuracy was `{qg.min_accuracy:.4f}`; the prior no-SSL document did not preserve a comparable min-accuracy row in the summary file.",
        "",
        "8. Does it reduce seed std?",
        "",
        f"It reduced variance relative to the newly run no-SSL+qEEG branch (`{qg.std_accuracy:.4f}` vs `{no_ssl_q.std_accuracy:.4f}`), but that branch itself was weak.",
        "",
        "9. Does it improve ROC AUC, PR AUC, Brier?",
        "",
        f"No versus the prior locked no-SSL seedmean reference: qEEG-guided seedmean ROC AUC `{qg.seedmean_roc_auc:.4f}`, PR AUC `{qg.seedmean_pr_auc:.4f}`, Brier `{qg.seedmean_brier_score:.4f}`.",
        "",
        "10. Does it reduce sub09/sub14/sub05/sub13 errors?",
        "",
        "Not enough. The watched-subject table shows persistent errors in this group; these subjects were monitored post hoc, not optimized directly.",
        "",
        "11. Is improvement mainly qEEG, SSL, or combination?",
        "",
        "This run does not support a main-model improvement from either qEEG branch alone or qEEG auxiliary SSL. The earlier qEEG residual result remains stronger, but it is a post-hoc calibration candidate rather than this in-model branch.",
        "",
        "12. Should this be main model, supplementary model, or hypothesis-generating candidate?",
        "",
        "Supplementary negative/neutral model test. It is useful evidence that simply moving the qEEG residual into a small supervised branch and auxiliary SSL target does not reproduce the prior residual-calibration gain.",
        "",
        "## Output Files",
        "",
        f"- Seedmean predictions: `{seedmean_path.relative_to(PROJECT_ROOT)}`",
        "- Model selection summary: `results/metrics/qeeg_guided_ssl_cnn_10seed_model_selection_summary.csv`",
        "- Seed stability summary: `results/metrics/qeeg_guided_ssl_cnn_seed_stability_summary.csv`",
        "- Statistical comparison: `results/metrics/qeeg_guided_ssl_cnn_statistical_comparison.csv`",
        "- Subject errors: `results/metrics/qeeg_guided_ssl_cnn_subject_error_frequency.csv`",
        "- Watched subjects: `results/metrics/sub09_sub14_qeeg_guided_scores_summary.csv`",
        "",
        "## Caveats",
        "",
        "- Seeds are repeated training runs, not independent patients.",
        "- The available paired comparison against no-SSL uses the local no-SSL schemeA 6-seed ensemble file when present; the prior 10-seed no-SSL values are included as locked reference metrics.",
        "- No clinical generalization is claimed without external validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
