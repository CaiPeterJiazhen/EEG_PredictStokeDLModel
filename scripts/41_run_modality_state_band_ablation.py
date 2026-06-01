from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.feature_tables import load_fc_feature_table, load_psd_band_power_table, merge_feature_tables
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id


DEFAULT_MODELS = ("logistic_l1", "logistic_l2", "svm_rbf")
MOTOR_CHANNELS = ("C3", "C4", "C1", "C2", "C5", "C6", "FC3", "FC4", "CP3", "CP4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or refresh modality/state/band LOSO tabular ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--feature-selection", choices=("none", "selectk100"), default="selectk100")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    subjects = labels["subject_id"].tolist()
    psd = load_psd_band_power_table(output_root / "data" / "features" / "psd", subject_ids=subjects)
    wpli = load_fc_feature_table(output_root / "data" / "features" / "fc", metric="wpli", subject_ids=subjects)
    full = merge_feature_tables(psd, wpli)
    clinical_module = _load_clinical_module()

    prediction_frames = []
    metric_frames = []
    for spec in ablation_specs():
        feature_table = subset_feature_table(full, spec)
        input_columns = [column for column in feature_table.columns if column != "subject_id"]
        if not input_columns:
            metric_frames.append(_missing_metric_row(spec, "No columns matched this ablation."))
            continue
        predictions, metrics = clinical_module.run_loso_tabular_models(
            feature_table,
            label_table=labels[["subject_id", "label"]],
            input_columns=input_columns,
            model_names=args.models,
            feature_selection=args.feature_selection,
            random_state=args.random_state,
            model_family=f"ablation_{spec['name']}",
        )
        predictions["ablation_name"] = spec["name"]
        predictions["ablation_description"] = spec["description"]
        metrics["ablation_name"] = spec["name"]
        metrics["ablation_description"] = spec["description"]
        metrics["input_features"] = spec["description"]
        metrics["n_input_columns"] = len(input_columns)
        metrics["explainability_alignment"] = explainability_alignment_note(output_root, spec)
        prediction_frames.append(predictions)
        metric_frames.append(metrics)

    metrics_all = pd.concat(metric_frames, ignore_index=True, sort=False)
    predictions_all = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    metrics_dir = output_root / "results" / "metrics"
    tables_dir = output_root / "results" / "tables"
    predictions_dir = output_root / "results" / "predictions"
    docs_dir = output_root / "docs"
    for directory in (metrics_dir, tables_dir, predictions_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_all.to_csv(metrics_dir / "modality_state_band_ablation.csv", index=False)
    predictions_all.to_csv(predictions_dir / "modality_state_band_ablation_predictions.csv", index=False)
    (tables_dir / "modality_state_band_ablation.md").write_text(_to_markdown(metrics_all), encoding="utf-8")
    write_ablation_doc(docs_dir / "modality_state_band_ablation_results.md", metrics_all)
    print(f"Wrote {metrics_dir / 'modality_state_band_ablation.csv'}")


def ablation_specs() -> list[dict[str, object]]:
    return [
        {"name": "psd_only", "description": "PSD only, EO+EC, all bands", "families": ("psd",)},
        {"name": "wpli_only", "description": "WPLI only, EO+EC, all bands", "families": ("wpli",)},
        {"name": "psd_wpli", "description": "PSD + WPLI, EO+EC, all bands", "families": ("psd", "wpli")},
        {"name": "eo_only", "description": "EO only, PSD+WPLI", "families": ("psd", "wpli"), "states": ("EO",)},
        {"name": "ec_only", "description": "EC only, PSD+WPLI", "families": ("psd", "wpli"), "states": ("EC",)},
        {"name": "psd_eo_only", "description": "PSD EO only", "families": ("psd",), "states": ("EO",)},
        {"name": "wpli_ec_only", "description": "WPLI EC only", "families": ("wpli",), "states": ("EC",)},
        {"name": "psd_eo_wpli_ec", "description": "PSD EO plus WPLI EC", "custom": "psd_eo_wpli_ec"},
        {"name": "beta_medium_only", "description": "Beta medium band only, PSD+WPLI", "families": ("psd", "wpli"), "bands": ("beta_medium",)},
        {"name": "beta_high_only", "description": "Beta high band only, PSD+WPLI", "families": ("psd", "wpli"), "bands": ("beta_high",)},
        {"name": "beta_medium_beta_high", "description": "Beta medium + beta high bands, PSD+WPLI", "families": ("psd", "wpli"), "bands": ("beta_medium", "beta_high")},
        {"name": "motor_wpli_edges", "description": "Motor/stimulation-side related WPLI edges around C3/C4 network", "families": ("wpli",), "motor_edges": True},
    ]


def subset_feature_table(full: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    if spec.get("custom") == "psd_eo_wpli_ec":
        columns = [
            column
            for column in full.columns
            if column == "subject_id"
            or (column.startswith("psd_EO_"))
            or (column.startswith("fc_wpli_EC_"))
        ]
        return full.loc[:, columns].copy()
    selected = ["subject_id"]
    families = set(spec.get("families", ()))
    states = tuple(spec.get("states", ("EO", "EC")))
    bands = tuple(spec.get("bands", ()))
    for column in full.columns:
        if column == "subject_id":
            continue
        if "psd" in families and column.startswith("psd_") and _matches_state(column, states) and _matches_band(column, bands):
            selected.append(column)
        if "wpli" in families and column.startswith("fc_wpli_") and _matches_state(column, states) and _matches_band(column, bands):
            if spec.get("motor_edges") and not _is_motor_edge_column(column):
                continue
            selected.append(column)
    return full.loc[:, selected].copy()


def explainability_alignment_note(output_root: Path, spec: dict[str, object]) -> str:
    explain_root = output_root / "results" / "explainability"
    notes = []
    if (explain_root / "branch_state_gate_weights.csv").exists():
        notes.append("gate weights available")
    if (explain_root / "occlusion_branch_state.csv").exists():
        notes.append("branch/state occlusion available")
    if spec.get("bands") and (explain_root / "psd_channel_band_importance.csv").exists():
        notes.append("PSD band attribution available")
    if spec.get("motor_edges") and (explain_root / "wpli_top_edges.csv").exists():
        notes.append("WPLI edge attribution available")
    return "; ".join(notes) if notes else "explainability summaries not available"


def write_ablation_doc(path: Path, metrics: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = metrics.sort_values(["roc_auc", "brier_score"], ascending=[False, True]).groupby("ablation_name", as_index=False).head(1)
    lines = [
        "# Modality, State, And Band Ablation Results",
        "",
        "All rows are patient-level LOSO tabular ablations over baseline EEG features. Feature selection, if enabled, is fitted inside each training fold only.",
        "",
        "## Best Model Per Ablation",
        "",
        _to_markdown(best[["ablation_name", "model", "n_input_columns", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score", "explainability_alignment"]]),
        "",
        "## Interpretation",
        "",
        "Compare PSD-only, WPLI-only, and PSD+WPLI rows against gate weights and branch/state occlusion. Band-specific beta rows are support checks for whether attribution-localized beta features carry predictive signal when isolated. Motor WPLI edges are a targeted biological plausibility check, not a replacement for the full locked model.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _matches_state(column: str, states: Sequence[str]) -> bool:
    return any(f"_{state}_" in column for state in states)


def _matches_band(column: str, bands: Sequence[str]) -> bool:
    if not bands:
        return True
    normalized = column.lower()
    return any(normalized.endswith(f"_{str(band).lower()}") for band in bands)


def _is_motor_edge_column(column: str) -> bool:
    return any(f"_{channel}-" in column or f"-{channel}_" in column for channel in MOTOR_CHANNELS)


def _missing_metric_row(spec: dict[str, object], reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ablation_name": spec["name"],
                "ablation_description": spec["description"],
                "status": "skipped",
                "skip_reason": reason,
            }
        ]
    )


def _load_clinical_module():
    script_path = PROJECT_ROOT / "scripts" / "39_train_clinical_and_incremental_baselines.py"
    spec = spec_from_file_location("clinical_incremental_for_ablation", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
