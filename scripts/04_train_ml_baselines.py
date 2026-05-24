from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.feature_tables import (
    load_fc_feature_table,
    load_psd_band_power_table,
    merge_feature_tables,
)
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.train_baselines import (
    BaselinePreprocessorConfig,
    run_loso_baselines,
    write_baseline_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train traditional ML LOSO baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "paths.example.yaml",
        help=(
            "Path YAML containing external EEG/workbook paths and output_root. "
            "Project default: configs/paths.example.yaml."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=("psd", "fc-wpli", "fc-icoh", "psd-fc-wpli", "psd-fc-icoh"),
        default="psd",
    )
    parser.add_argument("--selector-k", type=int, default=100)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--disable-selector", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional filename-safe tag appended to prediction and metric outputs.",
    )
    args = parser.parse_args()

    config = load_path_config(args.config)
    label_table = load_supervised_label_table(config)
    subject_ids = label_table["subject_id"].tolist()
    psd_dir = config.output_root / "data" / "features" / "psd"
    fc_dir = config.output_root / "data" / "features" / "fc"

    feature_table = _load_feature_set(args.feature_set, psd_dir, fc_dir, subject_ids)
    preprocessor_config = BaselinePreprocessorConfig(
        selector_k=args.selector_k,
        pca_components=args.pca_components,
        enable_selector=not args.disable_selector,
        enable_pca=args.pca_components is not None,
    )
    predictions, metrics = run_loso_baselines(
        feature_table,
        label_table,
        preprocessor_config=preprocessor_config,
        random_state=args.random_state,
    )
    prediction_path, metric_path = write_baseline_outputs(
        predictions,
        metrics,
        config.output_root,
        run_name=args.output_tag,
    )
    print(f"Wrote predictions: {prediction_path}")
    print(f"Wrote metrics: {metric_path}")


def _load_feature_set(feature_set: str, psd_dir: Path, fc_dir: Path, subject_ids: list[str]):
    if feature_set == "psd":
        return load_psd_band_power_table(psd_dir, subject_ids=subject_ids)
    if feature_set == "fc-wpli":
        return load_fc_feature_table(fc_dir, metric="wpli", subject_ids=subject_ids)
    if feature_set == "fc-icoh":
        return load_fc_feature_table(fc_dir, metric="imaginary_coherence", subject_ids=subject_ids)
    if feature_set == "psd-fc-wpli":
        return merge_feature_tables(
            load_psd_band_power_table(psd_dir, subject_ids=subject_ids),
            load_fc_feature_table(fc_dir, metric="wpli", subject_ids=subject_ids),
        )
    if feature_set == "psd-fc-icoh":
        return merge_feature_tables(
            load_psd_band_power_table(psd_dir, subject_ids=subject_ids),
            load_fc_feature_table(fc_dir, metric="imaginary_coherence", subject_ids=subject_ids),
        )
    raise ValueError(f"Unsupported feature set: {feature_set}")


if __name__ == "__main__":
    main()
