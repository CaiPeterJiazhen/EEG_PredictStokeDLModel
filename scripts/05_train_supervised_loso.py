from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
    write_dl_outputs,
    write_loss_history_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight supervised deep-learning LOSO models.",
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
    parser.add_argument("--architecture", choices=("dual_state", "fusion_3d", "multimodal"), default="dual_state")
    parser.add_argument(
        "--feature-kind",
        choices=("psd", "fc-wpli", "fc-icoh", "fc-both", "psd-fc-wpli", "psd-fc-icoh", "psd-fc-both"),
        default="psd",
    )
    parser.add_argument("--fusion", choices=("concat", "gated"), default="concat")
    parser.add_argument("--encoder", choices=("cnn", "linear", "gncnn", "rescnn"), default="cnn")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--embedding-adapter-dim", type=int, default=0)
    parser.add_argument("--embedding-adapter-scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qeeg-features-enabled",
        action="store_true",
        help="Add the locked fold-scaled qEEG biomarker branch to the multimodal CNN.",
    )
    parser.add_argument(
        "--qeeg-feature-name",
        default="qeeg_ec_global_slow_fast_bsi",
        help="Primary qEEG biomarker name. Primary mode accepts one feature only.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional filename-safe tag appended to prediction, metric, loss, and figure outputs.",
    )
    args = parser.parse_args()

    config = load_path_config(args.config)
    label_table = load_supervised_label_table(config)
    records = load_supervised_feature_records(
        config,
        label_table,
        feature_kind=args.feature_kind,
        qeeg_features_enabled=args.qeeg_features_enabled,
        qeeg_feature_names=(args.qeeg_feature_name,),
    )
    training_config = SupervisedTrainingConfig(
        architecture=args.architecture,
        feature_kind=args.feature_kind,
        fusion=args.fusion,
        encoder_kind=args.encoder,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        embedding_adapter_dim=args.embedding_adapter_dim,
        embedding_adapter_scale=args.embedding_adapter_scale,
        dropout=args.dropout,
        seed=args.seed,
        qeeg_features_enabled=args.qeeg_features_enabled,
        qeeg_feature_names=(args.qeeg_feature_name,),
        qeeg_primary_only=True,
        qeeg_hidden_dim=4,
        qeeg_clip_value=3.0,
    )
    predictions, metrics, loss_history = run_loso_supervised_with_history(records, training_config)
    prediction_path, metric_path = write_dl_outputs(
        predictions,
        metrics,
        config.output_root,
        run_name=args.output_tag,
    )
    loss_history_path, loss_curve_path = write_loss_history_outputs(
        loss_history,
        config.output_root,
        run_name=args.output_tag,
    )
    print(f"Wrote predictions: {prediction_path}")
    print(f"Wrote metrics: {metric_path}")
    print(f"Wrote loss history: {loss_history_path}")
    print(f"Wrote loss curve: {loss_curve_path}")


if __name__ == "__main__":
    main()
