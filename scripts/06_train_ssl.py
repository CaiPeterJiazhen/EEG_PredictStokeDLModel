from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.train_ssl import (
    SSL_DATA_SCOPES,
    SSLTrainingConfig,
    build_ssl_file_records,
    run_ssl_pretraining,
    write_ssl_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run self-supervised raw EEG pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "paths.example.yaml",
        help="Path YAML containing external EEG/workbook paths and output_root.",
    )
    parser.add_argument("--data-scope", choices=SSL_DATA_SCOPES, default="supervised-baseline")
    parser.add_argument(
        "--strict-loso-test-subject",
        default=None,
        help="Optional supervised test subject to exclude from SSL pretraining, e.g. sub01.",
    )
    parser.add_argument("--objective", choices=("contrastive", "reconstruction", "combined"), default="combined")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-samples", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-mask-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional cap for smoke runs. Omit for the full selected SSL pool.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output run name. Defaults to a name derived from data scope, objective, and seed.",
    )
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    records = build_ssl_file_records(
        path_config,
        supervised_ids,
        data_scope=args.data_scope,
        strict_loso_test_subject_id=args.strict_loso_test_subject,
    )
    if args.limit_records is not None:
        if args.limit_records < 1:
            raise SystemExit("--limit-records must be at least 1 when provided.")
        records = records[: args.limit_records]
    if not records:
        raise SystemExit("No EEG records selected for SSL pretraining.")

    training_config = SSLTrainingConfig(
        data_scope=args.data_scope,
        objective=args.objective,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_samples=args.crop_samples,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_mask_fraction=args.reconstruction_mask_fraction,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    history = run_ssl_pretraining(records, training_config)
    run_name = args.run_name or f"{args.data_scope}_{args.objective}_seed{args.seed}"
    history_path = write_ssl_outputs(history, path_config.output_root, run_name=run_name)
    print(f"Selected SSL records: {len(records)}")
    print(f"Wrote SSL history: {history_path}")


if __name__ == "__main__":
    main()
