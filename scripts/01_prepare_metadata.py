from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
# Allow running this script directly from a source checkout before package installation.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare supervised recovery metadata CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paths.example.yaml",
        help="Path YAML containing external data locations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <output_root>/data/processed/supervised_metadata.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_path_config(args.config)
    labels = load_supervised_label_table(config)

    output_path = args.output
    if output_path is None:
        output_path = config.output_root / "data" / "processed" / "supervised_metadata.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(labels)} supervised metadata rows to {output_path}")


if __name__ == "__main__":
    main()
