from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.training.patient_level_comparison import paired_patient_prediction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-level paired bootstrap and random-swap permutation comparison for two prediction CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    comparison = paired_patient_prediction_comparison(
        pd.read_csv(args.reference),
        pd.read_csv(args.candidate),
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    comparison["reference_name"] = args.reference_name
    comparison["candidate_name"] = args.candidate_name
    comparison["reference_path"] = str(Path(args.reference))
    comparison["candidate_path"] = str(Path(args.candidate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(f"Wrote comparison: {output}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
