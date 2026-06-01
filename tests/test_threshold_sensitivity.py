from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


def test_fold_local_threshold_excludes_test_subject_residual() -> None:
    module = _load_script()
    labels = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04", "sub05"],
            "Residual": [-100.0, 0.0, 1.5, 3.0, 100.0],
        }
    )

    thresholds = module.fold_local_median_thresholds(labels)

    sub01_threshold = thresholds.loc[thresholds["subject_id"] == "sub01", "residual_threshold"].iloc[0]
    assert sub01_threshold == 2.25
    assert sub01_threshold != labels["Residual"].median()


def test_threshold_sensitivity_marks_near_threshold_exclusions() -> None:
    module = _load_script()
    labels = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "Residual": [0.0, 1.25, 1.75, 3.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "subject_id": labels["subject_id"],
            "y_score": [0.9, 0.8, 0.4, 0.2],
        }
    )

    rows = module.evaluate_threshold_sensitivity_for_predictions(
        predictions,
        labels,
        fixed_thresholds=[1.5],
        margins=[0.5],
    )

    excluded = rows[rows["label_definition"] == "exclude_margin_0.5_at_1.5"].iloc[0]
    assert excluded["n_excluded_near_threshold"] == 2
    assert excluded["n_subjects"] == 2


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "42_residual_label_threshold_sensitivity.py"
    spec = spec_from_file_location("threshold_sensitivity_script_test", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
