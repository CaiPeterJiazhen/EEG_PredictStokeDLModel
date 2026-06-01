from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


def test_sample_from_metadata_uses_configured_output_root_for_predictions(tmp_path: Path) -> None:
    module = _load_explainability_script()
    prediction_dir = tmp_path / "results" / "predictions"
    prediction_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": ["sub01"],
            "y_true": [1],
            "y_score": [0.73],
            "y_pred": [1],
        }
    ).to_csv(
        prediction_dir / "dl_loso_predictions_patient_barlow_residualaware_highrank_swa_clsalpha1_seed0.csv",
        index=False,
    )
    label_table = pd.DataFrame(
        {
            "subject_id": ["sub01"],
            "label": [1],
            "Residual": [0.25],
        }
    ).set_index("subject_id", drop=False)
    targets = pd.DataFrame(
        {
            "subject_id": ["sub01"],
            "signed_distance": [1.25],
        }
    ).set_index("subject_id", drop=False)

    sample = module._sample_from_metadata(
        {"test_subject_id": "sub01", "fold_index": 0, "seed": 0},
        label_table,
        targets,
        output_root=tmp_path,
    )

    assert sample.subject_id == "sub01"
    assert sample.y_score == 0.73
    assert sample.signed_distance == 1.25


def _load_explainability_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "31_explain_residual_aware_ssl_cnn.py"
    spec = spec_from_file_location("explainability_script_path_test", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

