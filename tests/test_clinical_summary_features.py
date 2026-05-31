from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_recovery.features.clinical_summary import baseline_clinical_feature_matrix


def test_baseline_clinical_feature_matrix_uses_only_allowed_model_inputs():
    metadata = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02"],
            "age": [71, 63],
            "sex": ["M", "F"],
            "duration": [17, 41],
            "affected_hand": ["left", "right"],
            "FMA_pre": [63, 10],
            "MBI_pre": [80, 45],
            "FMA_post": [65, 12],
            "MBI_post": [90, 55],
            "Delta_FMA_obs": [2, 2],
            "Residual": [0.1, 37.2],
            "label": [1, 0],
        }
    )

    matrix, names = baseline_clinical_feature_matrix(metadata)

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert {"age", "duration", "FMA_pre", "MBI_pre"}.issubset(names)
    assert any(name.startswith("sex=") for name in names)
    assert any(name.startswith("affected_hand=") for name in names)
    assert not any("post" in name.lower() for name in names)
    assert "Residual" not in names
    assert "label" not in names


def test_baseline_clinical_feature_matrix_requires_subject_alignment_column():
    metadata = pd.DataFrame({"age": [71], "FMA_pre": [63], "MBI_pre": [80]})

    try:
        baseline_clinical_feature_matrix(metadata)
    except ValueError as exc:
        assert "subject_id" in str(exc)
    else:
        raise AssertionError("Expected missing subject_id to fail.")
