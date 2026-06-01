from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def test_post_treatment_columns_are_rejected_as_clinical_inputs() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="Post-treatment"):
        module.validate_clinical_predictors(["age", "FMA_pre", "FMA_post"])

    with pytest.raises(ValueError, match="Post-treatment"):
        module.validate_clinical_predictors(["age", "Residual"])


def test_mixed_preprocessor_is_fit_on_training_subjects_only() -> None:
    module = _load_script()
    frame = _toy_clinical_table(6)
    train = frame[frame["subject_id"] != "sub01"]
    test = frame[frame["subject_id"] == "sub01"]

    preprocessor = module.fit_fold_mixed_preprocessor(
        train,
        numeric_columns=["age", "FMA_pre"],
        categorical_columns=["sex"],
    )
    transformed = preprocessor.transform(test[["age", "FMA_pre", "sex"]])
    numeric = preprocessor.named_transformers_["numeric"]

    assert numeric.named_steps["imputer"].statistics_[0] == pytest.approx(train["age"].mean())
    assert numeric.named_steps["imputer"].statistics_[0] != pytest.approx(frame["age"].mean())
    assert transformed.shape[0] == 1


def test_clinical_loso_predictions_are_patient_level() -> None:
    module = _load_script()
    frame = _toy_clinical_table(8)

    predictions, metrics = module.run_loso_tabular_models(
        frame,
        label_table=frame[["subject_id", "label"]],
        input_columns=["age", "sex", "duration", "affected_hand", "FMA_pre", "MBI_pre"],
        model_names=["logistic_l2"],
        feature_selection="none",
        random_state=0,
    )

    assert len(predictions) == 8
    assert predictions["subject_id"].nunique() == 8
    assert metrics.loc[0, "model"] == "clinical_only_logistic_l2"
    assert metrics.loc[0, "n_subjects"] == 8


def _toy_clinical_table(n_subjects: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, n_subjects + 1)],
            "age": np.linspace(40, 70, n_subjects),
            "sex": ["F", "M"] * (n_subjects // 2),
            "duration": np.linspace(1, 6, n_subjects),
            "affected_hand": ["left", "right"] * (n_subjects // 2),
            "FMA_pre": np.linspace(5, 60, n_subjects),
            "MBI_pre": np.linspace(20, 90, n_subjects),
            "FMA_post": np.linspace(10, 65, n_subjects),
            "Residual": np.linspace(-3, 4, n_subjects),
            "label": [1, 0] * (n_subjects // 2),
        }
    )


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "39_train_clinical_and_incremental_baselines.py"
    spec = spec_from_file_location("clinical_incremental_script_test", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
