from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def test_clinical_specs_do_not_use_post_treatment_features() -> None:
    module = _load_clinical_script()

    forbidden = {"FMA_post", "MBI_post", "Delta_FMA_obs", "Residual", "label"}
    for spec in module.clinical_model_specs():
        assert forbidden.isdisjoint(spec.features)


def test_fold_preprocessor_excludes_test_subject_from_scaler_and_imputer() -> None:
    module = _load_clinical_script()
    frame = _synthetic_label_table(n_subjects=19)
    spec = module.ClinicalModelSpec("fma_pre_only", ("FMA_pre",))
    test_subject = frame.loc[0, "subject_id"]
    train = frame[frame["subject_id"] != test_subject]

    preprocessor = module.fit_clinical_preprocessor(train, spec.features)
    transformed_test = preprocessor.transform(frame[frame["subject_id"] == test_subject])

    train_mean = train["FMA_pre"].mean()
    all_subject_mean = frame["FMA_pre"].mean()
    imputer_mean = float(preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[0])

    assert imputer_mean == train_mean
    assert imputer_mean != all_subject_mean
    assert transformed_test.shape == (1, 1)


def test_run_clinical_baselines_loso_outputs_19_predictions() -> None:
    module = _load_clinical_script()
    frame = _synthetic_label_table(n_subjects=19)

    predictions, metrics = module.run_clinical_baselines_loso(
        frame,
        specs=(module.ClinicalModelSpec("clinical_only", ("age", "sex", "duration", "affected_hand", "FMA_pre", "MBI_pre")),),
    )

    assert len(predictions) == 19
    assert predictions["subject_id"].nunique() == 19
    assert set(["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]) <= set(metrics.columns)
    assert metrics.loc[0, "model"] == "clinical_only"


def _synthetic_label_table(n_subjects: int) -> pd.DataFrame:
    subjects = [f"sub{index:02d}" for index in range(1, n_subjects + 1)]
    labels = np.asarray([index % 2 for index in range(n_subjects)], dtype=int)
    return pd.DataFrame(
        {
            "subject_id": subjects,
            "age": np.linspace(42, 70, n_subjects),
            "sex": ["M" if index % 2 else "F" for index in range(n_subjects)],
            "duration": np.linspace(1, 12, n_subjects),
            "affected_hand": ["left" if index % 2 else "right" for index in range(n_subjects)],
            "FMA_pre": np.linspace(5, 55, n_subjects),
            "FMA_post": np.linspace(10, 64, n_subjects),
            "MBI_pre": np.linspace(20, 80, n_subjects),
            "MBI_post": np.linspace(30, 95, n_subjects),
            "Delta_FMA_pred": np.linspace(40, 5, n_subjects),
            "Delta_FMA_obs": np.linspace(5, 30, n_subjects),
            "Residual": np.linspace(10, -10, n_subjects),
            "label": labels,
        }
    )


def _load_clinical_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "32_train_clinical_baselines.py"
    spec = spec_from_file_location("clinical_baseline_script_test", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

