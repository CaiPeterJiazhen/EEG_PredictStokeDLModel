from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn

from eeg_recovery.explainability.occlusion import compute_occlusion_delta


class _SimpleModel(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = batch["psd_eo"].sum(dim=(1, 2), keepdim=False).reshape(-1, 1)
        logits = logits + 0.5 * batch["wpli_eo"].sum(dim=(1, 2), keepdim=False).reshape(-1, 1)
        return {"classification_logits": logits, "classification_probability": torch.sigmoid(logits)}


def test_occlusion_output_has_delta_logit_and_probability() -> None:
    batch = {
        "psd_eo": torch.ones((1, 2, 3)),
        "wpli_eo": torch.ones((1, 2, 2)),
        "y": torch.tensor([[1.0]]),
    }

    result = compute_occlusion_delta(
        _SimpleModel(),
        batch,
        occlude=lambda occluded: occluded["psd_eo"].zero_(),
    )

    assert result["delta_logit"] > 0
    assert result["delta_probability"] > 0


def test_branch_state_gate_weights_can_be_saved(tmp_path: Path) -> None:
    module = _load_explainability_script()
    sample = module.ExplainedSample(
        subject_id="sub01",
        fold_index=0,
        seed=0,
        y_true=1,
        y_score=0.75,
        y_pred=1,
        residual=1.0,
        signed_distance=2.0,
    )
    rows = module._gate_rows(
        sample,
        {
            "psd_state_weights": torch.tensor([[0.25, 0.75]]),
            "wpli_state_weights": torch.tensor([[0.60, 0.40]]),
        },
    )
    path = tmp_path / "branch_state_gate_weights.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    saved = pd.read_csv(path)
    assert {"subject_id", "branch", "state", "gate_weight"} <= set(saved.columns)
    assert len(saved) == 4
    assert set(saved["branch"]) == {"psd", "wpli"}
    assert set(saved["state"]) == {"EO", "EC"}


def _load_explainability_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "31_explain_residual_aware_ssl_cnn.py"
    spec = spec_from_file_location("explainability_script_for_tests", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
