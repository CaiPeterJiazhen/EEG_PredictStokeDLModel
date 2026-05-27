from __future__ import annotations

import torch

from eeg_recovery.models.segment_mil_model import WPLISegmentAttentionMILModel


def _bag(n_eo: int = 5, n_ec: int = 4) -> dict[str, object]:
    return {
        "wpli_eo_segments": torch.randn(n_eo, 1891, 6),
        "wpli_ec_segments": torch.randn(n_ec, 1891, 6),
        "y": torch.tensor([1.0]),
        "subject_id": "sub01",
    }


def test_wpli_segment_attention_mil_forward_returns_patient_probability() -> None:
    model = WPLISegmentAttentionMILModel(embedding_dim=4, attention_hidden_dim=3, dropout=0.0)
    model.eval()

    probability, aux = model(_bag())

    assert probability.shape == (1, 1)
    assert torch.all((probability >= 0.0) & (probability <= 1.0))
    assert aux["n_eo_segments"] == 5
    assert aux["n_ec_segments"] == 4


def test_wpli_segment_attention_weights_sum_to_one() -> None:
    model = WPLISegmentAttentionMILModel(embedding_dim=4, attention_hidden_dim=3, dropout=0.0)
    model.eval()

    _, aux = model(_bag(n_eo=6, n_ec=2))

    assert aux["eo_attention_weights"].shape == (6,)
    assert aux["ec_attention_weights"].shape == (2,)
    torch.testing.assert_close(aux["eo_attention_weights"].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aux["ec_attention_weights"].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_wpli_segment_state_weights_sum_to_one() -> None:
    model = WPLISegmentAttentionMILModel(embedding_dim=4, attention_hidden_dim=3, dropout=0.0)
    model.eval()

    _, aux = model(_bag(n_eo=3, n_ec=7))

    assert aux["state_weights"].shape == (2,)
    torch.testing.assert_close(aux["state_weights"].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_wpli_segment_attention_mil_accepts_variable_length_bags() -> None:
    model = WPLISegmentAttentionMILModel(embedding_dim=4, attention_hidden_dim=3, dropout=0.0)
    model.eval()

    first_probability, first_aux = model(_bag(n_eo=1, n_ec=3))
    second_probability, second_aux = model(_bag(n_eo=8, n_ec=2))

    assert first_probability.shape == (1, 1)
    assert second_probability.shape == (1, 1)
    assert first_aux["eo_attention_weights"].shape == (1,)
    assert second_aux["eo_attention_weights"].shape == (8,)

