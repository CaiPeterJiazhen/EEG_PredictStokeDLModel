from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from eeg_recovery.explainability.attribution import classification_logit, classification_probability


def clone_tensor_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_occlusion_delta(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    occlude: Callable[[dict[str, torch.Tensor]], None],
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        original_logit = classification_logit(model, batch).detach().reshape(-1)
        original_probability = classification_probability(model, batch).detach().reshape(-1)
        occluded = clone_tensor_batch(batch)
        occlude(occluded)
        occluded_logit = classification_logit(model, occluded).detach().reshape(-1)
        occluded_probability = classification_probability(model, occluded).detach().reshape(-1)
    model.train(model_was_training)
    return {
        "original_logit": float(original_logit.mean().cpu().item()),
        "occluded_logit": float(occluded_logit.mean().cpu().item()),
        "delta_logit": float((original_logit - occluded_logit).mean().cpu().item()),
        "original_probability": float(original_probability.mean().cpu().item()),
        "occluded_probability": float(occluded_probability.mean().cpu().item()),
        "delta_probability": float((original_probability - occluded_probability).mean().cpu().item()),
    }
