from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping

import torch
from torch import nn


def classification_logit(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(dict(batch))
    if isinstance(outputs, dict):
        if "classification_logits" not in outputs:
            raise KeyError("Model output must contain 'classification_logits' for attribution.")
        return outputs["classification_logits"]
    raise TypeError("Explainability requires a model returning a dict with classification_logits.")


def classification_probability(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(dict(batch))
    if isinstance(outputs, dict):
        if "classification_probability" in outputs:
            return outputs["classification_probability"]
        return torch.sigmoid(outputs["classification_logits"])
    raise TypeError("Explainability requires a model returning a dict with classification logits/probability.")


def manual_integrated_gradients(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    input_keys: Iterable[str],
    steps: int = 64,
    baseline: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute input attributions against the binary classification logit."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    input_keys = tuple(input_keys)
    if not input_keys:
        raise ValueError("input_keys must not be empty.")
    model_was_training = model.training
    model.eval()
    source = {key: value.detach() for key, value in batch.items()}
    base = {
        key: (
            baseline[key].detach().to(source[key].device, dtype=source[key].dtype)
            if baseline is not None and key in baseline
            else torch.zeros_like(source[key])
        )
        for key in input_keys
    }
    gradients = {key: torch.zeros_like(source[key], dtype=torch.float32) for key in input_keys}
    try:
        alphas = torch.linspace(
            1.0 / float(steps),
            1.0,
            steps,
            device=source[input_keys[0]].device,
            dtype=source[input_keys[0]].dtype,
        )
        interpolated = {
            key: _expand_batch_for_steps(value, steps).detach().clone()
            for key, value in source.items()
        }
        active_inputs: dict[str, torch.Tensor] = {}
        for key in input_keys:
            alpha_shape = (steps,) + (1,) * (source[key].ndim - 1)
            value = base[key] + alphas.reshape(alpha_shape) * (source[key] - base[key])
            value = value.detach().clone()
            value.requires_grad_(True)
            interpolated[key] = value
            active_inputs[key] = value
        logit = classification_logit(model, interpolated).sum()
        model.zero_grad(set_to_none=True)
        logit.backward()
        for key, value in active_inputs.items():
            if value.grad is None:
                raise RuntimeError(f"Missing gradient for attribution input {key!r}.")
            gradients[key] = value.grad.detach().sum(dim=0, keepdim=True)
        attributions = {
            key: (source[key] - base[key]).float() * gradients[key] / float(steps)
            for key in input_keys
        }
        for key, attribution in attributions.items():
            if not torch.isfinite(attribution).all():
                raise ValueError(f"Integrated gradients for {key} contain non-finite values.")
        return attributions
    finally:
        model.zero_grad(set_to_none=True)
        model.train(model_was_training)


def _expand_batch_for_steps(value: torch.Tensor, steps: int) -> torch.Tensor:
    if value.ndim == 0:
        return value.repeat(steps)
    if value.shape[0] != 1:
        raise ValueError("Vectorized integrated gradients expects one explained sample at a time.")
    return value.repeat((steps,) + (1,) * (value.ndim - 1))


def smoothgrad_integrated_gradients(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    input_keys: Iterable[str],
    steps: int = 64,
    noise_samples: int = 4,
    noise_std: float = 0.02,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    if noise_samples <= 0:
        raise ValueError("noise_samples must be positive.")
    generator = torch.Generator(device=next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu")
    generator.manual_seed(seed)
    input_keys = tuple(input_keys)
    totals: dict[str, torch.Tensor] | None = None
    for _ in range(noise_samples):
        noisy = dict(batch)
        for key in input_keys:
            noise = torch.randn(batch[key].shape, generator=generator, device=batch[key].device, dtype=batch[key].dtype)
            noisy[key] = batch[key] + noise_std * noise
        attrs = manual_integrated_gradients(model, noisy, input_keys=input_keys, steps=steps)
        if totals is None:
            totals = {key: value.detach().clone() for key, value in attrs.items()}
        else:
            for key in input_keys:
                totals[key] = totals[key] + attrs[key].detach()
    if totals is None:
        raise RuntimeError("No SmoothGrad samples were computed.")
    return {key: value / float(noise_samples) for key, value in totals.items()}


def attribution_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    x = first.detach().cpu().reshape(-1).float()
    y = second.detach().cpu().reshape(-1).float()
    if x.numel() != y.numel():
        raise ValueError("Attribution tensors must contain the same number of elements.")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom) <= 1e-12:
        return 0.0
    return float(torch.dot(x, y) / denom)


def randomize_classifier_weights(model: nn.Module, *, seed: int = 0) -> nn.Module:
    randomized = deepcopy(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    head = getattr(randomized, "classification_head", None)
    if head is None:
        raise AttributeError("Model has no classification_head to randomize.")
    for module in head.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                weight = torch.randn(module.weight.shape, generator=generator, dtype=module.weight.dtype)
                module.weight.copy_(weight.to(module.weight.device) * 0.1)
                if module.bias is not None:
                    bias = torch.randn(module.bias.shape, generator=generator, dtype=module.bias.dtype)
                    module.bias.copy_(bias.to(module.bias.device) * 0.1)
    return randomized
