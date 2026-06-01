from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper around a base PyTorch optimizer."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        base_optimizer_cls: type[torch.optim.Optimizer],
        *,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if rho <= 0:
            raise ValueError("rho must be positive.")
        defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                perturbation = torch.pow(parameter, 2) * parameter.grad if group["adaptive"] else parameter.grad
                perturbation = perturbation * scale.to(parameter)
                parameter.add_(perturbation)
                self.state[parameter]["e_w"] = perturbation
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.sub_(self.state[parameter].get("e_w", torch.zeros_like(parameter)))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor:
        if closure is None:
            raise RuntimeError("SAM requires a closure that performs forward, backward, and returns the loss.")
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step(zero_grad=True)
        return loss.detach()

    def zero_grad(self, set_to_none: bool = False) -> None:  # type: ignore[override]
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        self.base_optimizer.load_state_dict(state_dict)

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                scale = torch.abs(parameter) if group["adaptive"] else 1.0
                norms.append((scale * parameter.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def build_adamw_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_swa_model(model: nn.Module) -> AveragedModel:
    return AveragedModel(model)


def update_swa_batch_norm(batches: Iterable[Any], model: nn.Module) -> None:
    batch_norm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norm_modules:
        return
    was_training = model.training
    momenta = {module: module.momentum for module in batch_norm_modules}
    for module in batch_norm_modules:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    with torch.no_grad():
        for batch in batches:
            model(batch)
    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)
