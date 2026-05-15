from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import torch


def build_reduce_on_plateau(
    optimizer: torch.optim.Optimizer,
    *,
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 5,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
    )


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "min"
    min_delta: float = 0.0
    best_score: float | None = None
    bad_epochs: int = 0
    best_state_dict: dict[str, torch.Tensor] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")

    def step(self, score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None or self._is_improvement(score):
            self.best_score = float(score)
            self.bad_epochs = 0
            self.best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in deepcopy(model.state_dict()).items()
            }
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        if self.best_state_dict is None:
            return
        model.load_state_dict(self.best_state_dict)

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta
