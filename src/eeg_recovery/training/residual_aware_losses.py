from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_ranking_loss(
    scores: torch.Tensor,
    signed_distance: torch.Tensor,
    *,
    rank_margin: float,
) -> torch.Tensor:
    scores = scores.reshape(-1).float()
    signed_distance = signed_distance.reshape(-1).float()
    if scores.numel() < 2:
        return scores.sum() * 0.0
    diff_target = signed_distance[:, None] - signed_distance[None, :]
    diff_score = scores[:, None] - scores[None, :]
    mask = diff_target > float(rank_margin)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    losses = F.softplus(-diff_score[mask])
    return losses.mean()


def residual_aware_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_reg: float,
    lambda_rank: float,
    lambda_soft: float,
    rank_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["classification_logits"]
    residual_score = outputs["residual_score"]
    y = batch["y"].to(dtype=logits.dtype, device=logits.device)
    signed_distance_z = batch["signed_distance_z"].to(dtype=residual_score.dtype, device=residual_score.device)
    soft_y = batch["soft_y"].to(dtype=logits.dtype, device=logits.device)

    loss_bce = F.binary_cross_entropy_with_logits(logits, y)
    loss_reg = F.huber_loss(residual_score, signed_distance_z)
    loss_rank = pairwise_ranking_loss(residual_score, signed_distance_z, rank_margin=rank_margin)
    loss_soft = F.binary_cross_entropy_with_logits(logits, soft_y)
    total = loss_bce + float(lambda_reg) * loss_reg + float(lambda_rank) * loss_rank + float(lambda_soft) * loss_soft
    if not torch.isfinite(total):
        raise ValueError("residual-aware multitask loss produced a non-finite value.")
    return total, {
        "loss_bce": loss_bce.detach(),
        "loss_residual_regression": loss_reg.detach(),
        "loss_pairwise_ranking": loss_rank.detach(),
        "loss_soft_label": loss_soft.detach(),
        "loss_total": total.detach(),
    }
