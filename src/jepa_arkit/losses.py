from __future__ import annotations

import torch
from torch.nn import functional as F


def confidence_weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor | None = None,
    dimension_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    if confidence is None:
        return element.mean()
    while confidence.ndim < element.ndim:
        confidence = confidence.unsqueeze(-1)
    weights = confidence.expand_as(element).clamp_min(0)
    if dimension_weights is not None:
        weights = weights * dimension_weights.view(1, 1, -1)
    return (element * weights).sum() / weights.sum().clamp_min(1e-8)


def velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dimension_weights: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    element = F.smooth_l1_loss(
        prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1], reduction="none"
    )
    if dimension_weights is not None:
        element = element * dimension_weights.view(1, 1, -1)
    if confidence is not None:
        temporal_weights = torch.minimum(confidence[:, 1:], confidence[:, :-1]).unsqueeze(-1)
        element = element * temporal_weights
        dimensions = dimension_weights.sum() if dimension_weights is not None else element.shape[-1]
        denominator = temporal_weights.sum() * dimensions
        return element.sum() / denominator.clamp_min(1e-8)
    return element.mean()


def acceleration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dimension_weights: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape[1] < 3:
        return prediction.new_zeros(())
    pred_velocity = prediction[:, 1:] - prediction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    element = F.smooth_l1_loss(
        pred_velocity[:, 1:] - pred_velocity[:, :-1],
        target_velocity[:, 1:] - target_velocity[:, :-1],
        reduction="none",
    )
    if dimension_weights is not None:
        element = element * dimension_weights.view(1, 1, -1)
    if confidence is not None:
        temporal_weights = torch.minimum(
            torch.minimum(confidence[:, 2:], confidence[:, 1:-1]), confidence[:, :-2]
        ).unsqueeze(-1)
        element = element * temporal_weights
        dimensions = dimension_weights.sum() if dimension_weights is not None else element.shape[-1]
        denominator = temporal_weights.sum() * dimensions
        return element.sum() / denominator.clamp_min(1e-8)
    return element.mean()
