from __future__ import annotations

import math

import torch


@torch.no_grad()
def latent_statistics(
    tokens: torch.Tensor, low_variance_threshold: float = 1e-3
) -> dict[str, float]:
    if tokens.ndim < 2:
        raise ValueError("tokens must have a feature dimension")
    flat = tokens.reshape(-1, tokens.shape[-1]).float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    variance = centered.var(dim=0, unbiased=False)
    covariance = centered.T @ centered / max(1, flat.shape[0])
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp()
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    sample = normalized[: min(512, normalized.shape[0])]
    cosine = sample @ sample.T
    if sample.shape[0] > 1:
        off_diagonal = (cosine.sum() - cosine.diag().sum()) / (
            sample.shape[0] * (sample.shape[0] - 1)
        )
    else:
        off_diagonal = cosine.new_zeros(())
    return {
        "mean_std": math.sqrt(max(0.0, float(variance.mean()))),
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank / tokens.shape[-1]),
        "low_variance_fraction": float((variance < low_variance_threshold).float().mean()),
        "mean_pairwise_cosine": float(off_diagonal),
    }
