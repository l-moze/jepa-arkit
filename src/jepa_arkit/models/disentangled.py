from __future__ import annotations

import torch
from torch import nn

from jepa_arkit.models.direct import DirectCausalModel


class RegionalConditionalVAE(nn.Module):
    """A deterministic speech model with a stochastic regional decoder.

    The E01 encoder is shared. Mouth motion is copied from its deterministic head,
    eyes/brows/gaze/head receive a conditional stochastic residual. Sampling therefore
    cannot perturb lip sync by construction, while the prior mean can preserve E01.
    """

    def __init__(
        self,
        audio_dim: int,
        motion_dim: int,
        stochastic_indices: tuple[int, ...],
        *,
        residual_indices: tuple[int, ...] = (),
        model_dim: int = 224,
        deterministic_layers: int = 4,
        heads: int = 8,
        latent_dim: int = 32,
        regional_dim: int = 128,
        dropout: float = 0.1,
        max_frames: int = 120,
    ) -> None:
        super().__init__()
        if not stochastic_indices or len(set(stochastic_indices)) != len(stochastic_indices):
            raise ValueError("stochastic_indices must be non-empty and unique")
        if min(stochastic_indices) < 0 or max(stochastic_indices) >= motion_dim:
            raise ValueError("stochastic index is outside motion_dim")
        if not set(residual_indices).issubset(stochastic_indices):
            raise ValueError("residual_indices must be a subset of stochastic_indices")
        self.audio_dim = audio_dim
        self.motion_dim = motion_dim
        self.stochastic_indices = stochastic_indices
        self.residual_indices = tuple(residual_indices)
        self.deterministic_indices = tuple(
            index for index in range(motion_dim) if index not in set(stochastic_indices)
        )
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.deterministic = DirectCausalModel(
            audio_dim,
            motion_dim,
            model_dim=model_dim,
            layers=deterministic_layers,
            heads=heads,
            dropout=dropout,
            max_frames=max_frames,
        )
        self.prior = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, regional_dim),
            nn.GELU(),
            nn.Linear(regional_dim, 2 * latent_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(model_dim + len(stochastic_indices), regional_dim),
            nn.GELU(),
            nn.Linear(regional_dim, 2 * latent_dim),
        )
        self.stochastic_projection = nn.Sequential(
            nn.Linear(model_dim + latent_dim, regional_dim),
            nn.GELU(),
        )
        self.stochastic_decoder = nn.GRU(
            regional_dim,
            regional_dim,
            num_layers=1,
            batch_first=True,
        )
        self.stochastic_head = nn.Linear(regional_dim, len(stochastic_indices))
        nn.init.normal_(self.stochastic_head.weight, std=0.01)
        nn.init.zeros_(self.stochastic_head.bias)

    def train(self, mode: bool = True) -> RegionalConditionalVAE:
        """Keep a frozen E01 branch in eval mode while training E03 heads."""
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.deterministic.parameters()):
            self.deterministic.eval()
        return self

    @staticmethod
    def _split_parameters(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = values.chunk(2, dim=-1)
        return mean, log_variance.clamp(-6.0, 2.0)

    def _prior(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_parameters(self.prior(hidden.mean(dim=1)))

    def _posterior(
        self,
        hidden: torch.Tensor,
        target_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat((hidden, target_residual), dim=-1).mean(dim=1)
        return self._split_parameters(self.posterior(values))

    def _decode_residual(
        self,
        hidden: torch.Tensor,
        latent_delta: torch.Tensor,
    ) -> torch.Tensor:
        expanded = latent_delta.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        values = self.stochastic_projection(torch.cat((hidden, expanded), dim=-1))
        values, _ = self.stochastic_decoder(values)
        return self.stochastic_head(values)

    @staticmethod
    def _sample(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        generator: torch.Generator | None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
        elif noise.shape != mean.shape:
            raise ValueError("noise must have shape [B, latent_dim]")
        return mean + noise * torch.exp(0.5 * log_variance)

    def forward(
        self,
        audio: torch.Tensor,
        target_motion: torch.Tensor | None = None,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        regional_temperature: float = 1.0,
        head_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if audio.ndim != 3 or audio.shape[-1] != self.audio_dim:
            raise ValueError("audio must have shape [B, T, audio_dim]")
        if audio.shape[1] > self.max_frames:
            raise ValueError("sequence exceeds max_frames")
        if target_motion is not None and target_motion.shape != (
            audio.shape[0],
            audio.shape[1],
            self.motion_dim,
        ):
            raise ValueError("target_motion must have shape [B, T, motion_dim]")
        deterministic_hidden = self.deterministic.encode(audio)
        deterministic = self.deterministic.curve_head(deterministic_hidden)
        prior_mean, prior_log_variance = self._prior(deterministic_hidden)
        if target_motion is None:
            latent_mean, latent_log_variance = prior_mean, prior_log_variance
        else:
            target_residual = target_motion[..., list(self.stochastic_indices)] - deterministic[
                ..., list(self.stochastic_indices)
            ]
            latent_mean, latent_log_variance = self._posterior(
                deterministic_hidden, target_residual
            )
        latent = (
            self._sample(latent_mean, latent_log_variance, generator, noise)
            if sample
            else latent_mean
        )
        latent_delta = latent - prior_mean
        stochastic = self._decode_residual(deterministic_hidden, latent_delta)
        neutral = self._decode_residual(deterministic_hidden, torch.zeros_like(latent_delta))
        stochastic = stochastic - neutral
        temperatures = stochastic.new_full(
            (len(self.stochastic_indices),), float(regional_temperature)
        )
        for index in self.residual_indices:
            temperatures[self.stochastic_indices.index(index)] = float(head_temperature)
        stochastic = stochastic * temperatures
        prediction = deterministic.clone()
        prediction[..., list(self.stochastic_indices)] += stochastic
        return {
            "prediction": prediction,
            "deterministic": deterministic,
            "stochastic": stochastic,
            "posterior_mean": latent_mean,
            "posterior_log_variance": latent_log_variance,
            "prior_mean": prior_mean,
            "prior_log_variance": prior_log_variance,
            "latent": latent,
        }


def gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_variance: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, averaged over batch and dimensions."""
    variance_ratio = torch.exp(posterior_log_variance - prior_log_variance)
    mean_distance = (posterior_mean - prior_mean).square() * torch.exp(-prior_log_variance)
    element = 0.5 * (
        prior_log_variance - posterior_log_variance + variance_ratio + mean_distance - 1.0
    )
    if free_bits > 0:
        element = element.clamp_min(free_bits)
    return element.mean()
