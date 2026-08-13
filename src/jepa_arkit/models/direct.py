from __future__ import annotations

import torch
from torch import nn


class DirectCausalModel(nn.Module):
    """Small causal baseline over precomputed audio features and optional motion history."""

    def __init__(
        self,
        audio_dim: int,
        motion_dim: int,
        model_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        max_frames: int = 512,
    ) -> None:
        super().__init__()
        self.audio_dim = audio_dim
        self.motion_dim = motion_dim
        self.input_projection = nn.Linear(audio_dim + motion_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, max_frames, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.curve_head = nn.Linear(model_dim, motion_dim)
        nn.init.normal_(self.position, std=0.01)

    def forward(
        self,
        audio_features: torch.Tensor,
        motion_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.curve_head(self.encode(audio_features, motion_history))

    def encode(
        self,
        audio_features: torch.Tensor,
        motion_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio_features.ndim != 3 or audio_features.shape[-1] != self.audio_dim:
            raise ValueError("audio_features must have shape [B, T, audio_dim]")
        batch, frames, _ = audio_features.shape
        if frames > self.position.shape[1]:
            raise ValueError("sequence exceeds max_frames")
        if motion_history is None:
            motion_history = torch.zeros(
                batch,
                frames,
                self.motion_dim,
                device=audio_features.device,
                dtype=audio_features.dtype,
            )
        if motion_history.shape != (batch, frames, self.motion_dim):
            raise ValueError("motion_history must have shape [B, T, motion_dim]")
        hidden = self.input_projection(torch.cat((audio_features, motion_history), dim=-1))
        hidden = hidden + self.position[:, :frames]
        causal_mask = torch.triu(
            torch.ones(frames, frames, device=hidden.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.encoder(hidden, mask=causal_mask, is_causal=True)
        return self.norm(hidden)


def shifted_history(motion: torch.Tensor) -> torch.Tensor:
    if motion.ndim != 3:
        raise ValueError("motion must have shape [B, T, K]")
    neutral = torch.zeros_like(motion[:, :1])
    return torch.cat((neutral, motion[:, :-1]), dim=1)
