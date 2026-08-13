from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def _sinusoidal_time_encoding(
    frames: int, model_dim: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(frames, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, model_dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / model_dim)
    )
    encoding = torch.zeros(frames, model_dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if model_dim > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype).view(1, frames, 1, model_dim)


class SemanticMotionTokenizer(nn.Module):
    """Stride-1 semantic tokenizer: one token per group and frame."""

    def __init__(self, group_indices: dict[str, tuple[int, ...]], model_dim: int) -> None:
        super().__init__()
        self.group_names = tuple(group_indices)
        self.group_indices = group_indices
        self.projections = nn.ModuleDict(
            {name: nn.Linear(len(indices), model_dim) for name, indices in group_indices.items()}
        )
        self.group_embedding = nn.Parameter(torch.zeros(len(self.group_names), model_dim))
        nn.init.normal_(self.group_embedding, std=0.02)

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        if curves.ndim != 3:
            raise ValueError("curves must have shape [B, T, K]")
        tokens = []
        for group_index, name in enumerate(self.group_names):
            indices = torch.as_tensor(self.group_indices[name], device=curves.device)
            group_values = torch.index_select(curves, dim=-1, index=indices)
            token = self.projections[name](group_values) + self.group_embedding[group_index]
            tokens.append(token)
        return torch.stack(tokens, dim=2)


class MotionDecoder(nn.Module):
    def __init__(self, group_indices: dict[str, tuple[int, ...]], model_dim: int, motion_dim: int):
        super().__init__()
        self.group_names = tuple(group_indices)
        self.group_indices = group_indices
        self.heads = nn.ModuleDict(
            {name: nn.Linear(model_dim, len(indices)) for name, indices in group_indices.items()}
        )
        self.motion_dim = motion_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[2] != len(self.group_names):
            raise ValueError("tokens must have shape [B, T, G, D]")
        output = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            self.motion_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for group_index, name in enumerate(self.group_names):
            values = self.heads[name](tokens[:, :, group_index])
            output[:, :, self.group_indices[name]] = values.to(dtype=output.dtype)
        return output


class MotionJEPA(nn.Module):
    def __init__(
        self,
        group_indices: dict[str, tuple[int, ...]],
        motion_dim: int,
        model_dim: int = 128,
        encoder_layers: int = 3,
        predictor_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = SemanticMotionTokenizer(group_indices, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            4 * model_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.target_tokenizer = deepcopy(self.tokenizer)
        self.target_encoder = deepcopy(self.context_encoder)
        for parameter in self.target_tokenizer.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        predictor_layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            4 * model_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, predictor_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = MotionDecoder(group_indices, model_dim, motion_dim)

    @staticmethod
    def _flatten(tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, groups, dim = tokens.shape
        return tokens.reshape(batch, frames * groups, dim)

    @staticmethod
    def _unflatten(tokens: torch.Tensor, frames: int, groups: int) -> torch.Tensor:
        return tokens.reshape(tokens.shape[0], frames, groups, tokens.shape[-1])

    def forward(
        self,
        curves: torch.Tensor,
        mask: torch.Tensor,
        target_curves: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_tokens = self.tokenizer(curves)
        batch, frames, groups, _ = context_tokens.shape
        if mask.shape != (batch, frames, groups) or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B, T, G]")
        time_encoding = _sinusoidal_time_encoding(
            frames,
            context_tokens.shape[-1],
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
        context_tokens = context_tokens + time_encoding
        flat_context = self._flatten(context_tokens)
        flat_mask = mask.reshape(batch, frames * groups)
        masked_identity = self.mask_token + time_encoding + self.tokenizer.group_embedding.view(
            1, 1, groups, -1
        )
        masked = torch.where(
            flat_mask.unsqueeze(-1),
            self._flatten(masked_identity.expand(batch, -1, -1, -1)),
            flat_context,
        )
        z_context = self.context_encoder(masked)
        z_pred = self.predictor(z_context)
        with torch.no_grad():
            target_input = curves if target_curves is None else target_curves
            target_tokens = self.target_tokenizer(target_input) + time_encoding
            z_target = self.target_encoder(self._flatten(target_tokens))
        decoded = self.decoder(self._unflatten(z_pred, frames, groups))
        return {
            "z_context": z_context,
            "z_target": z_target,
            "z_pred": z_pred,
            "curves": decoded,
            "flat_mask": flat_mask,
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        online = list(self.tokenizer.parameters()) + list(self.context_encoder.parameters())
        target = list(self.target_tokenizer.parameters()) + list(self.target_encoder.parameters())
        if len(online) != len(target):
            raise RuntimeError("EMA model parameter mismatch")
        for online_parameter, target_parameter in zip(online, target, strict=True):
            target_parameter.mul_(momentum).add_(online_parameter, alpha=1 - momentum)


class AudioMotionJEPA(MotionJEPA):
    """Audio-conditioned JEPA using the same motion latent space and decoder."""

    def __init__(self, *args: object, audio_dim: int, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.audio_dim = audio_dim
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(
                audio_dim,
                self.tokenizer.projections[self.tokenizer.group_names[0]].out_features,
            ),
            nn.GELU(),
            nn.Linear(
                self.tokenizer.projections[self.tokenizer.group_names[0]].out_features,
                self.tokenizer.projections[self.tokenizer.group_names[0]].out_features,
            ),
        )

    def forward(
        self,
        curves: torch.Tensor,
        mask: torch.Tensor,
        audio: torch.Tensor,
        target_curves: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            audio.ndim != 3
            or audio.shape[:2] != curves.shape[:2]
            or audio.shape[-1] != self.audio_dim
        ):
            raise ValueError("audio must have shape [B, T, audio_dim] aligned to curves")
        context_tokens = self.tokenizer(curves)
        batch, frames, groups, _ = context_tokens.shape
        if mask.shape != (batch, frames, groups) or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B, T, G]")
        audio_tokens = self.audio_projection(audio).unsqueeze(2)
        time_encoding = _sinusoidal_time_encoding(
            frames,
            context_tokens.shape[-1],
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
        context_tokens = context_tokens + audio_tokens + time_encoding
        flat_context = self._flatten(context_tokens)
        flat_mask = mask.reshape(batch, frames * groups)
        masked_identity = (
            self.mask_token
            + audio_tokens
            + time_encoding
            + self.tokenizer.group_embedding.view(1, 1, groups, -1)
        )
        masked = torch.where(
            flat_mask.unsqueeze(-1),
            self._flatten(masked_identity.expand(batch, -1, -1, -1)),
            flat_context,
        )
        z_context = self.context_encoder(masked)
        z_pred = self.predictor(z_context)
        with torch.no_grad():
            target_input = curves if target_curves is None else target_curves
            target_tokens = self.target_tokenizer(target_input) + time_encoding
            z_target = self.target_encoder(self._flatten(target_tokens))
        decoded = self.decoder(self._unflatten(z_pred, frames, groups))
        return {
            "z_context": z_context,
            "z_target": z_target,
            "z_pred": z_pred,
            "curves": decoded,
            "flat_mask": flat_mask,
        }


def jepa_loss(
    outputs: dict[str, torch.Tensor],
    target_curves: torch.Tensor,
    decode_weight: float = 0.5,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
    confidence: torch.Tensor | None = None,
    dimension_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = outputs["flat_mask"]
    if not torch.any(mask):
        raise ValueError("JEPA mask must hide at least one token")
    valid_tokens = torch.ones_like(mask, dtype=torch.bool)
    if confidence is not None:
        groups = mask.shape[1] // confidence.shape[1]
        valid_tokens = (
            (confidence > 0).unsqueeze(-1).expand(-1, -1, groups).reshape_as(valid_tokens)
        )
    latent_mask = mask & valid_tokens
    if not torch.any(latent_mask):
        raise ValueError("JEPA mask must hide at least one valid token")
    latent = F.smooth_l1_loss(
        outputs["z_pred"][latent_mask], outputs["z_target"][latent_mask]
    )
    decoded_elements = F.smooth_l1_loss(outputs["curves"], target_curves, reduction="none")
    decoded_weights = torch.ones_like(decoded_elements)
    if confidence is not None:
        decoded_weights = decoded_weights * confidence.unsqueeze(-1)
    if dimension_weights is not None:
        decoded_weights = decoded_weights * dimension_weights.view(1, 1, -1)
    decoded = (decoded_elements * decoded_weights).sum() / decoded_weights.sum().clamp_min(1e-8)
    flat = outputs["z_pred"][valid_tokens]
    centered = flat - flat.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    covariance_matrix = centered.T @ centered / max(1, centered.shape[0] - 1)
    off_diagonal = covariance_matrix - torch.diag(torch.diag(covariance_matrix))
    covariance = off_diagonal.square().sum() / covariance_matrix.shape[0]
    total = (
        latent
        + decode_weight * decoded
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return total, {
        "latent": latent.detach(),
        "decode": decoded.detach(),
        "variance": variance.detach(),
        "covariance": covariance.detach(),
    }


def make_span_mask(
    batch: int,
    frames: int,
    groups: int,
    ratio: float,
    generator: torch.Generator,
    device: torch.device,
    minimum_span: int = 3,
    maximum_span: int = 15,
) -> torch.Tensor:
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    if not 1 <= minimum_span <= maximum_span:
        raise ValueError("span lengths must satisfy 1 <= minimum_span <= maximum_span")
    target = max(1, round(frames * groups * ratio))
    mask = torch.zeros(batch, frames, groups, dtype=torch.bool)
    for batch_index in range(batch):
        while int(mask[batch_index].sum()) < target:
            group = int(torch.randint(groups, (), generator=generator))
            length = int(
                torch.randint(minimum_span, maximum_span + 1, (), generator=generator)
            )
            length = min(length, frames)
            start = int(torch.randint(frames - length + 1, (), generator=generator))
            mask[batch_index, start : start + length, group] = True
    return mask.to(device)


def make_causal_future_mask(
    batch: int,
    frames: int,
    groups: int,
    horizon_frames: int,
    device: torch.device,
) -> torch.Tensor:
    if horizon_frames <= 0 or horizon_frames >= frames:
        raise ValueError("horizon_frames must be in [1, frames - 1]")
    mask = torch.zeros(batch, frames, groups, dtype=torch.bool, device=device)
    mask[:, frames - horizon_frames :] = True
    return mask
