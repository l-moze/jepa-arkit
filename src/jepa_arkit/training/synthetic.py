from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticBatch:
    audio: torch.Tensor
    motion: torch.Tensor
    confidence: torch.Tensor


def make_synthetic_batch(
    *,
    batch_size: int,
    frames: int,
    audio_dim: int,
    motion_dim: int,
    seed: int,
    device: torch.device,
) -> SyntheticBatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    audio = torch.randn(batch_size, frames, audio_dim, generator=generator)
    projection = torch.randn(audio_dim, motion_dim, generator=generator) / audio_dim**0.5
    motion = torch.sigmoid(audio @ projection)
    if frames > 1:
        motion[:, 1:] = 0.8 * motion[:, 1:] + 0.2 * motion[:, :-1]
    confidence = torch.rand(batch_size, frames, generator=generator) * 0.2 + 0.8
    return SyntheticBatch(audio.to(device), motion.to(device), confidence.to(device))

