from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from jepa_arkit.contracts.rights import Track, assert_track_compatible
from jepa_arkit.training.reproducibility import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    track: Track,
    data_release_id: str,
    feature_release_id: str,
    config: dict[str, Any],
    ancestor_tracks: list[Track] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "track": track.value,
            "ancestor_tracks": [item.value for item in (ancestor_tracks or [])],
            "data_release_id": data_release_id,
            "feature_release_id": feature_release_id,
            "config": config,
            "rng_state": capture_rng_state(),
            "extra": extra or {},
        },
        target,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    requested_track: Track,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    artifact_track = Track(checkpoint["track"])
    ancestors = [artifact_track, *(Track(value) for value in checkpoint["ancestor_tracks"])]
    assert_track_compatible(requested_track, [], ancestors)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint

