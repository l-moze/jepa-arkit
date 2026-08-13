from pathlib import Path

import pytest
import torch

from jepa_arkit.contracts.rights import Track
from jepa_arkit.errors import TrackViolation
from jepa_arkit.models.direct import DirectCausalModel
from jepa_arkit.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = DirectCausalModel(4, 6, model_dim=16, layers=1, heads=4, max_frames=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    audio = torch.randn(2, 5, 4)
    loss = model(audio).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
        track=Track.RESEARCH,
        data_release_id="synthetic",
        feature_release_id="synthetic",
        config={"seed": 3},
    )
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    restored = DirectCausalModel(4, 6, model_dim=16, layers=1, heads=4, max_frames=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    result = load_checkpoint(
        checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        requested_track=Track.RESEARCH,
    )
    assert result["step"] == 1
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_product_cannot_load_research_checkpoint(tmp_path: Path) -> None:
    model = DirectCausalModel(4, 6, model_dim=16, layers=1, heads=4, max_frames=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "research.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=0,
        track=Track.RESEARCH,
        data_release_id="research",
        feature_release_id="research",
        config={},
    )
    with pytest.raises(TrackViolation):
        load_checkpoint(
            path,
            model=model,
            optimizer=None,
            requested_track=Track.PRODUCT,
        )

