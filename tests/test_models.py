from pathlib import Path

import torch

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.diagnostics.latent import latent_statistics
from jepa_arkit.models.direct import DirectCausalModel
from jepa_arkit.models.disentangled import RegionalConditionalVAE, gaussian_kl
from jepa_arkit.models.jepa import (
    AudioMotionJEPA,
    MotionJEPA,
    jepa_loss,
    make_causal_future_mask,
    make_span_mask,
)

ROOT = Path(__file__).parents[1]


def test_direct_model_is_causal() -> None:
    torch.manual_seed(1)
    model = DirectCausalModel(8, 12, model_dim=32, layers=1, heads=4, max_frames=16).eval()
    audio = torch.randn(2, 10, 8)
    changed = audio.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:])
    with torch.no_grad():
        original_output = model(audio)
        changed_output = model(changed)
    torch.testing.assert_close(original_output[:, :6], changed_output[:, :6])


def test_motion_jepa_shapes_and_loss() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    model = MotionJEPA(schema.model_group_indices(), schema.motion_dim, 32, 1, 1, 4)
    curves = torch.rand(2, 12, schema.motion_dim)
    generator = torch.Generator().manual_seed(7)
    mask = make_span_mask(2, 12, 5, 0.5, generator, torch.device("cpu"))
    output = model(curves, mask)
    confidence = torch.ones(2, 12)
    confidence[:, -2:] = 0
    dimension_weights = torch.ones(schema.motion_dim)
    dimension_weights[51] = 0
    loss, parts = jepa_loss(
        output,
        curves,
        confidence=confidence,
        dimension_weights=dimension_weights,
    )
    assert output["curves"].shape == curves.shape
    assert loss.isfinite()
    assert parts["latent"].item() > 0
    stats = latent_statistics(output["z_pred"])
    assert 0 <= stats["effective_rank_ratio"] <= 1


def test_motion_jepa_time_encoding_distinguishes_identical_frames() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    model = MotionJEPA(schema.model_group_indices(), schema.motion_dim, 32, 1, 1, 4).eval()
    curves = torch.zeros(1, 6, schema.motion_dim)
    mask = torch.ones(1, 6, len(schema.model_group_indices()), dtype=torch.bool)
    with torch.no_grad():
        output = model(curves, mask)
    assert not torch.allclose(output["z_context"][:, 0], output["z_context"][:, -1])


def test_audio_motion_jepa_accepts_aligned_audio() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    model = AudioMotionJEPA(
        schema.model_group_indices(), schema.motion_dim, 32, 1, 1, 4, audio_dim=8
    )
    curves = torch.rand(2, 8, schema.motion_dim)
    audio = torch.rand(2, 8, 8)
    mask = torch.ones(2, 8, len(schema.model_group_indices()), dtype=torch.bool)
    output = model(curves, mask, audio, target_curves=curves)
    assert output["curves"].shape == curves.shape


def test_causal_future_mask_only_hides_suffix() -> None:
    mask = make_causal_future_mask(2, 12, 5, 4, torch.device("cpu"))
    assert not mask[:, :8].any()
    assert mask[:, 8:].all()


def test_span_mask_uses_contiguous_temporal_regions() -> None:
    generator = torch.Generator().manual_seed(12)
    mask = make_span_mask(2, 30, 5, 0.4, generator, torch.device("cpu"), 4, 8)
    assert mask.shape == (2, 30, 5)
    assert (mask.sum(dim=(1, 2)) >= 60).all()
    assert any(mask[0, :, group].unfold(0, 4, 1).all(dim=1).any() for group in range(5))


def test_regional_vae_never_changes_deterministic_dimensions() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    groups = schema.model_group_indices()
    stochastic = groups["eyes_brows"] + groups["gaze"] + groups["head"]
    model = RegionalConditionalVAE(
        8,
        schema.motion_dim,
        stochastic,
        residual_indices=groups["head"],
        model_dim=32,
        deterministic_layers=1,
        heads=4,
        latent_dim=4,
        regional_dim=16,
        max_frames=12,
        dropout=0.0,
    ).eval()
    audio = torch.randn(2, 12, 8)
    first = model(audio, generator=torch.Generator().manual_seed(1))
    second = model(audio, generator=torch.Generator().manual_seed(2))
    deterministic = list(model.deterministic_indices)
    stochastic_indices = list(model.stochastic_indices)
    torch.testing.assert_close(
        first["prediction"][..., deterministic], second["prediction"][..., deterministic]
    )
    assert not torch.allclose(
        first["prediction"][..., stochastic_indices],
        second["prediction"][..., stochastic_indices],
    )
    mean = model(audio, sample=False)
    torch.testing.assert_close(mean["prediction"], mean["deterministic"])
    noise = torch.randn(2, model.latent_dim)
    fixed_a = model(audio, noise=noise, regional_temperature=0.5, head_temperature=0.25)
    fixed_b = model(audio, noise=noise, regional_temperature=0.5, head_temperature=0.25)
    torch.testing.assert_close(fixed_a["prediction"], fixed_b["prediction"])


def test_regional_vae_keeps_frozen_deterministic_branch_in_eval_mode() -> None:
    model = RegionalConditionalVAE(
        8,
        59,
        tuple(range(20, 59)),
        residual_indices=tuple(range(50, 59)),
        model_dim=32,
        deterministic_layers=1,
        heads=4,
        latent_dim=4,
        regional_dim=16,
        max_frames=12,
        dropout=0.2,
    )
    for parameter in model.deterministic.parameters():
        parameter.requires_grad_(False)
    model.train()
    assert model.training
    assert not model.deterministic.training
    assert model.stochastic_decoder.training


def test_gaussian_kl_is_zero_for_identical_unit_gaussians() -> None:
    mean = torch.zeros(3, 5)
    log_variance = torch.zeros(3, 5)
    assert gaussian_kl(mean, log_variance, mean, log_variance).item() == 0.0
