from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.contracts.rights import Track
from jepa_arkit.diagnostics.latent import latent_statistics
from jepa_arkit.io import dump_json, load_yaml
from jepa_arkit.losses import confidence_weighted_huber, velocity_loss
from jepa_arkit.models.direct import DirectCausalModel, shifted_history
from jepa_arkit.models.jepa import (
    MotionJEPA,
    jepa_loss,
    make_causal_future_mask,
    make_span_mask,
)
from jepa_arkit.training.checkpoint import save_checkpoint
from jepa_arkit.training.reproducibility import set_determinism, state_dict_hash
from jepa_arkit.training.synthetic import make_synthetic_batch


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _append_trace(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def run_direct_smoke(config_path: str | Path) -> dict[str, object]:
    config = load_yaml(config_path)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "training_trace.jsonl"
    trace_path.unlink(missing_ok=True)
    seed = int(config["seed"])
    set_determinism(seed)
    device = _resolve_device(str(config.get("device", "cpu")))
    model_config = config["model"]
    train_config = config["training"]
    model = DirectCausalModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["learning_rate"]))
    batch = make_synthetic_batch(
        batch_size=int(train_config["batch_size"]),
        frames=int(train_config["frames"]),
        audio_dim=int(model_config["audio_dim"]),
        motion_dim=int(model_config["motion_dim"]),
        seed=seed + 1,
        device=device,
    )
    history = shifted_history(batch.motion)
    initial_hash = state_dict_hash(model.state_dict())
    start = time.perf_counter()
    initial_loss = None
    final_loss = None
    for step in range(1, int(train_config["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch.audio, history)
        curve = confidence_weighted_huber(prediction, batch.motion, batch.confidence)
        velocity = velocity_loss(prediction, batch.motion)
        loss = curve + float(train_config.get("velocity_weight", 0.5)) * velocity
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        if initial_loss is None:
            initial_loss = value
        final_loss = value
        _append_trace(
            trace_path,
            {
                "step": step,
                "loss": value,
                "curve_loss": float(curve.detach()),
                "velocity_loss": float(velocity.detach()),
                "gradient_norm": float(gradient_norm),
            },
        )
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        normal = confidence_weighted_huber(model(batch.audio, history), batch.motion)
        shuffled = confidence_weighted_huber(model(batch.audio.flip(0), history), batch.motion)
        silent = confidence_weighted_huber(
            model(torch.zeros_like(batch.audio), history), batch.motion
        )
    report: dict[str, object] = {
        "kind": "synthetic_non_comparable",
        "device": str(device),
        "seed": seed,
        "steps": int(train_config["steps"]),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": (initial_loss - final_loss) / initial_loss,
        "audio_shuffle_ratio": float(shuffled / normal.clamp_min(1e-12)),
        "silence_ratio": float(silent / normal.clamp_min(1e-12)),
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": state_dict_hash(model.state_dict()),
        "updates_per_second": int(train_config["steps"]) / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    report["infrastructure_passed"] = bool(
        report["loss_reduction"] >= float(train_config.get("min_loss_reduction", 0.5))
        and report["audio_shuffle_ratio"] >= float(train_config.get("min_shuffle_ratio", 1.1))
        and report["silence_ratio"] >= float(train_config.get("min_silence_ratio", 1.1))
    )
    report["passed"] = report["infrastructure_passed"]
    report["status"] = "passed" if report["passed"] else "blocked"
    dump_json(output / "metrics.json", report)
    save_checkpoint(
        output / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=int(train_config["steps"]),
        track=Track(str(config["track"])),
        data_release_id=str(config["data_release_id"]),
        feature_release_id=str(config["feature_release_id"]),
        config=config,
        extra={"non_comparable": True},
    )
    return report


def run_jepa_smoke(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "training_trace.jsonl"
    trace_path.unlink(missing_ok=True)
    schema_path = Path(config["canonical_schema"])
    if not schema_path.is_absolute():
        schema_path = (config_path.parent / schema_path).resolve()
    schema = CanonicalSchema.from_file(schema_path)
    seed = int(config["seed"])
    set_determinism(seed)
    device = _resolve_device(str(config.get("device", "cpu")))
    model_config = dict(config["model"])
    model = MotionJEPA(
        group_indices=schema.model_group_indices(),
        motion_dim=schema.motion_dim,
        **model_config,
    ).to(device)
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_config["learning_rate"]),
    )
    batch = make_synthetic_batch(
        batch_size=int(train_config["batch_size"]),
        frames=int(train_config["frames"]),
        audio_dim=16,
        motion_dim=schema.motion_dim,
        seed=seed + 1,
        device=device,
    )
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    initial_loss = None
    final_loss = None
    start = time.perf_counter()
    for step in range(1, int(train_config["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        mask_mode = str(train_config.get("mask_mode", "random_token"))
        if mask_mode == "random_token":
            mask = make_span_mask(
                batch=batch.motion.shape[0],
                frames=batch.motion.shape[1],
                groups=len(schema.model_group_indices()),
                ratio=float(train_config["mask_ratio"]),
                generator=mask_generator,
                device=device,
            )
        elif mask_mode == "causal_future":
            mask = make_causal_future_mask(
                batch=batch.motion.shape[0],
                frames=batch.motion.shape[1],
                groups=len(schema.model_group_indices()),
                horizon_frames=int(train_config["horizon_frames"]),
                device=device,
            )
        else:
            raise ValueError(f"Unknown mask mode: {mask_mode}")
        outputs = model(batch.motion, mask)
        loss, components = jepa_loss(outputs, batch.motion)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite JEPA loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target(float(train_config["ema_momentum"]))
        value = float(loss.detach())
        if initial_loss is None:
            initial_loss = value
        final_loss = value
        if step == 1 or step == int(train_config["steps"]):
            stats = latent_statistics(outputs["z_pred"])
        else:
            stats = {}
        _append_trace(
            trace_path,
            {
                "step": step,
                "loss": value,
                "latent_loss": float(components["latent"]),
                "decode_loss": float(components["decode"]),
                "variance_loss": float(components["variance"]),
                "covariance_loss": float(components["covariance"]),
                "gradient_norm": float(gradient_norm),
                **stats,
            },
        )
    elapsed = time.perf_counter() - start
    final_stats = latent_statistics(outputs["z_pred"])
    report: dict[str, object] = {
        "kind": "synthetic_non_comparable",
        "device": str(device),
        "seed": seed,
        "mask_mode": str(train_config.get("mask_mode", "random_token")),
        "steps": int(train_config["steps"]),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": (initial_loss - final_loss) / initial_loss,
        "updates_per_second": int(train_config["steps"]) / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "parameter_hash": state_dict_hash(model.state_dict()),
        **final_stats,
    }
    report["infrastructure_passed"] = bool(
        report["loss_reduction"] >= float(train_config.get("min_loss_reduction", 0.5))
        and report["low_variance_fraction"] < 1.0
    )
    report["representation_passed"] = bool(
        report["effective_rank_ratio"]
        >= float(train_config.get("min_effective_rank_ratio", 0.25))
        and report["low_variance_fraction"]
        <= float(train_config.get("max_low_variance_fraction", 0.1))
    )
    report["passed"] = report["infrastructure_passed"]
    report["status"] = (
        "passed"
        if report["representation_passed"]
        else "infrastructure_passed_representation_blocked"
    )
    dump_json(output / "metrics.json", report)
    save_checkpoint(
        output / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=int(train_config["steps"]),
        track=Track(str(config["track"])),
        data_release_id=str(config["data_release_id"]),
        feature_release_id="motion_only",
        config=config,
        extra={"non_comparable": True, "schema_fingerprint": schema.fingerprint},
    )
    return report
