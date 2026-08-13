from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_arkit.contracts.rights import Track
from jepa_arkit.io import dump_json, load_yaml
from jepa_arkit.losses import acceleration_loss, confidence_weighted_huber, velocity_loss
from jepa_arkit.models.direct import DirectCausalModel, shifted_history
from jepa_arkit.training.checkpoint import load_checkpoint, save_checkpoint
from jepa_arkit.training.dataset import MotionWindowDataset, Window
from jepa_arkit.training.environment import environment_report
from jepa_arkit.training.reproducibility import set_determinism, state_dict_hash
from jepa_arkit.training.sampler import IdentityBalancedBatchSampler


def _collate(windows: list[Window]) -> dict[str, object]:
    return {
        "audio": torch.stack([window.audio for window in windows]),
        "motion": torch.stack([window.motion for window in windows]),
        "confidence": torch.stack([window.confidence for window in windows]),
        "dimension_weights": torch.stack([window.dimension_weights for window in windows]),
        "clip_ids": [window.clip_id for window in windows],
    }


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _group_weight_vector(
    dataset: MotionWindowDataset,
    training: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(dataset.schema.motion_dim, device=device)
    configured = training.get("group_loss_weights", {})
    if not isinstance(configured, dict):
        raise ValueError("group_loss_weights must be a mapping")
    for name, multiplier in configured.items():
        indices = dataset.schema.model_group_indices().get(str(name))
        if indices is None:
            raise ValueError(f"Unknown loss-weight group: {name}")
        weights[list(indices)] = float(multiplier)
    return weights


@torch.no_grad()
def _validation_loss(
    model: DirectCausalModel,
    loader: DataLoader,
    device: torch.device,
    history_mode: str,
    group_weights: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch in loader:
        audio = batch["audio"].to(device)
        motion = batch["motion"].to(device)
        confidence = batch["confidence"].to(device)
        dimension_weights = batch["dimension_weights"].to(device).mean(dim=0) * group_weights
        history = shifted_history(motion) if history_mode == "shifted" else None
        prediction = model(audio, history)
        losses.append(
            float(confidence_weighted_huber(prediction, motion, confidence, dimension_weights))
        )
    model.train(was_training)
    return sum(losses) / len(losses)


def train_direct(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    seed = int(config["seed"])
    set_determinism(seed)
    track = Track(str(config["track"]))
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    paths = config["paths"]
    pilot = bool(config.get("pilot_non_comparable", False))
    allowed_gates = ("D0B", "D0P") if pilot else ("D0B",)
    dataset_arguments = {
        "manifest_path": _resolve(config_path, paths["manifest"]),
        "audit_report_path": _resolve(config_path, paths["audit_report"]),
        "schema_path": _resolve(config_path, paths["canonical_schema"]),
        "feature_store_path": _resolve(config_path, paths["feature_store"]),
        "frames": int(config["training"]["frames"]),
        "track": track,
        "allowed_gates": allowed_gates,
        "normalization_path": _resolve(config_path, paths["normalization"])
        if paths.get("normalization")
        else None,
    }
    train_dataset = MotionWindowDataset(split="train", **dataset_arguments)
    validation_dataset = MotionWindowDataset(split="validation", **dataset_arguments)
    feature_dim = train_dataset.feature_store.metadata.feature_dim
    motion_dim = train_dataset.schema.motion_dim
    model_config = dict(config["model"])
    model = DirectCausalModel(
        audio_dim=feature_dim,
        motion_dim=motion_dim,
        **model_config,
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    start_step = 0
    resume = paths.get("resume_checkpoint")
    if resume:
        checkpoint = load_checkpoint(
            _resolve(config_path, resume),
            model=model,
            optimizer=optimizer,
            requested_track=track,
            map_location=device,
        )
        start_step = int(checkpoint["step"])
    sampler = IdentityBalancedBatchSampler(
        train_dataset,
        batch_size=int(training["batch_size"]),
        min_identities=int(training.get("min_identities", 4)),
        max_identity_share=float(training.get("max_identity_share", 0.3)),
        seed=seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=_collate,
        num_workers=int(training.get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )
    group_weights = _group_weight_vector(train_dataset, training, device)
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    output = _resolve(config_path, paths["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "training_trace.jsonl"
    if start_step == 0:
        trace_path.unlink(missing_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step = start_step
    epoch = 0
    max_steps = int(training["steps"])
    validation_interval = int(training.get("validation_interval", max_steps))
    patience_evaluations = int(training.get("patience_evaluations", 0))
    best_validation = float("inf")
    best_step = start_step
    stale_evaluations = 0
    stopped_early = False
    while step < max_steps:
        sampler.set_epoch(epoch)
        emitted = False
        for batch in loader:
            emitted = True
            step += 1
            audio = batch["audio"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            confidence = batch["confidence"].to(device, non_blocking=True)
            history_mode = str(training.get("history_mode", "none"))
            history = shifted_history(motion) if history_mode == "shifted" else None
            dimension_weights = (
                batch["dimension_weights"].to(device, non_blocking=True).mean(dim=0)
                * group_weights
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                prediction = model(audio, history)
                curve = confidence_weighted_huber(
                    prediction, motion, confidence, dimension_weights
                )
                velocity = velocity_loss(prediction, motion, dimension_weights, confidence)
                acceleration = acceleration_loss(
                    prediction, motion, dimension_weights, confidence
                )
                loss = (
                    curve
                    + float(training.get("velocity_weight", 0.5)) * velocity
                    + float(training.get("acceleration_weight", 0.1)) * acceleration
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            event = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "curve_loss": float(curve.detach()),
                "velocity_loss": float(velocity.detach()),
                "acceleration_loss": float(acceleration.detach()),
                "gradient_norm": float(gradient_norm),
                "clip_ids": batch["clip_ids"],
            }
            with trace_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
            if step % validation_interval == 0 or step >= max_steps:
                validation = _validation_loss(
                    model,
                    validation_loader,
                    device,
                    str(training.get("history_mode", "none")),
                    group_weights,
                )
                event["validation_curve_loss"] = validation
                with trace_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
                if validation < best_validation:
                    best_validation = validation
                    best_step = step
                    stale_evaluations = 0
                    save_checkpoint(
                        output / "best_checkpoint.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        track=track,
                        data_release_id=train_dataset.feature_store.metadata.source_data_release_id,
                        feature_release_id=train_dataset.feature_store.metadata.feature_release_id,
                        config=config,
                        extra={
                            "formal": not pilot,
                            "pilot_non_comparable": pilot,
                            "parameter_hash": state_dict_hash(model.state_dict()),
                        },
                    )
                else:
                    stale_evaluations += 1
                if patience_evaluations and stale_evaluations >= patience_evaluations:
                    stopped_early = True
                    break
            if step >= max_steps:
                break
        if not emitted:
            raise RuntimeError("Identity-balanced sampler emitted no batches")
        epoch += 1
        if stopped_early:
            break
    elapsed = time.perf_counter() - started
    final_validation = _validation_loss(
        model,
        validation_loader,
        device,
        str(training.get("history_mode", "none")),
        group_weights,
    )
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    report: dict[str, object] = {
        "status": "pilot_non_comparable" if pilot else "completed",
        "pilot_non_comparable": pilot,
        "track": track.value,
        "steps": step,
        "device": str(device),
        "amp": use_amp,
        "updates_per_second": (step - start_step) / elapsed,
        "peak_memory_bytes": peak_memory,
        "validation_curve_loss": final_validation,
        "best_validation_curve_loss": best_validation,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "parameter_hash": state_dict_hash(model.state_dict()),
        "feature_metadata": asdict(train_dataset.feature_store.metadata)
        | {"track": train_dataset.feature_store.metadata.track.value},
        "history_mode": str(training.get("history_mode", "none")),
        "group_loss_weights": training.get("group_loss_weights", {}),
        "normalization": config.get("paths", {}).get("normalization"),
        "environment": environment_report(),
    }
    dump_json(output / "metrics.json", report)
    dump_json(
        output / "resource_report.json",
        {
            "elapsed_seconds": elapsed,
            "updates_per_second": report["updates_per_second"],
            "peak_memory_bytes": peak_memory,
            "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
        },
    )
    save_checkpoint(
        output / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        track=track,
        data_release_id=train_dataset.feature_store.metadata.source_data_release_id,
        feature_release_id=train_dataset.feature_store.metadata.feature_release_id,
        config=config,
        extra={
            "formal": not pilot,
            "pilot_non_comparable": pilot,
            "parameter_hash": state_dict_hash(model.state_dict()),
        },
    )
    return report
