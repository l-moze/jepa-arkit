from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_arkit.contracts.rights import Track
from jepa_arkit.diagnostics.latent import latent_statistics
from jepa_arkit.io import dump_json, load_json, load_yaml
from jepa_arkit.models.jepa import (
    AudioMotionJEPA,
    jepa_loss,
    make_causal_future_mask,
    make_span_mask,
)
from jepa_arkit.training.checkpoint import save_checkpoint
from jepa_arkit.training.dataset import MotionWindowDataset, Window
from jepa_arkit.training.environment import environment_report
from jepa_arkit.training.reproducibility import set_determinism, state_dict_hash
from jepa_arkit.training.sampler import IdentityBalancedBatchSampler


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _collate(windows: list[Window]) -> dict[str, object]:
    return {
        "audio": torch.stack([window.audio for window in windows]),
        "motion": torch.stack([window.motion for window in windows]),
        "confidence": torch.stack([window.confidence for window in windows]),
        "dimension_weights": torch.stack([window.dimension_weights for window in windows]),
        "clip_ids": [window.clip_id for window in windows],
    }


def _shifted(motion: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(motion[:, :1]), motion[:, :-1]), dim=1)


def _context(motion: torch.Tensor, training: dict[str, object], mode: str) -> torch.Tensor:
    if str(training.get("history_mode", "shifted")) == "none":
        return torch.zeros_like(motion)
    return _shifted(motion) if mode == "causal_future" else motion


def _make_mask(
    mode: str,
    batch: int,
    frames: int,
    groups: int,
    training: dict[str, object],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if mode == "random_span":
        mask = make_span_mask(
            batch,
            frames,
            groups,
            float(training["mask_ratio"]),
            generator,
            device,
            int(training.get("minimum_span", 3)),
            int(training.get("maximum_span", 15)),
        )
    elif mode == "causal_future":
        mask = make_causal_future_mask(
            batch,
            frames,
            groups,
            int(training["horizon_frames"]),
            device,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return mask


def _loss(
    outputs: dict[str, torch.Tensor],
    motion: torch.Tensor,
    confidence: torch.Tensor,
    weights: torch.Tensor,
    training: dict[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return jepa_loss(
        outputs,
        motion,
        decode_weight=float(training.get("decode_weight", 0.5)),
        variance_weight=float(training.get("variance_weight", 0.1)),
        covariance_weight=float(training.get("covariance_weight", 0.01)),
        confidence=confidence,
        dimension_weights=weights,
    )


@torch.no_grad()
def _validate(
    model: AudioMotionJEPA,
    loader: DataLoader,
    *,
    device: torch.device,
    training: dict[str, object],
    normalization: dict[str, object],
    mouth_indices: tuple[int, ...],
    seed: int,
) -> dict[str, dict[str, float]]:
    was_training = model.training
    model.eval()
    mean = torch.as_tensor(normalization["mean"], device=device)
    standard = torch.as_tensor(normalization["standard_deviation"], device=device)
    results: dict[str, dict[str, float]] = {}
    for mode_index, mode in enumerate(("random_span", "causal_future")):
        generator = torch.Generator(device="cpu").manual_seed(seed + mode_index)
        totals = {name: 0.0 for name in ("total", "latent", "decode", "variance", "covariance")}
        curve_sum = curve_weight = mouth_sum = mouth_weight = 0.0
        latent_values: list[torch.Tensor] = []
        sampled = batches = 0
        for data in loader:
            motion = data["motion"].to(device)
            audio = data["audio"].to(device)
            confidence = data["confidence"].to(device)
            weights = data["dimension_weights"].to(device).mean(dim=0)
            context = _context(motion, training, mode)
            mask = _make_mask(
                mode,
                motion.shape[0],
                motion.shape[1],
                len(model.tokenizer.group_names),
                training,
                generator,
                device,
            ) & (confidence > 0).unsqueeze(-1)
            if not torch.any(mask):
                continue
            outputs = model(context, mask, audio, target_curves=motion)
            loss, parts = _loss(outputs, motion, confidence, weights, training)
            totals["total"] += float(loss)
            for name in ("latent", "decode", "variance", "covariance"):
                totals[name] += float(parts[name])
            prediction = outputs["curves"] * standard + mean
            target = motion * standard + mean
            error = (prediction - target).abs()
            all_weights = confidence.unsqueeze(-1) * weights.view(1, 1, -1)
            curve_weights = all_weights[..., :52]
            curve_sum += float((error[..., :52] * curve_weights).sum())
            curve_weight += float(curve_weights.sum())
            mouth_weights = all_weights[..., list(mouth_indices)]
            mouth_sum += float((error[..., list(mouth_indices)] * mouth_weights).sum())
            mouth_weight += float(mouth_weights.sum())
            valid = (confidence > 0).unsqueeze(-1).expand(
                -1, -1, len(model.tokenizer.group_names)
            ).reshape(motion.shape[0], -1)
            values = outputs["z_pred"][valid].detach().float().cpu()
            latent_values.append(values[: max(0, 32_768 - sampled)])
            sampled += len(values)
            batches += 1
        stats = latent_statistics(torch.cat(latent_values))
        results[mode] = {
            **{name: value / max(batches, 1) for name, value in totals.items()},
            "physical_curve_mae": curve_sum / max(curve_weight, 1e-8),
            "physical_mouth_mae": mouth_sum / max(mouth_weight, 1e-8),
            **stats,
        }
    model.train(was_training)
    return results


def train_audio_motion_jepa(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    seed = int(config["seed"])
    set_determinism(seed)
    track = Track(str(config["track"]))
    pilot = bool(config.get("pilot_non_comparable", False))
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    paths = config["paths"]
    training = config["training"]
    arguments = {
        "manifest_path": _resolve(config_path, paths["manifest"]),
        "audit_report_path": _resolve(config_path, paths["audit_report"]),
        "schema_path": _resolve(config_path, paths["canonical_schema"]),
        "feature_store_path": _resolve(config_path, paths["feature_store"]),
        "frames": int(training["frames"]),
        "track": track,
        "allowed_gates": ("D0B", "D0P") if pilot else ("D0B",),
        "normalization_path": _resolve(config_path, paths["normalization"]),
    }
    train_dataset = MotionWindowDataset(split="train", **arguments)
    validation_dataset = MotionWindowDataset(split="validation", **arguments)
    model = AudioMotionJEPA(
        train_dataset.schema.model_group_indices(),
        train_dataset.schema.motion_dim,
        audio_dim=train_dataset.feature_store.metadata.feature_dim,
        **dict(config["model"]),
    ).to(device)
    pretrained_path = paths.get("motion_jepa_checkpoint")
    loaded_pretrain = False
    if pretrained_path:
        checkpoint = torch.load(
            _resolve(config_path, str(pretrained_path)), map_location=device, weights_only=False
        )
        if Track(str(checkpoint["track"])) is not track:
            raise RuntimeError("Motion-JEPA checkpoint track does not match Audio-JEPA track")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = {"audio_projection.0.weight", "audio_projection.0.bias"}
        allowed_missing |= {"audio_projection.1.weight", "audio_projection.1.bias"}
        allowed_missing |= {"audio_projection.3.weight", "audio_projection.3.bias"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                "Unexpected Motion-JEPA checkpoint keys: "
                f"missing={missing}, unexpected={unexpected}"
            )
        loaded_pretrain = True
        del checkpoint
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
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
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    output = _resolve(config_path, paths["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "training_trace.jsonl"
    trace.unlink(missing_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    normalization = load_json(_resolve(config_path, paths["normalization"]))
    mouth_indices = train_dataset.schema.group_indices()["mouth_jaw"]
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 29)
    maximum_steps = int(training["steps"])
    validation_interval = int(training.get("validation_interval", maximum_steps))
    patience = int(training.get("patience_evaluations", 0))
    best_causal = float("inf")
    best_step = 0
    stale = 0
    step = epoch = 0
    started = time.perf_counter()
    stopped_early = False
    last_validation: dict[str, dict[str, float]] | None = None
    while step < maximum_steps:
        sampler.set_epoch(epoch)
        for data in loader:
            step += 1
            motion = data["motion"].to(device, non_blocking=True)
            audio = data["audio"].to(device, non_blocking=True)
            confidence = data["confidence"].to(device, non_blocking=True)
            weights = data["dimension_weights"].to(device).mean(dim=0)
            mode = "causal_future" if step % 2 else "random_span"
            context = _context(motion, training, mode)
            mask = _make_mask(
                mode,
                motion.shape[0],
                motion.shape[1],
                len(model.tokenizer.group_names),
                training,
                mask_generator,
                device,
            ) & (confidence > 0).unsqueeze(-1)
            if not torch.any(mask):
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(context, mask, audio, target_curves=motion)
                loss, parts = _loss(outputs, motion, confidence, weights, training)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Audio-JEPA loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            progress = step / maximum_steps
            momentum = float(training.get("ema_momentum_start", 0.99)) + progress * (
                float(training.get("ema_momentum_end", 0.999))
                - float(training.get("ema_momentum_start", 0.99))
            )
            model.update_target(momentum)
            event: dict[str, object] = {
                "step": step,
                "epoch": epoch,
                "mask_mode": mode,
                "loss": float(loss.detach()),
                **{f"{name}_loss": float(value) for name, value in parts.items()},
                "gradient_norm": float(gradient_norm),
                "ema_momentum": momentum,
                "clip_ids": data["clip_ids"],
            }
            with trace.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
            if step % validation_interval == 0 or step >= maximum_steps:
                last_validation = _validate(
                    model,
                    validation_loader,
                    device=device,
                    training=training,
                    normalization=normalization,
                    mouth_indices=mouth_indices,
                    seed=seed + step,
                )
                causal = last_validation["causal_future"]["total"]
                if causal < best_causal:
                    best_causal = causal
                    best_step = step
                    stale = 0
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
                            "pilot_non_comparable": pilot,
                            "motion_jepa_pretrained": loaded_pretrain,
                            "schema_fingerprint": train_dataset.schema.fingerprint,
                            "parameter_hash": state_dict_hash(model.state_dict()),
                            "validation": last_validation,
                        },
                    )
                else:
                    stale += 1
                event["validation"] = last_validation
                with trace.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
                if patience and stale >= patience:
                    stopped_early = True
                    break
            if step >= maximum_steps:
                break
        epoch += 1
        if stopped_early:
            break
    elapsed = time.perf_counter() - started
    if last_validation is None:
        raise RuntimeError("Audio-JEPA training completed without validation")
    report: dict[str, object] = {
        "status": "pilot_non_comparable" if pilot else "completed",
        "pilot_non_comparable": pilot,
        "motion_jepa_pretrained": loaded_pretrain,
        "history_mode": str(training.get("history_mode", "shifted")),
        "steps": step,
        "best_step": best_step,
        "best_causal_validation_loss": best_causal,
        "last_validation": last_validation,
        "stopped_early": stopped_early,
        "updates_per_second": step / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "parameter_hash": state_dict_hash(model.state_dict()),
        "feature_metadata": asdict(train_dataset.feature_store.metadata)
        | {"track": train_dataset.feature_store.metadata.track.value},
        "environment": environment_report(),
    }
    dump_json(output / "metrics.json", report)
    dump_json(
        output / "resource_report.json",
        {
            "elapsed_seconds": elapsed,
            "updates_per_second": step / elapsed,
            "peak_memory_bytes": report["peak_memory_bytes"],
            "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
        },
    )
    return report


@torch.no_grad()
def evaluate_audio_motion_jepa(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "validation",
) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    track = Track(str(config["track"]))
    paths = config["paths"]
    training = config["training"]
    arguments = {
        "manifest_path": _resolve(config_path, paths["manifest"]),
        "audit_report_path": _resolve(config_path, paths["audit_report"]),
        "schema_path": _resolve(config_path, paths["canonical_schema"]),
        "feature_store_path": _resolve(config_path, paths["feature_store"]),
        "frames": int(training["frames"]),
        "track": track,
        "allowed_gates": ("D0B", "D0P") if config.get("pilot_non_comparable") else ("D0B",),
        "normalization_path": _resolve(config_path, paths["normalization"]),
    }
    dataset = MotionWindowDataset(split=split, **arguments)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
    )
    model = AudioMotionJEPA(
        dataset.schema.model_group_indices(),
        dataset.schema.motion_dim,
        audio_dim=dataset.feature_store.metadata.feature_dim,
        **dict(config["model"]),
    ).to(device)
    checkpoint = torch.load(
        Path(checkpoint_path).resolve(), map_location=device, weights_only=False
    )
    if Track(str(checkpoint["track"])) is not track:
        raise RuntimeError("Checkpoint track does not match evaluation track")
    model.load_state_dict(checkpoint["model"])
    report = _validate(
        model,
        loader,
        device=device,
        training=training,
        normalization=load_json(_resolve(config_path, paths["normalization"])),
        mouth_indices=dataset.schema.group_indices()["mouth_jaw"],
        seed=int(config["seed"]) + int(checkpoint["step"]),
    )
    result = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
        "split": split,
        "windows": len(dataset),
        "checkpoint_step": int(checkpoint["step"]),
        "motion_jepa_pretrained": bool(
            checkpoint.get("extra", {}).get("motion_jepa_pretrained", False)
        ),
        "history_mode": str(training.get("history_mode", "shifted")),
        "metrics": report,
    }
    output = _resolve(config_path, paths["output_dir"])
    dump_json(output / f"evaluation_{split}.json", result)
    return result
