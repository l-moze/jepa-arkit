from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_arkit.contracts.rights import Track
from jepa_arkit.io import dump_json, load_json, load_yaml
from jepa_arkit.losses import acceleration_loss, confidence_weighted_huber, velocity_loss
from jepa_arkit.models.disentangled import RegionalConditionalVAE, gaussian_kl
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


def _arguments(config_path: Path, config: dict[str, object], split: str) -> dict[str, object]:
    paths = config["paths"]
    training = config["training"]
    return {
        "manifest_path": _resolve(config_path, paths["manifest"]),
        "audit_report_path": _resolve(config_path, paths["audit_report"]),
        "schema_path": _resolve(config_path, paths["canonical_schema"]),
        "feature_store_path": _resolve(config_path, paths["feature_store"]),
        "normalization_path": _resolve(config_path, paths["normalization"]),
        "split": split,
        "frames": int(training["frames"]),
        "track": Track(str(config["track"])),
        "allowed_gates": ("D0B", "D0P") if config.get("pilot_non_comparable") else ("D0B",),
    }


def _indices(schema: object) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    groups = schema.model_group_indices()
    stochastic = groups["eyes_brows"] + groups["gaze"] + groups["head"]
    residual = groups["head"]
    mouth = groups["mouth_jaw"]
    return mouth, stochastic, residual


def _loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    confidence: torch.Tensor,
    dimension_weights: torch.Tensor,
    mouth: tuple[int, ...],
    stochastic: tuple[int, ...],
    residual: tuple[int, ...],
    training: dict[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output["prediction"]
    dimension_mean = dimension_weights.mean(dim=0)
    mouth_weights = dimension_mean[list(mouth)]
    stochastic_weights = dimension_mean[list(stochastic)]
    mouth_loss = confidence_weighted_huber(
        output["deterministic"][..., list(mouth)],
        target[..., list(mouth)],
        confidence,
        mouth_weights,
    )
    regional_loss = confidence_weighted_huber(
        prediction[..., list(stochastic)],
        target[..., list(stochastic)],
        confidence,
        stochastic_weights,
    )
    residual_loss = confidence_weighted_huber(
        prediction[..., list(residual)],
        target[..., list(residual)],
        confidence,
        dimension_mean[list(residual)],
    )
    kl = gaussian_kl(
        output["posterior_mean"],
        output["posterior_log_variance"],
        output["prior_mean"],
        output["prior_log_variance"],
        free_bits=float(training.get("kl_free_bits", 0.0)),
    )
    velocity = velocity_loss(prediction, target, dimension_mean, confidence)
    acceleration = acceleration_loss(prediction, target, dimension_mean, confidence)
    total = (
        float(training.get("mouth_weight", 1.0)) * mouth_loss
        + float(training.get("regional_weight", 0.8)) * regional_loss
        + float(training.get("head_residual_weight", 0.5)) * residual_loss
        + float(training.get("kl_weight", 0.01)) * kl
        + float(training.get("velocity_weight", 0.2)) * velocity
        + float(training.get("acceleration_weight", 0.05)) * acceleration
    )
    return total, {
        "mouth": mouth_loss.detach(),
        "regional": regional_loss.detach(),
        "head_residual": residual_loss.detach(),
        "kl": kl.detach(),
        "velocity": velocity.detach(),
        "acceleration": acceleration.detach(),
    }


@torch.no_grad()
def _validate(
    model: RegionalConditionalVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: dict[str, object],
    mouth: tuple[int, ...],
    stochastic: tuple[int, ...],
    eyes_brows: tuple[int, ...],
    gaze: tuple[int, ...],
    head_quaternion: tuple[int, ...],
    head_translation: tuple[int, ...],
    training: dict[str, object],
    seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    mean = torch.as_tensor(normalization["mean"], device=device)
    standard = torch.as_tensor(normalization["standard_deviation"], device=device)
    group_indices = {
        "mouth": mouth,
        "regional": stochastic,
        "eyes_brows": eyes_brows,
        "gaze": gaze,
        "head_quaternion": head_quaternion,
        "head_translation": head_translation,
    }
    totals = {"loss": 0.0, **{name: 0.0 for name in group_indices}}
    weights = {name: 0.0 for name in group_indices}
    variances: list[torch.Tensor] = []
    mouth_sample_differences: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    batches = 0
    for data in loader:
        audio = data["audio"].to(device)
        target = data["motion"].to(device)
        confidence = data["confidence"].to(device)
        dimensions = data["dimension_weights"].to(device)
        posterior_output = model(audio, target, sample=False)
        loss, _ = _loss(
            posterior_output,
            target,
            confidence,
            dimensions,
            mouth,
            stochastic,
            model.residual_indices,
            training,
        )
        deployed_output = model(audio, sample=False)
        physical_prediction = deployed_output["prediction"] * standard + mean
        physical_target = target * standard + mean
        for name, indices in group_indices.items():
            scale = confidence.unsqueeze(-1) * dimensions[:, list(indices)].unsqueeze(1)
            error = (
                physical_prediction[..., list(indices)] - physical_target[..., list(indices)]
            ).abs()
            totals[name] += float((error * scale).sum())
            weights[name] += float(scale.sum())
        totals["loss"] += float(loss)
        draws = [model(audio, sample=True, generator=generator)["prediction"] for _ in range(3)]
        physical_draws = [draw * standard + mean for draw in draws]
        draw_stack = torch.stack(physical_draws).float()
        variances.append(draw_stack.var(dim=0).mean(dim=(0, 1)).cpu())
        mouth_sample_differences.append(
            (draw_stack[..., list(mouth)] - draw_stack[0:1, ..., list(mouth)])
            .abs()
            .amax()
            .cpu()
        )
        batches += 1
    model.train(was_training)
    variance = torch.stack(variances).mean(dim=0)
    non_head = [index for index in stochastic if index not in model.residual_indices]
    metrics = {
        "loss": totals["loss"] / max(batches, 1),
        **{
            f"{name}_mae": totals[name] / max(weights[name], 1e-8)
            for name in group_indices
        },
        "stochastic_variance_mean": float(variance[list(stochastic)].mean()),
        "head_variance_mean": float(variance[list(model.residual_indices)].mean()),
        "eyes_brows_gaze_variance_mean": float(variance[non_head].mean()),
        "mouth_sample_max_abs_difference": float(torch.stack(mouth_sample_differences).max()),
    }
    return metrics


def train_regional(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    seed = int(config["seed"])
    set_determinism(seed)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_dataset = MotionWindowDataset(**_arguments(config_path, config, "train"))
    validation_dataset = MotionWindowDataset(**_arguments(config_path, config, "validation"))
    mouth, stochastic, residual = _indices(train_dataset.schema)
    schema_groups = train_dataset.schema.model_group_indices()
    model = RegionalConditionalVAE(
        train_dataset.feature_store.metadata.feature_dim,
        train_dataset.schema.motion_dim,
        stochastic,
        residual_indices=residual,
        **dict(config["model"]),
    ).to(device)
    deterministic_checkpoint = config["paths"].get("deterministic_checkpoint")
    if deterministic_checkpoint:
        checkpoint = torch.load(
            _resolve(config_path, deterministic_checkpoint), map_location=device, weights_only=False
        )
        model.deterministic.load_state_dict(checkpoint["model"])
        for parameter in model.deterministic.parameters():
            parameter.requires_grad_(False)
    e01_parameters = sum(parameter.numel() for parameter in model.deterministic.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_ratio = total_parameters / max(e01_parameters, 1)
    if parameter_ratio > float(config.get("maximum_parameter_ratio", 1.2)):
        raise RuntimeError(
            f"E03 parameter ratio {parameter_ratio:.3f} exceeds the configured budget"
        )
    training = config["training"]
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
    loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=_collate)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
    )
    output = _resolve(config_path, config["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "training_trace.jsonl"
    trace.unlink(missing_ok=True)
    normalization = load_json(_resolve(config_path, config["paths"]["normalization"]))
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    maximum_steps = int(training["steps"])
    validation_interval = int(training.get("validation_interval", maximum_steps))
    patience = int(training.get("patience_evaluations", 0))
    best_loss = float("inf")
    best_step = step = epoch = stale = 0
    stopped_early = False
    last_validation: dict[str, float] | None = None
    started = time.perf_counter()
    while step < maximum_steps:
        sampler.set_epoch(epoch)
        for data in loader:
            step += 1
            audio = data["audio"].to(device, non_blocking=True)
            motion = data["motion"].to(device, non_blocking=True)
            confidence = data["confidence"].to(device, non_blocking=True)
            dimensions = data["dimension_weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                values = model(audio, motion, sample=True)
                loss, parts = _loss(
                    values, motion, confidence, dimensions, mouth, stochastic, residual, training
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite regional loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            event: dict[str, object] = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach()),
                **{f"{name}_loss": float(value) for name, value in parts.items()},
                "gradient_norm": float(gradient_norm),
                "clip_ids": data["clip_ids"],
            }
            with trace.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
            if step % validation_interval == 0 or step >= maximum_steps:
                last_validation = _validate(
                    model,
                    validation_loader,
                    device=device,
                    normalization=normalization,
                    mouth=mouth,
                    stochastic=stochastic,
                    eyes_brows=schema_groups["eyes_brows"],
                    gaze=schema_groups["gaze"],
                    head_quaternion=schema_groups["head"][:4],
                    head_translation=schema_groups["head"][4:],
                    training=training,
                    seed=seed + step,
                )
                event["validation"] = last_validation
                with trace.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
                if last_validation["loss"] < best_loss:
                    best_loss, best_step, stale = last_validation["loss"], step, 0
                    save_checkpoint(
                        output / "best_checkpoint.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        track=Track(str(config["track"])),
                        data_release_id=train_dataset.feature_store.metadata.source_data_release_id,
                        feature_release_id=train_dataset.feature_store.metadata.feature_release_id,
                        config=config,
                        extra={
                            "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
                            "deterministic_checkpoint": deterministic_checkpoint,
                            "stochastic_indices": stochastic,
                            "residual_indices": residual,
                            "parameter_hash": state_dict_hash(model.state_dict()),
                            "e01_parameters": e01_parameters,
                            "total_parameters": total_parameters,
                            "parameter_ratio": parameter_ratio,
                            "validation": last_validation,
                        },
                    )
                else:
                    stale += 1
                if patience and stale >= patience:
                    stopped_early = True
                    break
            if step >= maximum_steps:
                break
        epoch += 1
        if stopped_early:
            break
    if last_validation is None:
        raise RuntimeError("regional training completed without validation")
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
        "steps": step,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "last_validation": last_validation,
        "stopped_early": stopped_early,
        "deterministic_checkpoint": deterministic_checkpoint,
        "e01_parameters": e01_parameters,
        "total_parameters": total_parameters,
        "parameter_ratio": parameter_ratio,
        "stochastic_indices": stochastic,
        "residual_indices": residual,
        "parameter_hash": state_dict_hash(model.state_dict()),
        "updates_per_second": step / max(elapsed, 1e-9),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "feature_metadata": asdict(train_dataset.feature_store.metadata)
        | {"track": train_dataset.feature_store.metadata.track.value},
        "environment": environment_report(),
    }
    dump_json(output / "metrics.json", report)
    dump_json(
        output / "resource_report.json",
        {
            "elapsed_seconds": elapsed,
            "updates_per_second": report["updates_per_second"],
            "peak_memory_bytes": report["peak_memory_bytes"],
            "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
        },
    )
    return report


@torch.no_grad()
def evaluate_regional(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "validation",
) -> dict[str, object]:
    """Evaluate E03 with a pure-audio prior and report stochastic diagnostics."""
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dataset = MotionWindowDataset(**_arguments(config_path, config, split))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
    )
    mouth, stochastic, residual = _indices(dataset.schema)
    schema_groups = dataset.schema.model_group_indices()
    model = RegionalConditionalVAE(
        dataset.feature_store.metadata.feature_dim,
        dataset.schema.motion_dim,
        stochastic,
        residual_indices=residual,
        **dict(config["model"]),
    ).to(device)
    checkpoint = torch.load(
        Path(checkpoint_path).resolve(), map_location=device, weights_only=False
    )
    if Track(str(checkpoint["track"])) is not Track(str(config["track"])):
        raise RuntimeError("Checkpoint track does not match regional evaluation track")
    model.load_state_dict(checkpoint["model"])
    metrics = _validate(
        model,
        loader,
        device=device,
        normalization=load_json(_resolve(config_path, config["paths"]["normalization"])),
        mouth=mouth,
        stochastic=stochastic,
        eyes_brows=schema_groups["eyes_brows"],
        gaze=schema_groups["gaze"],
        head_quaternion=schema_groups["head"][:4],
        head_translation=schema_groups["head"][4:],
        training=config["training"],
        seed=int(config["seed"]) + int(checkpoint["step"]),
    )
    result = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
        "split": split,
        "windows": len(dataset),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_parameter_hash": checkpoint.get("extra", {}).get("parameter_hash"),
        "deterministic_checkpoint": checkpoint.get("extra", {}).get("deterministic_checkpoint"),
        "parameter_ratio": checkpoint.get("extra", {}).get("parameter_ratio"),
        "metrics": metrics,
    }
    output = _resolve(config_path, config["paths"]["output_dir"])
    dump_json(output / f"evaluation_{split}.json", result)
    return result


def summarize_regional_runs(
    baseline_evaluation_path: str | Path,
    run_directories: list[str | Path],
    output_path: str | Path,
) -> dict[str, object]:
    """Aggregate E03 seeds without selecting a checkpoint on test performance."""
    if len(run_directories) < 3:
        raise ValueError("Regional summary requires at least three seed runs")
    baseline = load_json(baseline_evaluation_path)
    baseline_normal = baseline["metrics"]["normal"]
    rows: list[dict[str, object]] = []
    for directory_value in run_directories:
        directory = Path(directory_value).resolve()
        training_report = load_json(directory / "metrics.json")
        test_report = load_json(directory / "evaluation_test.json")
        validation = training_report["last_validation"]
        test = test_report["metrics"]
        rows.append(
            {
                "run": directory.name,
                "checkpoint": str(directory / "best_checkpoint.pt"),
                "best_step": training_report["best_step"],
                "parameter_ratio": training_report["parameter_ratio"],
                "validation_stochastic_variance": validation["stochastic_variance_mean"],
                "validation_head_variance": validation["head_variance_mean"],
                "validation_non_head_variance": validation[
                    "eyes_brows_gaze_variance_mean"
                ],
                "test_stochastic_variance": test["stochastic_variance_mean"],
                "test_head_variance": test["head_variance_mean"],
                "test_non_head_variance": test["eyes_brows_gaze_variance_mean"],
                "test_mouth_mae": test["mouth_mae"],
                "test_eyes_brows_mae": test["eyes_brows_mae"],
                "test_gaze_mae": test["gaze_mae"],
                "test_head_quaternion_mae": test["head_quaternion_mae"],
                "test_head_translation_mae": test["head_translation_mae"],
                "mouth_sample_max_abs_difference": test[
                    "mouth_sample_max_abs_difference"
                ],
            }
        )
    validation_variances = [float(row["validation_stochastic_variance"]) for row in rows]
    target = statistics.median(validation_variances)
    eligible = [
        row
        for row in rows
        if float(row["parameter_ratio"]) <= 1.2
        and float(row["mouth_sample_max_abs_difference"]) == 0.0
    ]
    if not eligible:
        raise RuntimeError("No regional seed satisfies the parameter and mouth invariance gates")
    selected = min(
        eligible,
        key=lambda row: abs(float(row["validation_stochastic_variance"]) - target),
    )

    def aggregate(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in rows]
        return {
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    exact_baseline = all(
        abs(float(row["test_mouth_mae"]) - float(baseline_normal["mouth_mae"])) < 1e-12
        and abs(float(row["test_eyes_brows_mae"]) - float(baseline_normal["eyes_brows_mae"]))
        < 1e-12
        and abs(float(row["test_gaze_mae"]) - float(baseline_normal["gaze_mae"])) < 1e-12
        and abs(
            float(row["test_head_quaternion_mae"])
            - float(baseline_normal["head_quaternion_mae"])
        )
        < 1e-12
        and abs(
            float(row["test_head_translation_mae"])
            - float(baseline_normal["head_translation_cm_mae"])
        )
        < 1e-12
        for row in rows
    )
    report: dict[str, object] = {
        "status": "pilot_non_comparable",
        "gate_decision": "architecture_passed_perceptual_diversity_pending",
        "selection_policy": "closest_to_median_validation_stochastic_variance",
        "selection_uses_test_metrics": False,
        "selected_run": selected["run"],
        "selected_checkpoint": selected["checkpoint"],
        "baseline_checkpoint_step": baseline["checkpoint_step"],
        "baseline_mean_exactly_preserved_across_seeds": exact_baseline,
        "all_mouth_samples_invariant": all(
            float(row["mouth_sample_max_abs_difference"]) == 0.0 for row in rows
        ),
        "all_parameter_ratios_within_1_2": all(
            float(row["parameter_ratio"]) <= 1.2 for row in rows
        ),
        "seeds": rows,
        "aggregate": {
            "validation_stochastic_variance": aggregate(
                "validation_stochastic_variance"
            ),
            "validation_head_variance": aggregate("validation_head_variance"),
            "validation_non_head_variance": aggregate("validation_non_head_variance"),
            "test_stochastic_variance": aggregate("test_stochastic_variance"),
            "test_head_variance": aggregate("test_head_variance"),
            "test_non_head_variance": aggregate("test_non_head_variance"),
        },
        "limitations": [
            "RAVDESS supervision is Silver and research-only.",
            "Non-zero variance is not evidence of perceptual quality or natural diversity.",
            "E03 remains an offline optional style layer until UE and perceptual validation.",
        ],
    }
    dump_json(output_path, report)
    return report
