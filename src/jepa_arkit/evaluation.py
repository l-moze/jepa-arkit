from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_arkit.contracts.rights import Track
from jepa_arkit.failure_analysis.report import build_failure_report
from jepa_arkit.io import dump_json, load_json, load_yaml
from jepa_arkit.models.direct import DirectCausalModel, shifted_history
from jepa_arkit.training.checkpoint import load_checkpoint
from jepa_arkit.training.dataset import MotionWindowDataset, Window


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


def _physical_values(
    normalized: torch.Tensor,
    normalization: dict[str, object],
) -> torch.Tensor:
    mean = torch.as_tensor(normalization["mean"], device=normalized.device, dtype=normalized.dtype)
    standard = torch.as_tensor(
        normalization["standard_deviation"], device=normalized.device, dtype=normalized.dtype
    )
    return normalized * standard + mean


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    dimension_weights: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
) -> dict[str, tuple[float, float]]:
    error = (prediction - target).abs()
    weighted = confidence.unsqueeze(-1) * dimension_weights.unsqueeze(1)
    valid = confidence.unsqueeze(-1)

    def group_mae(indices: tuple[int, ...]) -> tuple[float, float]:
        selected = error[..., list(indices)]
        selected_dimension_weights = dimension_weights[..., list(indices)].unsqueeze(1)
        selected_valid = valid * selected_dimension_weights
        return float((selected * selected_valid).sum()), float(selected_valid.sum())

    curve_indices = tuple(range(52))
    head_rotation = tuple(range(52, 56))
    head_translation = tuple(range(56, 59))
    return {
        "mae": (float((error * weighted).sum()), float(weighted.sum())),
        "curve_mae": group_mae(curve_indices),
        "mouth_mae": group_mae(groups["mouth_jaw"]),
        "eyes_brows_mae": group_mae(groups["eyes_brows"]),
        "gaze_mae": group_mae(groups["gaze"]),
        "nose_cheek_mae": group_mae(groups["nose_cheek"]),
        "head_quaternion_mae": group_mae(head_rotation),
        "head_translation_cm_mae": group_mae(head_translation),
    }


def _per_sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    dimension_weights: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
) -> list[dict[str, float]]:
    error = (prediction - target).abs()
    weights = confidence.unsqueeze(-1) * dimension_weights.unsqueeze(1)

    def value(sample_error: torch.Tensor, sample_weights: torch.Tensor) -> float:
        return float((sample_error * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8))

    output: list[dict[str, float]] = []
    for index in range(prediction.shape[0]):
        sample_error = error[index]
        sample_weights = weights[index]
        output.append(
            {
                "mae": value(sample_error, sample_weights),
                "curve_mae": value(sample_error[..., :52], sample_weights[..., :52]),
                "mouth_mae": value(
                    sample_error[..., list(groups["mouth_jaw"])],
                    sample_weights[..., list(groups["mouth_jaw"])],
                ),
                "eyes_brows_mae": value(
                    sample_error[..., list(groups["eyes_brows"])],
                    sample_weights[..., list(groups["eyes_brows"])],
                ),
                "gaze_mae": value(
                    sample_error[..., list(groups["gaze"])],
                    sample_weights[..., list(groups["gaze"])],
                ),
                "head_quaternion_mae": value(
                    sample_error[..., 52:56], sample_weights[..., 52:56]
                ),
                "head_translation_cm_mae": value(
                    sample_error[..., 56:59], sample_weights[..., 56:59]
                ),
            }
        )
    return output


@torch.no_grad()
def evaluate_direct(
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
    paths = config["paths"]
    dataset_args = {
        "manifest_path": _resolve(config_path, paths["manifest"]),
        "audit_report_path": _resolve(config_path, paths["audit_report"]),
        "schema_path": _resolve(config_path, paths["canonical_schema"]),
        "feature_store_path": _resolve(config_path, paths["feature_store"]),
        "normalization_path": _resolve(config_path, paths["normalization"]),
        "frames": int(config["training"]["frames"]),
        "track": Track(str(config["track"])),
        "allowed_gates": ("D0B", "D0P") if config.get("pilot_non_comparable") else ("D0B",),
    }
    dataset = MotionWindowDataset(split=split, **dataset_args)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
    )
    model = DirectCausalModel(
        audio_dim=dataset.feature_store.metadata.feature_dim,
        motion_dim=dataset.schema.motion_dim,
        **dict(config["model"]),
    ).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        requested_track=Track(str(config["track"])),
        map_location=device,
        restore_rng=False,
    )
    model.eval()
    normalization = load_json(_resolve(config_path, paths["normalization"]))
    groups = dataset.schema.group_indices()
    history_mode = str(config["training"].get("history_mode", "none"))
    aggregates: dict[str, list[dict[str, tuple[float, float]]]] = {
        "normal": [],
        "audio_shuffle": [],
        "audio_reverse": [],
        "audio_shift_500ms": [],
        "silence": [],
    }
    record_by_clip = {record.clip_id: record for record in dataset.records}
    per_clip_rows: list[dict[str, object]] = []
    for batch in loader:
        audio = batch["audio"].to(device)
        target_normalized = batch["motion"].to(device)
        target = _physical_values(target_normalized, normalization)
        confidence = batch["confidence"].to(device)
        dimension_weights = batch["dimension_weights"].to(device)
        history = shifted_history(target_normalized) if history_mode == "shifted" else None
        variants = {
            "normal": audio,
            "audio_shuffle": audio.flip(0),
            "audio_reverse": audio.flip(1),
            "audio_shift_500ms": torch.roll(audio, shifts=15, dims=1),
            "silence": torch.zeros_like(audio),
        }
        for name, variant in variants.items():
            predicted_normalized = model(variant, history)
            predicted = _physical_values(predicted_normalized, normalization)
            aggregates[name].append(
                _metrics(predicted, target, confidence, dimension_weights, groups)
            )
            if name == "normal":
                for clip_id, sample_metrics in zip(
                    batch["clip_ids"],
                    _per_sample_metrics(
                        predicted, target, confidence, dimension_weights, groups
                    ),
                    strict=True,
                ):
                    record = record_by_clip[str(clip_id)]
                    per_clip_rows.append(
                        {
                            "clip_id": str(clip_id),
                            "face_identity_id": record.face_identity_id,
                            "speaker_id": record.speaker_id,
                            "source_id": record.source_id,
                            "language": record.language,
                            "recording_condition": record.recording_condition,
                            "split": record.split,
                            "tracking_confidence": record.quality.tracking_confidence,
                            "av_sync_confidence": record.quality.av_sync_confidence,
                            **sample_metrics,
                        }
                    )
    summary = {}
    for name, values in aggregates.items():
        summary[name] = {}
        for metric in values[0]:
            numerator = sum(item[metric][0] for item in values)
            denominator = sum(item[metric][1] for item in values)
            summary[name][metric] = numerator / max(denominator, 1e-8)
    normal = summary["normal"]
    report: dict[str, object] = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_parameter_hash": checkpoint.get("extra", {}).get("parameter_hash"),
        "history_mode": history_mode,
        "split": split,
        "validation_windows": len(dataset),
        "metrics": summary,
        "audio_shuffle_ratio": summary["audio_shuffle"]["mouth_mae"]
        / max(normal["mouth_mae"], 1e-8),
        "audio_reverse_ratio": summary["audio_reverse"]["mouth_mae"]
        / max(normal["mouth_mae"], 1e-8),
        "audio_shift_500ms_ratio": summary["audio_shift_500ms"]["mouth_mae"]
        / max(normal["mouth_mae"], 1e-8),
        "silence_ratio": summary["silence"]["mouth_mae"] / max(normal["mouth_mae"], 1e-8),
        "per_clip_rows": len(per_clip_rows),
    }
    output = _resolve(config_path, paths["output_dir"])
    dump_json(output / "evaluation.json", report)
    failure_report = build_failure_report(
        per_clip_rows,
        output / f"failure_report_{split}.html",
        primary_metric="mouth_mae",
        largest_is_worst=True,
        limit=10,
    )
    report["failure_report"] = failure_report
    by_identity: dict[str, dict[str, float | int]] = {}
    for row in per_clip_rows:
        identity = str(row["face_identity_id"])
        aggregate = by_identity.setdefault(identity, {"clips": 0, "mouth_mae_sum": 0.0})
        aggregate["clips"] += 1
        aggregate["mouth_mae_sum"] += float(row["mouth_mae"])
    for aggregate in by_identity.values():
        aggregate["mouth_mae"] = float(aggregate["mouth_mae_sum"] / aggregate["clips"])
        del aggregate["mouth_mae_sum"]
    report["by_identity"] = by_identity
    dump_json(output / f"per_clip_metrics_{split}.json", {"rows": per_clip_rows})
    dump_json(output / "evaluation.json", report)
    return report
