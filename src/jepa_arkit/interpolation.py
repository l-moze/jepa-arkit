from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.errors import ContractError
from jepa_arkit.io import dump_json, file_hash


@dataclass(frozen=True)
class ResampledMotion:
    """Canonical motion resampled onto a higher-rate playback timeline."""

    curves: np.ndarray
    head_quaternion: np.ndarray
    head_translation: np.ndarray
    timestamps: np.ndarray
    source_fps: int
    target_fps: int


def _normalise_quaternion(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ContractError("quaternion cannot be normalized")
    return value / norm


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    first = _normalise_quaternion(left.astype(np.float64, copy=False))
    second = _normalise_quaternion(right.astype(np.float64, copy=False))
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalise_quaternion(first + fraction * (second - first)).astype(np.float32)
    angle = float(np.arccos(dot))
    sine = float(np.sin(angle))
    weights = (
        np.sin((1.0 - fraction) * angle) / sine,
        np.sin(fraction * angle) / sine,
    )
    return _normalise_quaternion(weights[0] * first + weights[1] * second).astype(np.float32)


def _validate_inputs(
    curves: np.ndarray,
    head_quaternion: np.ndarray,
    head_translation: np.ndarray,
    timestamps: np.ndarray,
    source_fps: int,
    target_fps: int,
) -> int:
    if source_fps <= 0 or target_fps <= 0 or target_fps < source_fps:
        raise ContractError("fps values must be positive and target_fps >= source_fps")
    if target_fps % source_fps:
        raise ContractError("target_fps must be an integer multiple of source_fps")
    if curves.ndim != 2 or head_quaternion.shape != (len(curves), 4):
        raise ContractError("motion arrays have incompatible shapes")
    if head_translation.shape != (len(curves), 3) or timestamps.shape != (len(curves),):
        raise ContractError("motion arrays have incompatible shapes")
    if len(curves) == 0:
        raise ContractError("motion must contain at least one frame")
    if not all(
        np.isfinite(values).all()
        for values in (curves, head_quaternion, head_translation, timestamps)
    ):
        raise ContractError("motion contains NaN or Inf")
    if len(curves) > 1:
        expected_step = 1.0 / source_fps
        if not np.allclose(np.diff(timestamps), expected_step, rtol=1e-5, atol=1e-7):
            raise ContractError("source timestamps must be uniformly sampled at source_fps")
    for value in head_quaternion:
        _normalise_quaternion(value)
    return target_fps // source_fps


def interpolate_motion(
    curves: np.ndarray,
    head_quaternion: np.ndarray,
    head_translation: np.ndarray,
    timestamps: np.ndarray,
    *,
    source_fps: int = 30,
    target_fps: int = 60,
) -> ResampledMotion:
    """Resample canonical motion for UE playback.

    Curves and translations are linearly interpolated. Quaternions use shortest-path
    spherical interpolation and are normalized at every output frame. The source
    endpoints are preserved exactly and the output timeline contains no padded frames.
    """
    ratio = _validate_inputs(
        curves, head_quaternion, head_translation, timestamps, source_fps, target_fps
    )
    frame_count = (len(curves) - 1) * ratio + 1
    output_times = timestamps[0] + np.arange(frame_count, dtype=np.float64) / target_fps
    output_curves = np.empty((frame_count, curves.shape[1]), dtype=np.float32)
    output_translation = np.empty((frame_count, 3), dtype=np.float32)
    output_quaternion = np.empty((frame_count, 4), dtype=np.float32)
    for index in range(frame_count):
        source_position = index / ratio
        left_index = min(int(np.floor(source_position)), len(curves) - 1)
        fraction = source_position - left_index
        right_index = min(left_index + 1, len(curves) - 1)
        output_curves[index] = (
            curves[left_index] + fraction * (curves[right_index] - curves[left_index])
        )
        output_translation[index] = (
            head_translation[left_index]
            + fraction * (head_translation[right_index] - head_translation[left_index])
        )
        output_quaternion[index] = _slerp(
            head_quaternion[left_index], head_quaternion[right_index], fraction
        )
    return ResampledMotion(
        curves=output_curves,
        head_quaternion=output_quaternion,
        head_translation=output_translation,
        timestamps=output_times,
        source_fps=source_fps,
        target_fps=target_fps,
    )


def round_trip_metrics(
    source_curves: np.ndarray,
    source_quaternion: np.ndarray,
    source_translation: np.ndarray,
    resampled: ResampledMotion,
) -> dict[str, float | bool]:
    """Measure exact decimation of a 60 fps result back onto the source frames."""
    ratio = resampled.target_fps // resampled.source_fps
    selected = np.arange(0, len(resampled.timestamps), ratio)
    curves_error = float(np.max(np.abs(resampled.curves[selected] - source_curves)))
    translation_error = float(
        np.max(np.abs(resampled.head_translation[selected] - source_translation))
    )
    quaternion_error = float(
        np.max(np.abs(resampled.head_quaternion[selected] - source_quaternion))
    )
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(resampled.head_quaternion, axis=1) - 1.0))
    )
    return {
        "passed": bool(
            curves_error <= 1e-6
            and translation_error <= 1e-6
            and quaternion_error <= 1e-6
            and quaternion_norm_error <= 1e-5
        ),
        "source_frames": len(source_curves),
        "target_frames": len(resampled.curves),
        "curve_max_abs_error": curves_error,
        "translation_max_abs_error": translation_error,
        "quaternion_max_abs_error": quaternion_error,
        "quaternion_norm_max_abs_error": quaternion_norm_error,
    }


def resample_canonical_npz(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    schema_path: str | Path = "configs/contracts/canonical_arkit_v1.json",
    target_fps: int = 60,
) -> dict[str, object]:
    """Create a UE playback NPZ and a machine-readable round-trip report."""
    schema = CanonicalSchema.from_file(schema_path)
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()
    with np.load(input_file, allow_pickle=False) as archive:
        curves = np.asarray(archive["curves"], dtype=np.float32)
        names = [str(value) for value in archive["curve_names"].tolist()]
        quaternion = np.asarray(archive["head_quaternion"], dtype=np.float32)
        translation = np.asarray(archive["head_translation"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        source_fps = int(np.asarray(archive["fps"]).item())
    schema.validate_motion(curves, names, quaternion, translation, timestamps)
    if source_fps != schema.fps:
        raise ContractError("input motion fps does not match canonical schema")
    result = interpolate_motion(
        curves,
        quaternion,
        translation,
        timestamps,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    metrics = round_trip_metrics(curves, quaternion, translation, result)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        curves=result.curves,
        curve_names=np.asarray(schema.curve_names),
        head_quaternion=result.head_quaternion,
        head_translation=result.head_translation,
        timestamps=result.timestamps,
        fps=np.asarray(target_fps),
        source_fps=np.asarray(source_fps),
        sample_rate=np.asarray(16_000),
        source_motion_sha256=np.asarray(file_hash(input_file)),
    )
    report = {
        "status": "passed" if metrics["passed"] else "failed",
        "input": str(input_file),
        "output": str(output_file),
        "schema_id": schema.schema_id,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "interpolation": "linear_curves_translation_slerp_quaternion",
        "ue_resampling_owner": "export_or_animation_blueprint",
        **metrics,
    }
    dump_json(report_path, report)
    return report
