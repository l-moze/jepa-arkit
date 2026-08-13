from __future__ import annotations

import json
import math
import subprocess
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.data.motion import load_and_validate_motion
from jepa_arkit.errors import ContractError, GateBlocked
from jepa_arkit.io import dump_json, dump_jsonl, dump_jsonl_atomic, file_hash, load_json, load_jsonl
from jepa_arkit.solver import MediaPipeFaceSolver


@dataclass(frozen=True)
class PreparedClip:
    clip_id: str
    status: str
    label_path: str
    audio_path: str
    frames: int
    valid_fraction: float
    duration_seconds: float
    elapsed_seconds: float
    error: str = ""


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != 16_000 or audio.getnchannels() != 1:
            raise ContractError(f"Audio contract mismatch: {path}")
        return audio.getnframes() / audio.getframerate()


def _extract_audio(video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.wav")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ContractError(f"ffmpeg failed for {video}: {completed.stderr.strip()}")
    _audio_duration(temporary)
    temporary.replace(output)


def _prepared_output_is_valid(
    label_path: Path, audio_path: Path, schema_path: Path
) -> PreparedClip | None:
    if not label_path.is_file() or not audio_path.is_file():
        return None
    try:
        schema = CanonicalSchema.from_file(schema_path)
        motion = load_and_validate_motion(label_path, schema)
        duration = _audio_duration(audio_path)
        confidence = motion["frame_confidence"]
        return PreparedClip(
            clip_id=label_path.stem,
            status="cached",
            label_path=str(label_path),
            audio_path=str(audio_path),
            frames=int(len(confidence)),
            valid_fraction=float(confidence.mean()),
            duration_seconds=duration,
            elapsed_seconds=0.0,
        )
    except (ContractError, OSError, ValueError):
        return None


def _prepare_one(
    item: dict[str, object],
    release_root: str,
    output_root: str,
    model_asset: str,
    schema_path: str,
    policy_path: str,
    minimum_valid_fraction: float,
) -> PreparedClip:
    started = time.perf_counter()
    clip_id = str(item["clip_id"])
    actor = str(item["actor_id"]).rsplit("_", 1)[-1]
    output = Path(output_root)
    video = Path(release_root) / "videos" / str(item["video_path"])
    label = output / "motion" / f"Actor_{actor}" / f"{clip_id}.npz"
    audio = output / "audio" / f"Actor_{actor}" / f"{clip_id}.wav"
    try:
        cached = _prepared_output_is_valid(label, audio, Path(schema_path))
        if cached is not None:
            return cached
        _extract_audio(video, audio)
        schema = CanonicalSchema.from_file(schema_path)
        solver = MediaPipeFaceSolver(model_asset=model_asset, schema=schema)
        motion = solver.solve_video(video)
        valid_fraction = float(motion.frame_confidence.mean())
        if valid_fraction < minimum_valid_fraction:
            raise GateBlocked(
                f"valid frame fraction {valid_fraction:.3f} is below "
                f"{minimum_valid_fraction:.3f}"
            )
        solver.save(motion, label, missing_curve_policy=policy_path)
        return PreparedClip(
            clip_id=clip_id,
            status="prepared",
            label_path=str(label),
            audio_path=str(audio),
            frames=len(motion.timestamps),
            valid_fraction=valid_fraction,
            duration_seconds=_audio_duration(audio),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # failure is serialized for resumable batch processing
        return PreparedClip(
            clip_id=clip_id,
            status="failed",
            label_path=str(label),
            audio_path=str(audio),
            frames=0,
            valid_fraction=0.0,
            duration_seconds=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def prepare_ravdess(
    release_root: str | Path,
    output_root: str | Path,
    model_asset: str | Path,
    schema_path: str | Path,
    policy_path: str | Path,
    *,
    workers: int = 4,
    minimum_valid_fraction: float = 0.8,
    limit: int | None = None,
) -> dict[str, object]:
    release = Path(release_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(release / "inventory.jsonl")
    if limit is not None:
        items = items[:limit]
    if not items:
        raise GateBlocked("RAVDESS inventory is empty")
    completed: list[PreparedClip] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _prepare_one,
                item,
                str(release),
                str(output),
                str(Path(model_asset).resolve()),
                str(Path(schema_path).resolve()),
                str(Path(policy_path).resolve()),
                minimum_valid_fraction,
            ): str(item["clip_id"])
            for item in items
        }
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 25 == 0 or len(completed) == len(items):
                completed.sort(key=lambda result: result.clip_id)
                dump_jsonl_atomic(output / "preparation_results.jsonl", map(asdict, completed))
                print(
                    json.dumps(
                        {
                            "completed": len(completed),
                            "total": len(items),
                            "failed": sum(item.status == "failed" for item in completed),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    failed = [item for item in completed if item.status == "failed"]
    report: dict[str, object] = {
        "dataset_id": "ravdess",
        "source_release": "ravdess_v1_0_0",
        "status": "passed" if not failed and len(completed) == len(items) else "failed",
        "requested": len(items),
        "prepared": sum(item.status in {"prepared", "cached"} for item in completed),
        "cached": sum(item.status == "cached" for item in completed),
        "failed": len(failed),
        "minimum_valid_fraction": minimum_valid_fraction,
        "mean_valid_fraction": float(
            np.mean([item.valid_fraction for item in completed if item.status != "failed"])
        ),
        "elapsed_seconds": elapsed,
        "clips_per_second": len(completed) / elapsed,
        "failures": [asdict(item) for item in failed],
    }
    dump_json(output / "preparation_report.json", report)
    return report


def _jitter_score(curves: np.ndarray) -> float:
    if len(curves) < 3:
        return 0.0
    acceleration = np.diff(curves.astype(np.float64), n=2, axis=0)
    return float(np.mean(np.abs(acceleration)))


def _split_for_actor(actor_id: str) -> str:
    actor = int(actor_id.rsplit("_", 1)[-1])
    if actor <= 14:
        return "train"
    if actor <= 19:
        return "validation"
    return "test"


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[:3] *= -1
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _relative_head(quaternion: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse_start = _quaternion_conjugate(quaternion[0])
    relative_rotation = _quaternion_multiply(
        np.broadcast_to(inverse_start, quaternion.shape), quaternion
    )
    relative_rotation /= np.linalg.norm(relative_rotation, axis=1, keepdims=True).clip(min=1e-8)
    for frame in range(1, len(relative_rotation)):
        if float(np.dot(relative_rotation[frame - 1], relative_rotation[frame])) < 0:
            relative_rotation[frame] *= -1
    return relative_rotation.astype(np.float32), (translation - translation[0]).astype(np.float32)


def _write_relative_motion(
    source: Path,
    target: Path,
    schema: CanonicalSchema,
) -> dict[str, np.ndarray]:
    motion = load_and_validate_motion(source, schema)
    rotation, translation = _relative_head(
        motion["head_quaternion"].astype(np.float64),
        motion["head_translation"].astype(np.float64),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        curves=motion["curves"].astype(np.float32),
        curve_names=motion["curve_names"],
        head_quaternion=rotation,
        head_translation=translation,
        frame_confidence=motion["frame_confidence"].astype(np.float32),
        timestamps=motion["timestamps"].astype(np.float64),
        preprocessing_pipeline=np.asarray("relative_head_clip_origin_v1"),
        source_motion_sha256=np.asarray(file_hash(source)),
        degraded_curves=np.asarray(["tongueOut"]),
    )
    return load_and_validate_motion(target, schema)


def build_ravdess_pilot_release(
    raw_release_root: str | Path,
    prepared_root: str | Path,
    output_root: str | Path,
    schema_path: str | Path,
) -> dict[str, object]:
    raw_root = Path(raw_release_root).resolve()
    prepared = Path(prepared_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schema = CanonicalSchema.from_file(schema_path)
    inventory = load_jsonl(raw_root / "inventory.jsonl")
    preparation = load_json(prepared / "preparation_report.json")
    if preparation.get("status") != "passed" or preparation.get("prepared") != len(inventory):
        raise GateBlocked("Full RAVDESS preparation must pass before release construction")

    rows: list[dict[str, object]] = []
    durations: list[float] = []
    train_motion_chunks: list[np.ndarray] = []
    for item in inventory:
        clip_id = str(item["clip_id"])
        actor_id = str(item["actor_id"])
        actor = actor_id.rsplit("_", 1)[-1]
        audio = prepared / "audio" / f"Actor_{actor}" / f"{clip_id}.wav"
        source_motion = prepared / "motion" / f"Actor_{actor}" / f"{clip_id}.npz"
        motion_path = output / "motion" / f"Actor_{actor}" / f"{clip_id}.npz"
        motion = _write_relative_motion(source_motion, motion_path, schema)
        audio_duration = _audio_duration(audio)
        motion_duration = len(motion["timestamps"]) / schema.fps
        duration_error = abs(audio_duration - motion_duration)
        av_sync_proxy = math.exp(-duration_error / 0.1)
        confidence = motion["frame_confidence"].astype(np.float64)
        durations.append(audio_duration)
        split = _split_for_actor(actor_id)
        full_motion = np.concatenate(
            (motion["curves"], motion["head_quaternion"], motion["head_translation"]), axis=-1
        ).astype(np.float64)
        if split == "train":
            train_motion_chunks.append(full_motion)
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": str(audio),
                "motion_path": str(motion_path),
                "source_id": "ravdess_v1_0_0",
                "speaker_id": actor_id,
                "face_identity_id": actor_id,
                "language": str(item["language"]),
                "recording_condition": "controlled_720p_emotional_speech",
                "rights_profile_id": str(item["rights_profile_id"]),
                "withdrawal_key": str(item["withdrawal_key"]),
                "fps": 30,
                "sample_rate": 16000,
                "curve_schema": schema.schema_id,
                "motion_label_version": "mediapipe_face_landmarker_v1_silver",
                "preprocessing_pipeline": "raw30_relative_head_tongueout_zero_weight_v1",
                "split": split,
                "quality": {
                    "face_visibility": float(confidence.mean()),
                    "tracking_confidence": float(confidence.mean()),
                    "av_sync_confidence": float(av_sync_proxy),
                    "motion_jitter_score": _jitter_score(motion["curves"]),
                },
                "synthetic": False,
            }
        )
    dump_jsonl(output / "manifest.jsonl", rows)
    train_motion = np.concatenate(train_motion_chunks, axis=0)
    mean = train_motion.mean(axis=0)
    standard_deviation = train_motion.std(axis=0)
    standard_deviation = np.maximum(standard_deviation, 1e-4)
    dimension_names = [
        *schema.curve_names,
        "head_qx",
        "head_qy",
        "head_qz",
        "head_qw",
        "head_tx_cm",
        "head_ty_cm",
        "head_tz_cm",
    ]
    supervision_weights = np.ones(schema.motion_dim, dtype=np.float64)
    supervision_weights[schema.curve_names.index("tongueOut")] = 0.0
    dump_json(
        output / "motion_normalization.json",
        {
            "normalization_id": "ravdess_train_identities_zscore_v1",
            "fitted_split": "train",
            "fitted_actor_ids": [f"ravdess_actor_{actor:02d}" for actor in range(1, 15)],
            "frames": int(len(train_motion)),
            "dimension_names": dimension_names,
            "mean": mean.tolist(),
            "standard_deviation": standard_deviation.tolist(),
            "supervision_weights": supervision_weights.tolist(),
            "degraded_dimensions": ["tongueOut"],
        },
    )
    report: dict[str, object] = {
        "release_id": "ravdess_mediapipe_pilot_v1",
        "gate": "D0P",
        "track": "research",
        "records": len(rows),
        "hours": sum(durations) / 3600,
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "identities": len({row["face_identity_id"] for row in rows}),
        "label_status": "silver_machine_labels_not_e00_approved",
        "motion_normalization": "motion_normalization.json",
        "motion_preprocessing": "relative_head_clip_origin_v1",
    }
    dump_json(output / "release_summary.json", report)
    return report
