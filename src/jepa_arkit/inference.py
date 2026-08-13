from __future__ import annotations

import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.contracts.provenance import Provenance
from jepa_arkit.contracts.rights import Track
from jepa_arkit.contracts.streaming import StreamingProtocol
from jepa_arkit.features.wavlm import wavlm_frame_timestamps
from jepa_arkit.io import dump_json, file_hash, load_json, load_yaml
from jepa_arkit.models.direct import DirectCausalModel
from jepa_arkit.models.disentangled import RegionalConditionalVAE
from jepa_arkit.streaming import run_reference_trace
from jepa_arkit.training.checkpoint import load_checkpoint
from jepa_arkit.training.environment import environment_report


def _read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != 16_000 or audio.getnchannels() != 1:
            raise ValueError("Inference audio must be mono 16 kHz PCM")
        if audio.getsampwidth() != 2:
            raise ValueError("Inference audio must be PCM16")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _wavlm_features(
    audio: np.ndarray, *, device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    except ImportError as exc:
        raise RuntimeError("Install the features extra before running inference") from exc
    model_id = "microsoft/wavlm-base"
    revision = "efa81aae7ff777e464159e0f877d54eac5b84f81"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=revision)
    model = WavLMModel.from_pretrained(model_id, revision=revision).to(device).eval()
    inputs = processor(
        audio,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
        return_attention_mask=True,
    )
    with torch.inference_mode():
        hidden = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
        ).last_hidden_state[0]
    output_length = int(model._get_feat_extract_output_lengths(len(audio)))
    features = hidden[:output_length].float().cpu().numpy()
    timestamps = wavlm_frame_timestamps(
        output_length,
        sample_rate=16_000,
        convolution_kernels=tuple(int(value) for value in model.config.conv_kernel),
        convolution_strides=tuple(int(value) for value in model.config.conv_stride),
    )
    metadata = {
        "model_id": model_id,
        "model_revision": revision,
        "frame_hz": 50.0,
        "timestamp_policy": "convolution_receptive_field_center",
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, timestamps, metadata


def _nearest(
    features: np.ndarray, feature_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(feature_times, target_times, side="left")
    indices = np.clip(indices, 0, len(feature_times) - 1)
    previous = np.clip(indices - 1, 0, len(feature_times) - 1)
    choose_previous = np.abs(feature_times[previous] - target_times) < np.abs(
        feature_times[indices] - target_times
    )
    return features[np.where(choose_previous, previous, indices)]


def _normalise_quaternion(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    result = values / np.maximum(norm, 1e-8)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0:
            result[index] *= -1
    return result


def _postprocess_motion(
    prediction: np.ndarray,
    timestamps: np.ndarray,
    schema: CanonicalSchema,
    normalization: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    standard = np.asarray(normalization["standard_deviation"], dtype=np.float32)
    physical = prediction * standard + mean
    curves = np.clip(physical[:, : len(schema.curves)], 0.0, 1.0)
    for index, spec in enumerate(schema.curves):
        curves[:, index] = np.clip(curves[:, index], spec.minimum, spec.maximum)
    tongue_index = schema.curve_names.index("tongueOut")
    curves[:, tongue_index] = 0.0
    head_quaternion = _normalise_quaternion(
        physical[:, len(schema.curves) : len(schema.curves) + 4]
    )
    head_translation = physical[:, len(schema.curves) + 4 :]
    schema.validate_motion(
        curves,
        list(schema.curve_names),
        head_quaternion,
        head_translation,
        timestamps,
    )
    return curves, head_quaternion, head_translation


@torch.no_grad()
def infer_direct(
    config_path: str | Path,
    checkpoint_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    character_profile_id: str = "canonical_arkit_v1",
    ue_engine: str = "UE5.6.x",
) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    paths = config["paths"]
    schema = CanonicalSchema.from_file(_resolve(config_path, paths["canonical_schema"]))
    normalization = load_json(_resolve(config_path, paths["normalization"]))
    audio_file = Path(audio_path).resolve()
    audio = _read_audio(audio_file)
    feature_values, feature_times, feature_metadata = _wavlm_features(audio, device=device)
    duration = len(audio) / 16_000
    frames = max(1, int(np.ceil(duration * schema.fps)))
    timestamps = np.arange(frames, dtype=np.float64) / schema.fps
    aligned = _nearest(feature_values, feature_times, timestamps)
    model = DirectCausalModel(
        audio_dim=aligned.shape[-1],
        motion_dim=schema.motion_dim,
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
    window = int(config["training"]["frames"])
    chunks: list[torch.Tensor] = []
    for start in range(0, frames, window):
        chunk = torch.from_numpy(aligned[start : start + window]).to(device).unsqueeze(0)
        padding = window - chunk.shape[1]
        if padding:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, padding))
        prediction = model(chunk)[0, : min(window, frames - start)].float().cpu()
        chunks.append(prediction)
    normalized_prediction = torch.cat(chunks, dim=0).numpy()
    curves, head_quaternion, head_translation = _postprocess_motion(
        normalized_prediction, timestamps, schema, normalization
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        curves=curves.astype(np.float32),
        curve_names=np.asarray(schema.curve_names),
        head_quaternion=head_quaternion.astype(np.float32),
        head_translation=head_translation.astype(np.float32),
        timestamps=timestamps,
        fps=np.asarray(schema.fps),
        sample_rate=np.asarray(16_000),
        source_audio_sha256=np.asarray(file_hash(audio_file)),
        model_checkpoint_sha256=np.asarray(file_hash(checkpoint_path)),
    )
    checkpoint_hash = f"sha256:{file_hash(checkpoint_path)}"
    environment = environment_report()
    provenance = Provenance.from_mapping(
        {
            "model_checkpoint_hash": checkpoint_hash,
            "training_data_release_id": str(checkpoint["data_release_id"]),
            "feature_release_id": str(checkpoint["feature_release_id"]),
            "rights_profile_ids": [
                str(config.get("rights_profile_id", "ravdess_cc_by_nc_sa_4_0_research_v1"))
            ],
            "track": str(config["track"]),
            "inference_date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "inference_environment_hash": str(environment["environment_hash"]),
            "curve_schema_version": schema.schema_id,
            "character_profile_id": character_profile_id,
            "export_pipeline_version": "offline_direct_v1",
            "ue_engine_compatibility": [ue_engine],
        }
    )
    sidecar = output.with_suffix(".provenance.json")
    dump_json(sidecar, provenance.__dict__ | {"track": provenance.track.value})
    report = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "output": str(output),
        "provenance": str(sidecar),
        "frames": frames,
        "duration_seconds": duration,
        "fps": schema.fps,
        "feature_metadata": feature_metadata,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_hash": checkpoint_hash,
        "motion_schema_fingerprint": schema.fingerprint,
        "curve_range_max": float(curves.max()),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(head_quaternion, axis=1) - 1))
        ),
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
    }
    dump_json(output.with_suffix(".inference.json"), report)
    return report


@torch.no_grad()
def infer_regional(
    config_path: str | Path,
    checkpoint_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    sampling_seed: int = 0,
    regional_temperature: float = 0.75,
    head_temperature: float = 0.5,
    character_profile_id: str = "canonical_arkit_v1",
    ue_engine: str = "UE5.6.x",
) -> dict[str, object]:
    """Export E03 motion with one reproducible style sample per audio clip."""
    if regional_temperature < 0 or head_temperature < 0:
        raise ValueError("sampling temperatures must be non-negative")
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    paths = config["paths"]
    schema = CanonicalSchema.from_file(_resolve(config_path, paths["canonical_schema"]))
    normalization = load_json(_resolve(config_path, paths["normalization"]))
    audio_file = Path(audio_path).resolve()
    audio = _read_audio(audio_file)
    feature_values, feature_times, feature_metadata = _wavlm_features(audio, device=device)
    duration = len(audio) / 16_000
    frames = max(1, int(np.ceil(duration * schema.fps)))
    timestamps = np.arange(frames, dtype=np.float64) / schema.fps
    aligned = _nearest(feature_values, feature_times, timestamps)
    groups = schema.model_group_indices()
    stochastic = groups["eyes_brows"] + groups["gaze"] + groups["head"]
    model = RegionalConditionalVAE(
        audio_dim=aligned.shape[-1],
        motion_dim=schema.motion_dim,
        stochastic_indices=stochastic,
        residual_indices=groups["head"],
        **dict(config["model"]),
    ).to(device)
    checkpoint = torch.load(
        Path(checkpoint_path).resolve(), map_location=device, weights_only=False
    )
    if Track(str(checkpoint["track"])) is not Track(str(config["track"])):
        raise RuntimeError("Checkpoint track does not match regional inference track")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    style_noise = torch.randn(
        (1, model.latent_dim), device=device, generator=generator, dtype=torch.float32
    )
    window = int(config["training"]["frames"])
    chunks: list[torch.Tensor] = []
    for start in range(0, frames, window):
        chunk = torch.from_numpy(aligned[start : start + window]).to(device).unsqueeze(0)
        padding = window - chunk.shape[1]
        if padding:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, padding))
        prediction = model(
            chunk,
            noise=style_noise,
            regional_temperature=regional_temperature,
            head_temperature=head_temperature,
        )["prediction"][0, : min(window, frames - start)]
        chunks.append(prediction.float().cpu())
    normalized_prediction = torch.cat(chunks, dim=0).numpy()
    curves, head_quaternion, head_translation = _postprocess_motion(
        normalized_prediction, timestamps, schema, normalization
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        curves=curves.astype(np.float32),
        curve_names=np.asarray(schema.curve_names),
        head_quaternion=head_quaternion.astype(np.float32),
        head_translation=head_translation.astype(np.float32),
        timestamps=timestamps,
        fps=np.asarray(schema.fps),
        sample_rate=np.asarray(16_000),
        source_audio_sha256=np.asarray(file_hash(audio_file)),
        model_checkpoint_sha256=np.asarray(file_hash(checkpoint_path)),
        sampling_seed=np.asarray(sampling_seed),
        regional_temperature=np.asarray(regional_temperature),
        head_temperature=np.asarray(head_temperature),
    )
    environment = environment_report()
    provenance = Provenance.from_mapping(
        {
            "model_checkpoint_hash": f"sha256:{file_hash(checkpoint_path)}",
            "training_data_release_id": str(checkpoint["data_release_id"]),
            "feature_release_id": str(checkpoint["feature_release_id"]),
            "rights_profile_ids": [
                str(config.get("rights_profile_id", "ravdess_cc_by_nc_sa_4_0_research_v1"))
            ],
            "track": str(config["track"]),
            "inference_date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "inference_environment_hash": str(environment["environment_hash"]),
            "curve_schema_version": schema.schema_id,
            "character_profile_id": character_profile_id,
            "export_pipeline_version": "offline_regional_v1",
            "ue_engine_compatibility": [ue_engine],
        }
    )
    sampling = {
        "sampling_seed": sampling_seed,
        "noise_strategy": "fixed_clip_latent_noise",
        "regional_temperature": regional_temperature,
        "head_temperature": head_temperature,
    }
    sidecar = output.with_suffix(".provenance.json")
    dump_json(
        sidecar,
        provenance.__dict__ | {"track": provenance.track.value, "sampling": sampling},
    )
    report = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "output": str(output),
        "provenance": str(sidecar),
        "frames": frames,
        "duration_seconds": duration,
        "fps": schema.fps,
        "feature_metadata": feature_metadata,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_hash": f"sha256:{file_hash(checkpoint_path)}",
        "motion_schema_fingerprint": schema.fingerprint,
        "curve_range_max": float(curves.max()),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(head_quaternion, axis=1) - 1))
        ),
        "sampling": sampling,
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
    }
    dump_json(output.with_suffix(".inference.json"), report)
    return report


@torch.no_grad()
def infer_streaming_direct(
    config_path: str | Path,
    checkpoint_path: str | Path,
    audio_path: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
    *,
    repeat_seconds: float | None = None,
    character_profile_id: str = "canonical_arkit_v1",
    ue_engine: str = "UE5.6.x",
) -> dict[str, object]:
    """Run the frozen reference streaming protocol and export canonical motion."""
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    protocol = StreamingProtocol.from_file(protocol_path)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    paths = config["paths"]
    schema = CanonicalSchema.from_file(_resolve(config_path, paths["canonical_schema"]))
    normalization = load_json(_resolve(config_path, paths["normalization"]))
    audio_file = Path(audio_path).resolve()
    audio = _read_audio(audio_file)
    feature_values, feature_times, feature_metadata = _wavlm_features(audio, device=device)
    duration = len(audio) / 16_000
    frames = max(1, int(np.ceil(duration * schema.fps)))
    timestamps = np.arange(frames, dtype=np.float64) / schema.fps
    aligned = _nearest(feature_values, feature_times, timestamps)
    trace_mode = "original"
    if repeat_seconds is not None:
        if repeat_seconds <= 0:
            raise ValueError("repeat_seconds must be positive")
        desired = max(frames, int(np.ceil(repeat_seconds * schema.fps)))
        aligned = np.resize(aligned, (desired, aligned.shape[1]))
        frames = desired
        timestamps = np.arange(frames, dtype=np.float64) / schema.fps
        trace_mode = "repeated_feature_trace"
    model = DirectCausalModel(
        audio_dim=aligned.shape[-1],
        motion_dim=schema.motion_dim,
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
    started = time.perf_counter()
    normalized_prediction, stream_timestamps, stream_report = run_reference_trace(
        model,
        protocol,
        aligned,
        device=device,
        max_frames=int(config["model"]["max_frames"]),
    )
    elapsed = time.perf_counter() - started
    curves, head_quaternion, head_translation = _postprocess_motion(
        normalized_prediction, stream_timestamps, schema, normalization
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        curves=curves.astype(np.float32),
        curve_names=np.asarray(schema.curve_names),
        head_quaternion=head_quaternion.astype(np.float32),
        head_translation=head_translation.astype(np.float32),
        timestamps=stream_timestamps,
        fps=np.asarray(schema.fps),
        sample_rate=np.asarray(16_000),
        source_audio_sha256=np.asarray(file_hash(audio_file)),
        model_checkpoint_sha256=np.asarray(file_hash(checkpoint_path)),
        protocol_fingerprint=np.asarray(protocol.fingerprint),
    )
    environment = environment_report()
    provenance = Provenance.from_mapping(
        {
            "model_checkpoint_hash": f"sha256:{file_hash(checkpoint_path)}",
            "training_data_release_id": str(checkpoint["data_release_id"]),
            "feature_release_id": str(checkpoint["feature_release_id"]),
            "rights_profile_ids": [
                str(config.get("rights_profile_id", "ravdess_cc_by_nc_sa_4_0_research_v1"))
            ],
            "track": str(config["track"]),
            "inference_date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "inference_environment_hash": str(environment["environment_hash"]),
            "curve_schema_version": schema.schema_id,
            "character_profile_id": character_profile_id,
            "export_pipeline_version": "streaming_direct_v1",
            "ue_engine_compatibility": [ue_engine],
        }
    )
    sidecar = output.with_suffix(".provenance.json")
    dump_json(sidecar, provenance.__dict__ | {"track": provenance.track.value})
    report = {
        "status": "pilot_non_comparable" if config.get("pilot_non_comparable") else "completed",
        "trace_mode": trace_mode,
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "output": str(output),
        "provenance": str(sidecar),
        "frames": frames,
        "duration_seconds": frames / schema.fps,
        "elapsed_seconds": elapsed,
        "real_time_factor": (frames / schema.fps) / max(elapsed, 1e-9),
        "feature_metadata": feature_metadata,
        "stream": stream_report,
        "timestamps_contiguous": bool(
            np.array_equal(stream_timestamps, np.arange(frames) / schema.fps)
        ),
        "curve_range_max": float(curves.max()),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(head_quaternion, axis=1) - 1))
        ),
        "pilot_non_comparable": bool(config.get("pilot_non_comparable")),
    }
    dump_json(output.with_suffix(".streaming.json"), report)
    return report
