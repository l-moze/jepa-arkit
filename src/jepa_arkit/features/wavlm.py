from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import torch

from jepa_arkit.contracts.rights import Track
from jepa_arkit.errors import ContractError, GateBlocked
from jepa_arkit.features.store import FeatureMetadata, FeatureStore
from jepa_arkit.io import dump_json, load_json, load_jsonl


def _read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != 16_000 or audio.getnchannels() != 1:
            raise ContractError(f"Expected mono 16 kHz audio: {path}")
        if audio.getsampwidth() != 2:
            raise ContractError(f"Expected PCM16 audio: {path}")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def wavlm_frame_timestamps(
    output_length: int,
    *,
    sample_rate: int,
    convolution_kernels: tuple[int, ...],
    convolution_strides: tuple[int, ...],
) -> np.ndarray:
    """Return feature-frame centers from the convolutional receptive field."""
    if output_length < 0:
        raise ValueError("output_length must be non-negative")
    if len(convolution_kernels) != len(convolution_strides) or not convolution_kernels:
        raise ValueError("convolution kernels and strides must be non-empty and equal length")
    receptive_field = 1
    accumulated_stride = 1
    for kernel, stride in zip(convolution_kernels, convolution_strides, strict=True):
        receptive_field += (kernel - 1) * accumulated_stride
        accumulated_stride *= stride
    centers = np.arange(output_length, dtype=np.float64) * accumulated_stride
    centers += (receptive_field - 1) / 2
    return centers / sample_rate


def extract_wavlm_features(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    model_id: str = "microsoft/wavlm-base",
    revision: str,
    source_data_release_id: str,
    feature_release_id: str = "wavlm_base_fp16_ravdess_v2",
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, object]:
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    except ImportError as exc:
        raise GateBlocked("Install the features extra: uv sync --extra features") from exc
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    manifest = Path(manifest_path).resolve()
    records = load_jsonl(manifest)
    if not records:
        raise GateBlocked("Feature extraction manifest is empty")
    output = Path(output_root).resolve()
    metadata = FeatureMetadata(
        feature_release_id=feature_release_id,
        model_id=model_id,
        model_revision=revision,
        layer="last_hidden_state",
        frame_hz=50.0,
        feature_dim=768,
        dtype="float16",
        normalization=(
            "wav2vec2_zero_mean_unit_variance_per_clip;"
            "timestamps=convolution_receptive_field_center"
        ),
        track=Track.RESEARCH,
        source_data_release_id=source_data_release_id,
    )
    if (output / "metadata.json").is_file():
        store = FeatureStore(output)
        if store.metadata.fingerprint != metadata.fingerprint:
            raise ContractError("Existing feature store metadata does not match requested release")
    else:
        store = FeatureStore.create(output, metadata)
    indexed = set(load_json(output / "index.json").get("clips", {}))
    pending = [record for record in records if str(record["clip_id"]) not in indexed]

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=revision)
    model = WavLMModel.from_pretrained(model_id, revision=revision).to(device).eval()
    convolution_kernels = tuple(int(value) for value in model.config.conv_kernel)
    convolution_strides = tuple(int(value) for value in model.config.conv_stride)
    started = time.perf_counter()
    frames = 0
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        audio_arrays = [_read_audio(Path(str(record["audio_path"]))) for record in batch]
        inputs = processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.inference_mode():
            hidden = model(input_values, attention_mask=attention_mask).last_hidden_state
        entries: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for index, (record, samples) in enumerate(zip(batch, audio_arrays, strict=True)):
            output_length = int(model._get_feat_extract_output_lengths(len(samples)))
            features = hidden[index, :output_length].float().cpu().numpy()
            timestamps = wavlm_frame_timestamps(
                output_length,
                sample_rate=16_000,
                convolution_kernels=convolution_kernels,
                convolution_strides=convolution_strides,
            )
            entries.append(
                (
                    str(record["clip_id"]),
                    str(record["withdrawal_key"]),
                    features,
                    timestamps,
                )
            )
            frames += output_length
        store.write_batch(entries)
        print(
            {
                "completed": min(batch_start + len(batch), len(pending)),
                "pending": len(pending),
            },
            flush=True,
        )
    elapsed = time.perf_counter() - started
    final_index = load_json(output / "index.json").get("clips", {})
    report: dict[str, object] = {
        "status": "passed" if len(final_index) == len(records) else "failed",
        "clips": len(final_index),
        "new_clips": len(pending),
        "frames_extracted": frames,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "model_id": model_id,
        "model_revision": revision,
        "feature_fingerprint": metadata.fingerprint,
        "convolution_kernels": convolution_kernels,
        "convolution_strides": convolution_strides,
    }
    dump_json(output / "extraction_report.json", report)
    return report
