from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.io import dump_json, dump_jsonl, file_hash


def _write_wave(path: Path, frequency: float, seconds: float = 1.0) -> None:
    sample_rate = 16000
    samples = [
        int(0.2 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(int(seconds * sample_rate))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def create_demo_dataset(output: str | Path, schema_path: str | Path) -> dict[str, object]:
    output = Path(output).resolve()
    schema = CanonicalSchema.from_file(schema_path)
    rng = np.random.default_rng(20260812)
    rows: list[dict[str, object]] = []
    splits = ["train"] * 15 + ["validation"] * 5 + ["test"] * 5
    for index, split in enumerate(splits):
        clip_id = f"synthetic/identity_{index:03d}/clip_000"
        audio_path = output / "audio" / f"clip_{index:03d}.wav"
        motion_path = output / "motion" / f"clip_{index:03d}.npz"
        _write_wave(audio_path, 180 + index * 7)
        frames = 30
        curves = rng.uniform(0.0, 0.35, size=(frames, len(schema.curves))).astype(np.float32)
        time = np.arange(frames, dtype=np.float64) / schema.fps
        jaw = schema.curve_names.index("jawOpen")
        curves[:, jaw] = (0.5 + 0.45 * np.sin(2 * np.pi * (2 + index % 3) * time)).astype(
            np.float32
        )
        quaternion = np.zeros((frames, 4), dtype=np.float32)
        quaternion[:, 3] = 1.0
        translation = np.zeros((frames, 3), dtype=np.float32)
        confidence = np.full(frames, 0.95, dtype=np.float32)
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            motion_path,
            curves=curves,
            curve_names=np.asarray(schema.curve_names),
            head_quaternion=quaternion,
            head_translation=translation,
            frame_confidence=confidence,
            timestamps=time,
        )
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": str(audio_path),
                "motion_path": str(motion_path),
                "source_id": f"synthetic_source_{index:03d}",
                "speaker_id": f"speaker_{index:03d}",
                "face_identity_id": f"identity_{index:03d}",
                "language": "en",
                "recording_condition": "synthetic",
                "rights_profile_id": "rights_synthetic_research_v1",
                "withdrawal_key": f"withdrawal_{index:03d}",
                "fps": 30,
                "sample_rate": 16000,
                "curve_schema": schema.schema_id,
                "motion_label_version": "synthetic_raw_v1",
                "preprocessing_pipeline": "synthetic_identity_v1",
                "quality": {
                    "face_visibility": 1.0,
                    "tracking_confidence": 0.95,
                    "av_sync_confidence": 0.95,
                    "motion_jitter_score": 0.01,
                },
                "split": split,
                "synthetic": True,
            }
        )
    dump_jsonl(output / "manifest.jsonl", rows)
    evidence = output / "synthetic_rights_evidence.txt"
    evidence.write_text("Synthetic fixture generated locally; research infrastructure only.\n")
    dump_json(
        output / "rights_registry.json",
        {
            "profiles": [
                {
                    "profile_id": "rights_synthetic_research_v1",
                    "research_allowed": True,
                    "product_training_allowed": False,
                    "derivatives_allowed": True,
                    "public_benchmark_allowed": False,
                    "evidence_hash": file_hash(evidence),
                    "evidence_path": evidence.name,
                    "status": "verified",
                }
            ]
        },
    )
    return {"output": str(output), "records": len(rows), "schema": schema.schema_id}
