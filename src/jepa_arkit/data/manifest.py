from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa_arkit.errors import ContractError
from jepa_arkit.io import load_jsonl


@dataclass(frozen=True)
class Quality:
    face_visibility: float
    tracking_confidence: float
    av_sync_confidence: float
    motion_jitter_score: float

    @property
    def supervision_weight(self) -> float:
        return max(0.0, min(1.0, self.tracking_confidence * self.av_sync_confidence))


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    audio_path: str
    motion_path: str
    source_id: str
    speaker_id: str
    face_identity_id: str
    language: str
    recording_condition: str
    rights_profile_id: str
    withdrawal_key: str
    fps: int
    sample_rate: int
    curve_schema: str
    motion_label_version: str
    preprocessing_pipeline: str
    split: str
    quality: Quality
    synthetic: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ManifestRecord:
        required = {
            "clip_id",
            "audio_path",
            "motion_path",
            "source_id",
            "speaker_id",
            "face_identity_id",
            "language",
            "recording_condition",
            "rights_profile_id",
            "withdrawal_key",
            "fps",
            "sample_rate",
            "curve_schema",
            "motion_label_version",
            "preprocessing_pipeline",
            "split",
            "quality",
        }
        missing = required - value.keys()
        if missing:
            raise ContractError(f"Missing manifest fields: {sorted(missing)}")
        quality_raw = value["quality"]
        if not isinstance(quality_raw, dict):
            raise ContractError("quality must be an object")
        quality = Quality(**{name: float(quality_raw[name]) for name in Quality.__annotations__})
        record = cls(
            **{name: value[name] for name in required - {"quality"}},
            quality=quality,
            synthetic=bool(value.get("synthetic", False)),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.split not in {"train", "validation", "test", "wild", "perceptual"}:
            raise ContractError(f"Unknown split: {self.split}")
        if self.fps != 30 or self.sample_rate != 16000:
            raise ContractError("Manifest must use 30 fps motion and 16 kHz audio")
        for name, value in self.quality.__dict__.items():
            if name == "motion_jitter_score":
                if value < 0:
                    raise ContractError("motion_jitter_score cannot be negative")
            elif not 0 <= value <= 1:
                raise ContractError(f"{name} must be in [0, 1]")


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records = [ManifestRecord.from_mapping(row) for row in load_jsonl(path)]
    clip_ids = [record.clip_id for record in records]
    if len(clip_ids) != len(set(clip_ids)):
        raise ContractError("clip_id must be unique")
    return records

