from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.contracts.rights import Track
from jepa_arkit.data.audit import AuditReport
from jepa_arkit.data.manifest import ManifestRecord, load_manifest
from jepa_arkit.data.motion import load_and_validate_motion
from jepa_arkit.errors import GateBlocked, TrackViolation
from jepa_arkit.features.store import FeatureStore, align_features_to_motion
from jepa_arkit.io import load_json


@dataclass(frozen=True)
class Window:
    audio: torch.Tensor
    motion: torch.Tensor
    confidence: torch.Tensor
    dimension_weights: torch.Tensor
    clip_id: str


class MotionWindowDataset(Dataset[Window]):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        audit_report_path: str | Path,
        schema_path: str | Path,
        feature_store_path: str | Path,
        split: str,
        frames: int,
        track: Track,
        allowed_gates: tuple[str, ...] = ("D0B",),
        normalization_path: str | Path | None = None,
    ) -> None:
        audit = AuditReport.from_dict(load_json(audit_report_path))
        if not audit.passed or audit.gate not in allowed_gates:
            raise GateBlocked(
                "Training requires a passed audit report for one of "
                f"{allowed_gates}; received {audit.gate}"
            )
        if Track(audit.track) is not track:
            raise TrackViolation("Requested track does not match the D0B release")
        self.schema = CanonicalSchema.from_file(schema_path)
        self.normalization = (
            load_json(normalization_path) if normalization_path is not None else None
        )
        if self.normalization is not None:
            expected = self.schema.motion_dim
            for field in ("mean", "standard_deviation", "supervision_weights"):
                if len(self.normalization[field]) != expected:
                    raise GateBlocked(f"Normalization {field} has wrong dimension")
            if self.normalization.get("fitted_split") != "train":
                raise GateBlocked("Motion normalization must be fitted on train only")
        self.feature_store = FeatureStore(feature_store_path)
        if self.feature_store.metadata.track is not track:
            raise TrackViolation("Feature release track does not match the training track")
        if self.feature_store.metadata.source_data_release_id != audit.release_id:
            raise GateBlocked("Feature release ancestry does not match the D0B data release")
        self.frames = frames
        self.records = [record for record in load_manifest(manifest_path) if record.split == split]
        self.manifest_root = Path(manifest_path).resolve().parent
        self.windows: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            motion = self._motion(record)
            motion_frames = int(motion["curves"].shape[0])
            if motion_frames <= frames:
                starts = [0]
            else:
                starts = sorted({0, motion_frames - frames})
            self.windows.extend((record_index, start) for start in starts)
        if not self.windows:
            raise GateBlocked(f"No {frames}-frame windows available for split={split}")

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.manifest_root / path).resolve()

    def _motion(self, record: ManifestRecord) -> dict[str, np.ndarray]:
        return load_and_validate_motion(self._resolve(record.motion_path), self.schema)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Window:
        record_index, start = self.windows[index]
        record = self.records[record_index]
        motion = self._motion(record)
        stop = start + self.frames
        features, feature_timestamps = self.feature_store.read(record.clip_id)
        aligned = align_features_to_motion(features, feature_timestamps, motion["timestamps"])
        curve_motion = motion["curves"]
        full_motion = np.concatenate(
            (curve_motion, motion["head_quaternion"], motion["head_translation"]), axis=-1
        )
        if self.normalization is not None:
            mean = np.asarray(self.normalization["mean"], dtype=np.float32)
            standard_deviation = np.asarray(
                self.normalization["standard_deviation"], dtype=np.float32
            )
            full_motion = (full_motion - mean) / standard_deviation
            dimension_weights = np.asarray(
                self.normalization["supervision_weights"], dtype=np.float32
            )
        else:
            dimension_weights = np.ones(self.schema.motion_dim, dtype=np.float32)
        audio_window = aligned[start:stop].astype(np.float32)
        motion_window = full_motion[start:stop].astype(np.float32)
        confidence_window = motion["frame_confidence"][start:stop].astype(np.float32)
        padding = self.frames - len(motion_window)
        if padding > 0:
            audio_window = np.pad(audio_window, ((0, padding), (0, 0)))
            motion_window = np.pad(motion_window, ((0, padding), (0, 0)))
            confidence_window = np.pad(confidence_window, (0, padding))
        return Window(
            audio=torch.from_numpy(audio_window),
            motion=torch.from_numpy(motion_window),
            confidence=torch.from_numpy(confidence_window),
            dimension_weights=torch.from_numpy(dimension_weights),
            clip_id=record.clip_id,
        )
