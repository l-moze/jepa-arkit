from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jepa_arkit.contracts.rights import Track
from jepa_arkit.errors import ContractError
from jepa_arkit.io import dump_json, load_json, stable_hash


@dataclass(frozen=True)
class FeatureMetadata:
    feature_release_id: str
    model_id: str
    model_revision: str
    layer: str
    frame_hz: float
    feature_dim: int
    dtype: str
    normalization: str
    track: Track
    source_data_release_id: str

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> FeatureMetadata:
        metadata = cls(
            feature_release_id=str(value["feature_release_id"]),
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            layer=str(value["layer"]),
            frame_hz=float(value["frame_hz"]),
            feature_dim=int(value["feature_dim"]),
            dtype=str(value["dtype"]),
            normalization=str(value["normalization"]),
            track=Track(str(value["track"])),
            source_data_release_id=str(value["source_data_release_id"]),
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.frame_hz <= 0 or self.feature_dim <= 0:
            raise ContractError("feature frame rate and dimension must be positive")
        if self.dtype not in {"float16", "float32"}:
            raise ContractError("feature dtype must be float16 or float32")
        if not self.model_revision:
            raise ContractError("feature model revision cannot be empty")

    @property
    def fingerprint(self) -> str:
        value = asdict(self)
        value["track"] = self.track.value
        return stable_hash(value)


class FeatureStore:
    """Sample-addressable feature store with one NPZ per clip."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.metadata = FeatureMetadata.from_mapping(load_json(self.root / "metadata.json"))
        self.index = load_json(self.root / "index.json")

    @classmethod
    def create(cls, root: str | Path, metadata: FeatureMetadata) -> FeatureStore:
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        value = asdict(metadata)
        value["track"] = metadata.track.value
        dump_json(root / "metadata.json", value)
        dump_json(root / "index.json", {"clips": {}})
        return cls(root)

    def write(
        self,
        clip_id: str,
        withdrawal_key: str,
        features: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        if features.ndim != 2 or features.shape[1] != self.metadata.feature_dim:
            raise ContractError("features must have shape [T, feature_dim]")
        if timestamps.shape != (features.shape[0],):
            raise ContractError("feature timestamps must have shape [T]")
        if not np.isfinite(features).all() or not np.all(np.diff(timestamps) > 0):
            raise ContractError("features must be finite and timestamps strictly increasing")
        key = stable_hash(clip_id)[:24]
        relative_path = Path("clips") / f"{key}.npz"
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            features=features.astype(self.metadata.dtype),
            timestamps=timestamps.astype(np.float64),
        )
        index = load_json(self.root / "index.json")
        clips = index.setdefault("clips", {})
        clips[clip_id] = {
            "path": relative_path.as_posix(),
            "withdrawal_key": withdrawal_key,
            "frames": int(features.shape[0]),
        }
        dump_json(self.root / "index.json", index)
        self.index = index

    def write_batch(
        self,
        entries: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> None:
        index = load_json(self.root / "index.json")
        clips = index.setdefault("clips", {})
        for clip_id, withdrawal_key, features, timestamps in entries:
            if features.ndim != 2 or features.shape[1] != self.metadata.feature_dim:
                raise ContractError("features must have shape [T, feature_dim]")
            if timestamps.shape != (features.shape[0],):
                raise ContractError("feature timestamps must have shape [T]")
            if not np.isfinite(features).all() or not np.all(np.diff(timestamps) > 0):
                raise ContractError("features must be finite and timestamps strictly increasing")
            key = stable_hash(clip_id)[:24]
            relative_path = Path("clips") / f"{key}.npz"
            target = self.root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                features=features.astype(self.metadata.dtype),
                timestamps=timestamps.astype(np.float64),
            )
            temporary.replace(target)
            clips[clip_id] = {
                "path": relative_path.as_posix(),
                "withdrawal_key": withdrawal_key,
                "frames": int(features.shape[0]),
            }
        temporary_index = self.root / "index.tmp.json"
        dump_json(temporary_index, index)
        temporary_index.replace(self.root / "index.json")
        self.index = index

    def read(self, clip_id: str) -> tuple[np.ndarray, np.ndarray]:
        entry = self.index.get("clips", {}).get(clip_id)
        if entry is None:
            raise KeyError(clip_id)
        with np.load(self.root / entry["path"], allow_pickle=False) as archive:
            features = archive["features"].astype(np.float32)
            timestamps = archive["timestamps"].astype(np.float64)
        return features, timestamps

    def affected_by_withdrawal(self, withdrawal_key: str) -> tuple[str, ...]:
        clips = self.index.get("clips", {})
        return tuple(
            sorted(
                clip_id
                for clip_id, entry in clips.items()
                if entry["withdrawal_key"] == withdrawal_key
            )
        )


def align_features_to_motion(
    features: np.ndarray,
    feature_timestamps: np.ndarray,
    motion_timestamps: np.ndarray,
) -> np.ndarray:
    if features.ndim != 2:
        raise ContractError("features must be [T, D]")
    if not np.all(np.diff(feature_timestamps) > 0) or not np.all(np.diff(motion_timestamps) > 0):
        raise ContractError("timestamps must be strictly increasing")
    indices = np.searchsorted(feature_timestamps, motion_timestamps, side="left")
    indices = np.clip(indices, 0, len(feature_timestamps) - 1)
    previous = np.clip(indices - 1, 0, len(feature_timestamps) - 1)
    choose_previous = np.abs(feature_timestamps[previous] - motion_timestamps) < np.abs(
        feature_timestamps[indices] - motion_timestamps
    )
    selected = np.where(choose_previous, previous, indices)
    return features[selected]
