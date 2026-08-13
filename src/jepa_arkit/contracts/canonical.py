from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jepa_arkit.errors import ContractError
from jepa_arkit.io import load_json, stable_hash


@dataclass(frozen=True)
class CurveSpec:
    name: str
    group: str
    minimum: float
    maximum: float


@dataclass(frozen=True)
class CanonicalSchema:
    schema_id: str
    fps: int
    curves: tuple[CurveSpec, ...]
    head_rotation: str
    head_translation_unit: str

    @classmethod
    def from_file(cls, path: str | Path) -> CanonicalSchema:
        raw = load_json(path)
        curves = tuple(
            CurveSpec(
                name=str(item["name"]),
                group=str(item["group"]),
                minimum=float(item.get("min", 0.0)),
                maximum=float(item.get("max", 1.0)),
            )
            for item in raw["curves"]
        )
        schema = cls(
            schema_id=str(raw["schema_id"]),
            fps=int(raw["fps"]),
            curves=curves,
            head_rotation=str(raw["head"]["rotation"]),
            head_translation_unit=str(raw["head"]["translation_unit"]),
        )
        schema.validate()
        return schema

    def validate(self) -> None:
        if self.fps <= 0:
            raise ContractError("fps must be positive")
        names = self.curve_names
        if not names or len(names) != len(set(names)):
            raise ContractError("curve names must be non-empty and unique")
        allowed_groups = {"mouth_jaw", "eyes_brows", "gaze", "nose_cheek"}
        for curve in self.curves:
            if curve.group not in allowed_groups:
                raise ContractError(f"Unknown semantic group: {curve.group}")
            if curve.minimum >= curve.maximum:
                raise ContractError(f"Invalid range for {curve.name}")
        if self.head_rotation != "quaternion_xyzw":
            raise ContractError("Only quaternion_xyzw is supported")

    @property
    def curve_names(self) -> tuple[str, ...]:
        return tuple(curve.name for curve in self.curves)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "schema_id": self.schema_id,
                "fps": self.fps,
                "curves": [curve.__dict__ for curve in self.curves],
                "head_rotation": self.head_rotation,
                "head_translation_unit": self.head_translation_unit,
            }
        )

    def group_indices(self) -> dict[str, tuple[int, ...]]:
        groups: dict[str, list[int]] = {}
        for index, curve in enumerate(self.curves):
            groups.setdefault(curve.group, []).append(index)
        return {name: tuple(indices) for name, indices in groups.items()}

    @property
    def motion_dim(self) -> int:
        return len(self.curves) + 7

    def model_group_indices(self) -> dict[str, tuple[int, ...]]:
        groups = self.group_indices()
        start = len(self.curves)
        groups["head"] = tuple(range(start, start + 7))
        return groups

    def validate_motion(
        self,
        curves: np.ndarray,
        curve_names: list[str] | tuple[str, ...],
        head_quaternion: np.ndarray,
        head_translation: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        if tuple(curve_names) != self.curve_names:
            raise ContractError("curve_names do not exactly match the canonical schema")
        if curves.ndim != 2 or curves.shape[1] != len(self.curves):
            raise ContractError("curves must have shape [T, K]")
        frames = curves.shape[0]
        if head_quaternion.shape != (frames, 4):
            raise ContractError("head_quaternion must have shape [T, 4]")
        if head_translation.shape != (frames, 3):
            raise ContractError("head_translation must have shape [T, 3]")
        if timestamps.shape != (frames,):
            raise ContractError("timestamps must have shape [T]")
        if not np.isfinite(curves).all() or not np.isfinite(head_quaternion).all():
            raise ContractError("motion contains NaN or Inf")
        if frames > 1 and not np.all(np.diff(timestamps) > 0):
            raise ContractError("timestamps must be strictly increasing")
        norms = np.linalg.norm(head_quaternion, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ContractError("head quaternions must be normalized")
        for index, spec in enumerate(self.curves):
            values = curves[:, index]
            if values.min() < spec.minimum - 1e-5 or values.max() > spec.maximum + 1e-5:
                raise ContractError(f"curve {spec.name} is outside its declared range")
