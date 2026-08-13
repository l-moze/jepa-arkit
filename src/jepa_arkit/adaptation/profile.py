from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.errors import ContractError
from jepa_arkit.io import load_json, stable_hash


@dataclass(frozen=True)
class Degradation:
    strategy: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CharacterProfile:
    character_profile_id: str
    engine_version: str
    metahuman_version: str
    engine_compatibility: tuple[str, ...]
    canonical_schema: str
    target_control_space: str
    curve_map_version: str
    curve_map: dict[str, str]
    degraded_curves: dict[str, Degradation]
    neutral_pose_asset: str
    head_neck_constraint: str
    lod_policy: str
    validation_scene: str

    @classmethod
    def from_file(cls, path: str | Path) -> CharacterProfile:
        value = load_json(path)
        engine = value["engine"]
        return cls(
            character_profile_id=str(value["character_profile_id"]),
            engine_version=str(engine["version"]),
            metahuman_version=str(engine["metahuman_version"]),
            engine_compatibility=tuple(value["engine_compatibility"]),
            canonical_schema=str(value["canonical_schema"]),
            target_control_space=str(value["target_control_space"]),
            curve_map_version=str(value["curve_map_version"]),
            curve_map={str(key): str(item) for key, item in value["curve_map"].items()},
            degraded_curves={
                str(key): Degradation(
                    strategy=str(item["strategy"]),
                    sources=tuple(str(source) for source in item.get("sources", [])),
                )
                for key, item in value.get("degraded_curves", {}).items()
            },
            neutral_pose_asset=str(value["neutral_pose_asset"]),
            head_neck_constraint=str(value["head_neck_constraint"]),
            lod_policy=str(value["lod_policy"]),
            validation_scene=str(value["validation_scene"]),
        )

    def validate(self, schema: CanonicalSchema) -> dict[str, object]:
        if self.canonical_schema != schema.schema_id:
            raise ContractError("character profile references a different canonical schema")
        if not self.engine_compatibility or self.engine_version not in self.engine_compatibility:
            raise ContractError("engine version is not declared compatible")
        allowed_degradations = {"zero", "derived", "nearest"}
        mapped = set(self.curve_map)
        degraded = set(self.degraded_curves)
        overlap = mapped & degraded
        if overlap:
            raise ContractError(f"curves cannot be both mapped and degraded: {sorted(overlap)}")
        canonical = set(schema.curve_names)
        unknown = (mapped | degraded) - canonical
        if unknown:
            raise ContractError(f"profile contains unknown canonical curves: {sorted(unknown)}")
        missing = canonical - mapped - degraded
        if missing:
            raise ContractError(f"profile has undeclared canonical curves: {sorted(missing)}")
        for name, rule in self.degraded_curves.items():
            if rule.strategy not in allowed_degradations:
                raise ContractError(f"invalid degradation strategy for {name}: {rule.strategy}")
            if rule.strategy in {"derived", "nearest"} and not rule.sources:
                raise ContractError(f"degradation {name} requires sources")
            if any(source not in canonical for source in rule.sources):
                raise ContractError(f"degradation {name} references unknown sources")
        coverage = len(mapped) / len(canonical)
        return {
            "character_profile_id": self.character_profile_id,
            "coverage": coverage,
            "mapped": len(mapped),
            "degraded": len(degraded),
            "fingerprint": stable_hash(
                {
                    **self.__dict__,
                    "degraded_curves": {
                        name: rule.__dict__ for name, rule in self.degraded_curves.items()
                    },
                }
            ),
        }

