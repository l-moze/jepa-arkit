from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jepa_arkit.contracts.rights import Track
from jepa_arkit.errors import ContractError
from jepa_arkit.io import load_json


@dataclass(frozen=True)
class Provenance:
    model_checkpoint_hash: str
    training_data_release_id: str
    feature_release_id: str
    rights_profile_ids: tuple[str, ...]
    track: Track
    inference_date: str
    inference_environment_hash: str
    curve_schema_version: str
    character_profile_id: str
    export_pipeline_version: str
    ue_engine_compatibility: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Provenance:
        required = set(cls.__annotations__)
        missing = required - value.keys()
        if missing:
            raise ContractError(f"Missing provenance fields: {sorted(missing)}")
        provenance = cls(
            model_checkpoint_hash=str(value["model_checkpoint_hash"]),
            training_data_release_id=str(value["training_data_release_id"]),
            feature_release_id=str(value["feature_release_id"]),
            rights_profile_ids=tuple(str(item) for item in value["rights_profile_ids"]),
            track=Track(str(value["track"])),
            inference_date=str(value["inference_date"]),
            inference_environment_hash=str(value["inference_environment_hash"]),
            curve_schema_version=str(value["curve_schema_version"]),
            character_profile_id=str(value["character_profile_id"]),
            export_pipeline_version=str(value["export_pipeline_version"]),
            ue_engine_compatibility=tuple(str(item) for item in value["ue_engine_compatibility"]),
        )
        provenance.validate()
        return provenance

    @classmethod
    def from_file(cls, path: str | Path) -> Provenance:
        return cls.from_mapping(load_json(path))

    def validate(self) -> None:
        if not self.model_checkpoint_hash.startswith("sha256:"):
            raise ContractError("model_checkpoint_hash must be an explicit SHA-256")
        if not self.inference_environment_hash.startswith("sha256:"):
            raise ContractError("inference_environment_hash must be an explicit SHA-256")
        if not self.rights_profile_ids:
            raise ContractError("rights_profile_ids cannot be empty")
        if not self.ue_engine_compatibility:
            raise ContractError("ue_engine_compatibility cannot be empty")
        try:
            datetime.fromisoformat(self.inference_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError("inference_date must be ISO-8601") from exc

