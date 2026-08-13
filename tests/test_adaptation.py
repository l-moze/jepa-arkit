import json
from pathlib import Path

import pytest

from jepa_arkit.adaptation import CharacterProfile
from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.errors import ContractError

ROOT = Path(__file__).parents[1]


def test_character_profile_requires_explicit_mapping_or_degradation(tmp_path: Path) -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    profile_path = tmp_path / "profile.json"
    value = {
        "character_profile_id": "test",
        "engine": {"version": "5.6.2", "metahuman_version": "test"},
        "engine_compatibility": ["5.6.2"],
        "canonical_schema": schema.schema_id,
        "target_control_space": "arkit",
        "curve_map_version": "v1",
        "curve_map": {name: name for name in schema.curve_names[:-1]},
        "degraded_curves": {},
        "neutral_pose_asset": "sha256:test",
        "head_neck_constraint": "sha256:test",
        "lod_policy": "sha256:test",
        "validation_scene": "sha256:test",
    }
    profile_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="undeclared"):
        CharacterProfile.from_file(profile_path).validate(schema)

