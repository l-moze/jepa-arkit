from pathlib import Path

import numpy as np
import pytest

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.contracts.rights import RightsProfile, Track, assert_track_compatible
from jepa_arkit.contracts.streaming import StreamingProtocol
from jepa_arkit.errors import ContractError, TrackViolation

ROOT = Path(__file__).parents[1]


def test_canonical_schema_validates_motion() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    frames = 5
    curves = np.zeros((frames, len(schema.curves)), dtype=np.float32)
    quaternion = np.zeros((frames, 4), dtype=np.float32)
    quaternion[:, 3] = 1
    schema.validate_motion(
        curves,
        schema.curve_names,
        quaternion,
        np.zeros((frames, 3), dtype=np.float32),
        np.arange(frames, dtype=np.float64) / 30,
    )


def test_canonical_schema_rejects_wrong_curve_order() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    frames = 2
    quaternion = np.zeros((frames, 4), dtype=np.float32)
    quaternion[:, 3] = 1
    with pytest.raises(ContractError, match="curve_names"):
        schema.validate_motion(
            np.zeros((frames, len(schema.curves)), dtype=np.float32),
            tuple(reversed(schema.curve_names)),
            quaternion,
            np.zeros((frames, 3), dtype=np.float32),
            np.arange(frames) / 30,
        )


def test_product_track_rejects_research_ancestry() -> None:
    with pytest.raises(TrackViolation, match="inherit"):
        assert_track_compatible(Track.PRODUCT, [], [Track.RESEARCH])


def test_unverified_rights_block_research() -> None:
    profile = RightsProfile("x", True, False, True, False, "", "unverified")
    with pytest.raises(TrackViolation, match="not verified"):
        assert_track_compatible(Track.RESEARCH, [profile])


def test_streaming_contract_requires_ue_interpolation(tmp_path: Path) -> None:
    protocol = StreamingProtocol.from_file(
        ROOT / "configs/contracts/streaming_protocol_candidate.json"
    )
    assert protocol.output_timestamp(30) == 1.0
    invalid = protocol.__dict__ | {"interpolation_owner": "model"}
    with pytest.raises(ContractError, match="owned by UE"):
        StreamingProtocol(**invalid).validate()
