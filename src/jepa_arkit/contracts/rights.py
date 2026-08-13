from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from jepa_arkit.errors import ContractError, TrackViolation
from jepa_arkit.io import file_hash, load_json


class Track(StrEnum):
    RESEARCH = "research"
    PRODUCT = "product"


@dataclass(frozen=True)
class RightsProfile:
    profile_id: str
    research_allowed: bool
    product_training_allowed: bool
    derivatives_allowed: bool
    public_benchmark_allowed: bool
    evidence_hash: str
    status: str
    evidence_path: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> RightsProfile:
        return cls(
            profile_id=str(value["profile_id"]),
            research_allowed=bool(value["research_allowed"]),
            product_training_allowed=bool(value["product_training_allowed"]),
            derivatives_allowed=bool(value["derivatives_allowed"]),
            public_benchmark_allowed=bool(value.get("public_benchmark_allowed", False)),
            evidence_hash=str(value.get("evidence_hash", "")),
            status=str(value.get("status", "unverified")),
            evidence_path=str(value.get("evidence_path", "")),
        )


def load_rights_registry(path: str | Path) -> dict[str, RightsProfile]:
    registry_path = Path(path).resolve()
    raw = load_json(registry_path)
    profiles = [RightsProfile.from_mapping(item) for item in raw.get("profiles", [])]
    registry = {profile.profile_id: profile for profile in profiles}
    if len(registry) != len(profiles):
        raise ContractError("rights profile IDs must be unique")
    for profile in profiles:
        if profile.status != "verified":
            continue
        if not profile.evidence_path:
            raise ContractError(f"Verified profile {profile.profile_id} has no evidence_path")
        evidence_path = Path(profile.evidence_path)
        if not evidence_path.is_absolute():
            evidence_path = (registry_path.parent / evidence_path).resolve()
        if not evidence_path.is_file():
            raise ContractError(f"Rights evidence does not exist: {evidence_path}")
        if file_hash(evidence_path) != profile.evidence_hash:
            raise ContractError(f"Rights evidence hash mismatch: {profile.profile_id}")
    return registry


def assert_track_compatible(
    track: Track,
    profiles: Iterable[RightsProfile],
    ancestor_tracks: Iterable[Track] = (),
) -> None:
    ancestors = set(ancestor_tracks)
    if track is Track.PRODUCT and Track.RESEARCH in ancestors:
        raise TrackViolation("Product artifacts cannot inherit research-track weights or features")
    for profile in profiles:
        if profile.status != "verified" or not profile.evidence_hash:
            raise TrackViolation(f"Rights profile {profile.profile_id} is not verified")
        if track is Track.RESEARCH and not profile.research_allowed:
            raise TrackViolation(f"Research use is not allowed by {profile.profile_id}")
        if track is Track.PRODUCT and not (
            profile.product_training_allowed and profile.derivatives_allowed
        ):
            raise TrackViolation(f"Product training is not allowed by {profile.profile_id}")
