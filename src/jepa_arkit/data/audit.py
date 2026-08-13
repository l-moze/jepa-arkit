from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.contracts.rights import Track, assert_track_compatible, load_rights_registry
from jepa_arkit.data.manifest import ManifestRecord, load_manifest
from jepa_arkit.data.motion import load_and_validate_motion
from jepa_arkit.errors import ContractError, TrackViolation
from jepa_arkit.io import file_hash, load_yaml, stable_hash


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    clip_id: str | None = None


@dataclass(frozen=True)
class AuditReport:
    gate: str
    passed: bool
    release_id: str
    track: str
    manifest_fingerprint: str
    counts: dict[str, object]
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "issues": [asdict(issue) for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> AuditReport:
        return cls(
            gate=str(value["gate"]),
            passed=bool(value["passed"]),
            release_id=str(value["release_id"]),
            track=str(value["track"]),
            manifest_fingerprint=str(value["manifest_fingerprint"]),
            counts=dict(value["counts"]),
            issues=tuple(AuditIssue(**issue) for issue in value.get("issues", [])),
        )


def _leakage_issues(records: list[ManifestRecord]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for field in ("speaker_id", "face_identity_id", "withdrawal_key"):
        split_by_value: dict[str, set[str]] = defaultdict(set)
        for record in records:
            split_by_value[str(getattr(record, field))].add(record.split)
        for value, splits in split_by_value.items():
            compared = splits & {"train", "validation", "test"}
            if len(compared) > 1:
                issues.append(
                    AuditIssue(
                        "error",
                        f"split_leak_{field}",
                        f"{field}={value} appears in splits {sorted(compared)}",
                    )
                )
    return issues


def _distribution_issues(records: list[ManifestRecord]) -> list[AuditIssue]:
    train = [record for record in records if record.split == "train"]
    issues: list[AuditIssue] = []
    if not train:
        return [AuditIssue("error", "empty_train", "Training split is empty")]
    for field in ("source_id", "face_identity_id"):
        counts = Counter(str(getattr(record, field)) for record in train)
        value, count = counts.most_common(1)[0]
        share = count / len(train)
        if share > 0.40:
            issues.append(
                AuditIssue(
                    "warning" if field == "source_id" else "error",
                    f"train_dominance_{field}",
                    f"{field}={value} occupies {share:.1%} of training records",
                )
            )
    validation_identities = {
        record.face_identity_id for record in records if record.split == "validation"
    }
    if len(validation_identities) < 5 and not all(record.synthetic for record in records):
        issues.append(
            AuditIssue(
                "error",
                "validation_identity_count",
                f"Validation contains {len(validation_identities)} identities; "
                "at least 5 are required",
            )
        )
    return issues


def audit_release(config_path: str | Path, validate_files: bool = True) -> AuditReport:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    base = config_path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    release_id = str(config["release_id"])
    track = Track(str(config["track"]))
    gate = str(config.get("gate", "D0A"))
    records = load_manifest(resolve(str(config["manifest"])))
    schema = CanonicalSchema.from_file(resolve(str(config["canonical_schema"])))
    registry = load_rights_registry(resolve(str(config["rights_registry"])))
    issues = _leakage_issues(records) + _distribution_issues(records)
    pipelines = {record.preprocessing_pipeline for record in records}
    label_versions = {record.motion_label_version for record in records}
    if gate == "D0B" and len(pipelines) != 1:
        issues.append(
            AuditIssue("error", "mixed_preprocessing", "D0B release must freeze one pipeline")
        )
    if gate == "D0B" and len(label_versions) != 1:
        issues.append(
            AuditIssue("error", "mixed_label_version", "D0B release must freeze one label")
        )
    if track is Track.PRODUCT and any(record.synthetic for record in records):
        issues.append(
            AuditIssue("error", "synthetic_product", "Synthetic records cannot enter product")
        )
    for record in records:
        profile = registry.get(record.rights_profile_id)
        if profile is None:
            issues.append(
                AuditIssue("error", "missing_rights", record.rights_profile_id, record.clip_id)
            )
            continue
        try:
            assert_track_compatible(track, [profile])
        except TrackViolation as exc:
            issues.append(AuditIssue("error", "rights_violation", str(exc), record.clip_id))
        if record.curve_schema != schema.schema_id:
            issues.append(
                AuditIssue("error", "schema_mismatch", record.curve_schema, record.clip_id)
            )
        if validate_files:
            audio_path = resolve(record.audio_path)
            motion_path = resolve(record.motion_path)
            if not audio_path.is_file():
                issues.append(AuditIssue("error", "missing_audio", str(audio_path), record.clip_id))
            if not motion_path.is_file():
                issues.append(
                    AuditIssue("error", "missing_motion", str(motion_path), record.clip_id)
                )
            else:
                try:
                    load_and_validate_motion(motion_path, schema)
                except ContractError as exc:
                    issues.append(AuditIssue("error", "invalid_motion", str(exc), record.clip_id))
    counts: dict[str, object] = {
        "records": len(records),
        "splits": dict(Counter(record.split for record in records)),
        "identities": len({record.face_identity_id for record in records}),
        "speakers": len({record.speaker_id for record in records}),
        "sources": len({record.source_id for record in records}),
        "rights_profiles": sorted({record.rights_profile_id for record in records}),
        "synthetic": sum(record.synthetic for record in records),
        "schema_fingerprint": schema.fingerprint,
    }
    if validate_files:
        counts["file_fingerprints"] = {
            record.clip_id: {
                "audio": file_hash(resolve(record.audio_path))
                if resolve(record.audio_path).is_file()
                else None,
                "motion": file_hash(resolve(record.motion_path))
                if resolve(record.motion_path).is_file()
                else None,
            }
            for record in records
        }
    fingerprint = stable_hash(
        [record.__dict__ | {"quality": record.quality.__dict__} for record in records]
    )
    return AuditReport(
        gate=gate,
        passed=not any(issue.severity == "error" for issue in issues),
        release_id=release_id,
        track=track.value,
        manifest_fingerprint=fingerprint,
        counts=counts,
        issues=tuple(issues),
    )
