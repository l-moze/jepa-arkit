from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jepa_arkit.errors import ContractError
from jepa_arkit.io import file_hash, load_yaml, stable_hash


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_id: str
    name: str
    source_url: str | None
    modalities: tuple[str, ...]
    role: str
    access: str
    stated_license: str
    evidence_status: str
    research_candidate: bool
    product_candidate: bool
    priority: int
    blockers: tuple[str, ...]
    evidence_path: str | None = None
    evidence_hash: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DatasetCandidate:
        return cls(
            dataset_id=str(value["dataset_id"]),
            name=str(value["name"]),
            source_url=str(value["source_url"]) if value.get("source_url") else None,
            modalities=tuple(str(item) for item in value["modalities"]),
            role=str(value["role"]),
            access=str(value["access"]),
            stated_license=str(value["stated_license"]),
            evidence_status=str(value["evidence_status"]),
            research_candidate=bool(value["research_candidate"]),
            product_candidate=bool(value["product_candidate"]),
            priority=int(value["priority"]),
            blockers=tuple(str(item) for item in value["blockers"]),
            evidence_path=str(value["evidence_path"]) if value.get("evidence_path") else None,
            evidence_hash=str(value["evidence_hash"]) if value.get("evidence_hash") else None,
        )

    def validate(self) -> None:
        if self.priority not in {1, 2, 3, 4}:
            raise ContractError(f"Invalid priority for {self.dataset_id}")
        if self.evidence_status not in {"unverified", "verified", "rejected"}:
            raise ContractError(f"Invalid evidence status for {self.dataset_id}")
        if self.product_candidate and self.evidence_status != "verified":
            raise ContractError(
                f"Unverified dataset cannot be a product candidate: {self.dataset_id}"
            )
        if self.evidence_status == "verified" and not (self.evidence_path and self.evidence_hash):
            raise ContractError(f"Verified dataset needs evidence path and hash: {self.dataset_id}")
        if not self.modalities or not self.blockers and self.evidence_status == "unverified":
            raise ContractError(
                f"Unverified dataset needs modalities and blockers: {self.dataset_id}"
            )


def audit_dataset_catalog(path: str | Path) -> dict[str, object]:
    catalog_path = Path(path).resolve()
    value = load_yaml(catalog_path)
    candidates = [DatasetCandidate.from_mapping(item) for item in value.get("datasets", [])]
    ids = [candidate.dataset_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ContractError("dataset_id must be unique")
    for candidate in candidates:
        candidate.validate()
        if candidate.evidence_status == "verified":
            evidence_path = Path(candidate.evidence_path or "")
            if not evidence_path.is_absolute():
                evidence_path = (catalog_path.parent / evidence_path).resolve()
            if not evidence_path.is_file():
                raise ContractError(f"Dataset evidence does not exist: {evidence_path}")
            if file_hash(evidence_path) != candidate.evidence_hash:
                raise ContractError(f"Dataset evidence hash mismatch: {candidate.dataset_id}")
    verified = [candidate for candidate in candidates if candidate.evidence_status == "verified"]
    acquired = [candidate for candidate in verified if not candidate.blockers]
    priority_research = [
        candidate.dataset_id
        for candidate in candidates
        if candidate.priority == 1 and candidate.research_candidate
    ]
    product = [candidate.dataset_id for candidate in candidates if candidate.product_candidate]
    roles: dict[str, list[str]] = {}
    for candidate in candidates:
        roles.setdefault(candidate.role, []).append(candidate.dataset_id)
    return {
        "catalog_version": value["catalog_version"],
        "source_document": value["source_document"],
        "fingerprint": stable_hash([asdict(candidate) for candidate in candidates]),
        "datasets": len(candidates),
        "verified_evidence": len(verified),
        "license_evidence_ready": bool(verified),
        "d0a_ready": bool(acquired),
        "acquired_and_unblocked": [candidate.dataset_id for candidate in acquired],
        "priority_research_candidates": priority_research,
        "product_candidates": product,
        "roles": roles,
        "blocked": {
            candidate.dataset_id: list(candidate.blockers)
            for candidate in candidates
            if candidate.blockers
        },
    }
