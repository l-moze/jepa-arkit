from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from jepa_arkit.data.manifest import ManifestRecord, load_manifest
from jepa_arkit.io import dump_jsonl, stable_hash


@dataclass(frozen=True)
class WithdrawalPlan:
    withdrawal_key: str
    removed_clip_ids: tuple[str, ...]
    affected_feature_shards: tuple[str, ...]
    revoked_checkpoint_ids: tuple[str, ...]
    output_manifest_hash: str


def plan_withdrawal(
    manifest_path: str | Path,
    withdrawal_key: str,
    output_manifest: str | Path,
    feature_index: dict[str, str] | None = None,
    checkpoint_ancestry: dict[str, list[str]] | None = None,
) -> WithdrawalPlan:
    records = load_manifest(manifest_path)
    removed = [record for record in records if record.withdrawal_key == withdrawal_key]
    kept = [record for record in records if record.withdrawal_key != withdrawal_key]
    if not removed:
        raise KeyError(f"withdrawal_key not found: {withdrawal_key}")
    rows = []
    for record in kept:
        row = asdict(record)
        row["quality"] = asdict(record.quality)
        rows.append(row)
    dump_jsonl(output_manifest, rows)
    removed_ids = {record.clip_id for record in removed}
    shards = sorted({feature_index[clip] for clip in removed_ids if feature_index and clip in feature_index})
    checkpoints = sorted(
        checkpoint
        for checkpoint, ancestry in (checkpoint_ancestry or {}).items()
        if removed_ids.intersection(ancestry)
    )
    return WithdrawalPlan(
        withdrawal_key=withdrawal_key,
        removed_clip_ids=tuple(sorted(removed_ids)),
        affected_feature_shards=tuple(shards),
        revoked_checkpoint_ids=tuple(checkpoints),
        output_manifest_hash=stable_hash(rows),
    )

