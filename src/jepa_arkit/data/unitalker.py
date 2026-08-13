from __future__ import annotations

import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from jepa_arkit.io import dump_json, file_hash

SOURCE_RIGHTS = {
    "D0_BIWI": {
        "source": "BIWI B3D(AC)2",
        "status": "blocked",
        "reason": "upstream terms and redistribution authority not verified",
    },
    "D1_vocaset": {
        "source": "VOCASET",
        "status": "blocked",
        "reason": "research-only application terms; derived redistribution authority unverified",
    },
    "D2_meshtalk": {
        "source": "MeshTalk / Multiface",
        "status": "blocked",
        "reason": "upstream dataset terms and subject consent scope not verified",
    },
    "D3_HDTF": {
        "source": "HDTF via 3DETF",
        "status": "blocked",
        "reason": "source-media and derived annotation rights not verified",
    },
    "D4_RAVDESS": {
        "source": "RAVDESS via 3DETF",
        "status": "blocked",
        "reason": "derived labels lack a verified pipeline and release ancestry",
    },
    "D5_unitalker_faceforensics++": {
        "source": "FaceForensics++",
        "status": "blocked",
        "reason": "application terms and original source-media rights require review",
    },
    "D6_unitalker_Chinese_speech": {
        "source": "UniTalker in-house Chinese speech",
        "status": "blocked",
        "reason": "no participant consent or dataset license accompanies the archive",
    },
    "D7_unitalker_song": {
        "source": "UniTalker in-house song",
        "status": "blocked",
        "reason": "no participant, recording, or music-rights license accompanies the archive",
    },
}


def _source_key(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 2:
        return "unknown"
    if parts[1] == "D3D4_3DETF" and len(parts) > 2:
        return parts[2]
    return parts[1]


def _load_npy_header(archive: zipfile.ZipFile, name: str) -> dict[str, object]:
    value = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
    return {
        "path": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "minimum": float(np.nanmin(value)),
        "maximum": float(np.nanmax(value)),
    }


def audit_unitalker_candidate(
    archive_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Audit the mixed-source UniTalker archive without releasing it for training."""
    archive_path = Path(archive_path).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        bad_member = archive.testzip()
        split_files = sorted(name for name in names if name.endswith(".json"))
        source_stats: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "split_files": 0,
                "records": 0,
                "duration_seconds": 0.0,
                "identities": set(),
                "annotation_types": Counter(),
                "fps": Counter(),
                "missing_audio": 0,
                "missing_annotation": 0,
                "sample_annotation": None,
            }
        )
        for split_file in split_files:
            payload = json.loads(archive.read(split_file))
            source = _source_key(split_file)
            stats = source_stats[source]
            stats["split_files"] += 1
            info = payload.get("info", {})
            stats["duration_seconds"] += float(info.get("total_duration", 0.0))
            stats["identities"].update(str(value) for value in info.get("id_list", []))
            base = split_file.rsplit("/", 1)[0]
            for record in payload.get("data", []):
                stats["records"] += 1
                stats["annotation_types"][str(record.get("annot_type", "unknown"))] += 1
                stats["fps"][str(record.get("fps", "unknown"))] += 1
                audio_name = f"{base}/{record.get('audio_path', '')}"
                annotation_name = f"{base}/{record.get('annot_path', '')}"
                if audio_name not in names:
                    stats["missing_audio"] += 1
                if annotation_name not in names:
                    stats["missing_annotation"] += 1
                elif stats["sample_annotation"] is None:
                    stats["sample_annotation"] = _load_npy_header(archive, annotation_name)
        serializable_sources: dict[str, object] = {}
        for source, stats in sorted(source_stats.items()):
            rights = SOURCE_RIGHTS.get(
                source,
                {"source": source, "status": "blocked", "reason": "unregistered source"},
            )
            serializable_sources[source] = {
                **rights,
                **stats,
                "identities": sorted(stats["identities"]),
                "identity_count": len(stats["identities"]),
                "annotation_types": dict(stats["annotation_types"]),
                "fps": dict(stats["fps"]),
            }
    report = {
        "dataset_id": "unitalker_released_v1_candidate",
        "status": "quarantined_rights_review",
        "ready_for_training": False,
        "ready_for_evaluation": False,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": file_hash(archive_path),
        "zip_crc_passed": bad_member is None,
        "first_bad_member": bad_member,
        "entries": len(names),
        "split_files": len(split_files),
        "sources": serializable_sources,
        "blockers": [
            "verify every upstream license and redistribution authority",
            "preserve source-specific rights profiles and release ancestry",
            "define each annotation-space to canonical-motion conversion",
            "create identity/source-disjoint splits after authorization",
        ],
    }
    dump_json(output_path, report)
    return report

