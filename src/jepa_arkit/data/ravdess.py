from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from jepa_arkit.errors import ContractError
from jepa_arkit.io import dump_json, dump_jsonl, file_hash

ARCHIVE_PATTERN = re.compile(r"^Video_Speech_Actor_(?P<actor>\d{2})\.zip$")
MEMBER_PATTERN = re.compile(
    r"^Actor_(?P<directory_actor>\d{2})/"
    r"(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.mp4$"
)
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
INTENSITIES = {"01": "normal", "02": "strong"}
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


@dataclass(frozen=True)
class RavdessClip:
    clip_id: str
    source_id: str
    source_record: str
    source_archive: str
    source_archive_md5: str
    source_member: str
    video_path: str
    video_sha256: str
    video_bytes: int
    actor_id: str
    gender: str
    emotion: str
    intensity: str
    statement: str
    repetition: int
    language: str
    rights_profile_id: str
    withdrawal_key: str


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_member(archive_actor: str, member: str) -> dict[str, str] | None:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"Unsafe RAVDESS archive member: {member}")
    match = MEMBER_PATTERN.fullmatch(member)
    if not match:
        if member.endswith("/"):
            return None
        raise ContractError(f"Unexpected RAVDESS archive member: {member}")
    values = match.groupdict()
    if values["actor"] != archive_actor or values["directory_actor"] != archive_actor:
        raise ContractError(f"Actor mismatch in archive member: {member}")
    if values["channel"] != "01":
        raise ContractError(f"Non-speech member in speech archive: {member}")
    return values


def ingest_ravdess(
    archives_dir: str | Path,
    output_dir: str | Path,
    official_checksums: dict[str, str],
    *,
    expected_actors: int = 24,
    expected_av_clips_per_actor: int = 60,
) -> dict[str, object]:
    archives_root = Path(archives_dir).resolve()
    output_root = Path(output_dir).resolve()
    videos_root = output_root / "videos"
    archives = sorted(archives_root.glob("Video_Speech_Actor_*.zip"))
    if len(archives) != expected_actors:
        raise ContractError(f"Expected {expected_actors} RAVDESS archives, found {len(archives)}")

    clips: list[RavdessClip] = []
    archive_inventory: list[dict[str, object]] = []
    seen_clip_ids: set[str] = set()
    for archive in archives:
        archive_match = ARCHIVE_PATTERN.fullmatch(archive.name)
        if not archive_match:
            raise ContractError(f"Unexpected RAVDESS archive name: {archive.name}")
        archive_actor = archive_match.group("actor")
        expected_md5 = official_checksums.get(archive.name, "").lower()
        if not expected_md5:
            raise ContractError(f"Missing official checksum for {archive.name}")
        actual_md5 = _md5(archive)
        if actual_md5 != expected_md5:
            raise ContractError(
                f"Checksum mismatch for {archive.name}: {actual_md5} != {expected_md5}"
            )

        actor_clips = 0
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                values = _parse_member(archive_actor, info.filename)
                if values is None or values["modality"] != "01":
                    continue
                clip_id = Path(info.filename).stem
                if clip_id in seen_clip_ids:
                    raise ContractError(f"Duplicate RAVDESS clip: {clip_id}")
                if values["emotion"] not in EMOTIONS:
                    raise ContractError(f"Unknown emotion in {info.filename}")
                if values["intensity"] not in INTENSITIES:
                    raise ContractError(f"Unknown intensity in {info.filename}")
                if values["statement"] not in STATEMENTS:
                    raise ContractError(f"Unknown statement in {info.filename}")

                relative_video = Path(f"Actor_{archive_actor}") / f"{clip_id}.mp4"
                target = videos_root / relative_video
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if target.stat().st_size != info.file_size:
                    raise ContractError(f"Extracted size mismatch for {info.filename}")

                actor_number = int(archive_actor)
                seen_clip_ids.add(clip_id)
                actor_clips += 1
                clips.append(
                    RavdessClip(
                        clip_id=clip_id,
                        source_id="ravdess_v1_0_0",
                        source_record="https://doi.org/10.5281/zenodo.1188976",
                        source_archive=archive.name,
                        source_archive_md5=actual_md5,
                        source_member=info.filename,
                        video_path=relative_video.as_posix(),
                        video_sha256=file_hash(target),
                        video_bytes=target.stat().st_size,
                        actor_id=f"ravdess_actor_{archive_actor}",
                        gender="male" if actor_number % 2 else "female",
                        emotion=EMOTIONS[values["emotion"]],
                        intensity=INTENSITIES[values["intensity"]],
                        statement=STATEMENTS[values["statement"]],
                        repetition=int(values["repetition"]),
                        language="en-US",
                        rights_profile_id="ravdess_cc_by_nc_sa_4_0_research_v1",
                        withdrawal_key=f"ravdess_v1_0_0:{clip_id}",
                    )
                )
        if actor_clips != expected_av_clips_per_actor:
            raise ContractError(
                f"Expected {expected_av_clips_per_actor} AV clips for actor {archive_actor}, "
                f"found {actor_clips}"
            )
        archive_inventory.append(
            {
                "archive": archive.name,
                "bytes": archive.stat().st_size,
                "md5": actual_md5,
                "full_av_clips": actor_clips,
            }
        )

    expected_clips = expected_actors * expected_av_clips_per_actor
    if len(clips) != expected_clips:
        raise ContractError(f"Expected {expected_clips} AV clips, found {len(clips)}")
    clips.sort(key=lambda clip: clip.clip_id)
    dump_jsonl(output_root / "inventory.jsonl", (asdict(clip) for clip in clips))
    report: dict[str, object] = {
        "dataset_id": "ravdess",
        "source_id": "ravdess_v1_0_0",
        "source_record": "https://doi.org/10.5281/zenodo.1188976",
        "license": "CC-BY-NC-SA-4.0",
        "track": "research_only",
        "archives_verified": len(archives),
        "archive_bytes": sum(int(item["bytes"]) for item in archive_inventory),
        "clips_extracted": len(clips),
        "video_bytes": sum(clip.video_bytes for clip in clips),
        "actors": len({clip.actor_id for clip in clips}),
        "emotions": sorted({clip.emotion for clip in clips}),
        "archive_inventory": archive_inventory,
        "inventory_path": "inventory.jsonl",
        "ready_for_solver": True,
        "ready_for_training": False,
    }
    dump_json(output_root / "release.json", report)
    return report

