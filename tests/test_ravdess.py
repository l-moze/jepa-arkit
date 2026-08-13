from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from jepa_arkit.data.ravdess import ingest_ravdess
from jepa_arkit.errors import ContractError


def _make_archive(root: Path, actor: int, *, bad_actor: bool = False) -> tuple[str, str]:
    name = f"Video_Speech_Actor_{actor:02d}.zip"
    archive = root / name
    member_actor = 99 if bad_actor else actor
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            f"Actor_{member_actor:02d}/01-01-03-01-01-01-{member_actor:02d}.mp4",
            b"full-av",
        )
        bundle.writestr(
            f"Actor_{actor:02d}/02-01-03-01-01-01-{actor:02d}.mp4",
            b"video-only",
        )
    digest = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    return name, digest


def test_ingest_ravdess_extracts_only_full_av_and_writes_inventory(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    name, digest = _make_archive(archives, 1)
    output = tmp_path / "release"

    report = ingest_ravdess(
        archives,
        output,
        {name: digest},
        expected_actors=1,
        expected_av_clips_per_actor=1,
    )

    assert report["clips_extracted"] == 1
    assert report["ready_for_solver"] is True
    assert (output / "videos/Actor_01/01-01-03-01-01-01-01.mp4").read_bytes() == b"full-av"
    assert not (output / "videos/Actor_01/02-01-03-01-01-01-01.mp4").exists()
    inventory = (output / "inventory.jsonl").read_text(encoding="utf-8")
    assert '"emotion": "happy"' in inventory
    assert '"gender": "male"' in inventory


def test_ingest_ravdess_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    name, _ = _make_archive(archives, 1)
    with pytest.raises(ContractError, match="Checksum mismatch"):
        ingest_ravdess(
            archives,
            tmp_path / "release",
            {name: "0" * 32},
            expected_actors=1,
            expected_av_clips_per_actor=1,
        )


def test_ingest_ravdess_rejects_actor_mismatch(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    name, digest = _make_archive(archives, 1, bad_actor=True)
    with pytest.raises(ContractError, match="Actor mismatch"):
        ingest_ravdess(
            archives,
            tmp_path / "release",
            {name: digest},
            expected_actors=1,
            expected_av_clips_per_actor=1,
        )
