from pathlib import Path

import pytest

from jepa_arkit.contracts.rights import Track
from jepa_arkit.errors import GateBlocked
from jepa_arkit.io import dump_json
from jepa_arkit.training.dataset import MotionWindowDataset


def test_formal_dataset_rejects_d0a_report(tmp_path: Path) -> None:
    report = tmp_path / "audit.json"
    dump_json(
        report,
        {
            "gate": "D0A",
            "passed": True,
            "release_id": "candidate",
            "track": "research",
            "manifest_fingerprint": "x",
            "counts": {},
            "issues": [],
        },
    )
    with pytest.raises(GateBlocked, match="passed audit report"):
        MotionWindowDataset(
            manifest_path=tmp_path / "manifest.jsonl",
            audit_report_path=report,
            schema_path=tmp_path / "schema.json",
            feature_store_path=tmp_path / "features",
            split="train",
            frames=30,
            track=Track.RESEARCH,
        )
