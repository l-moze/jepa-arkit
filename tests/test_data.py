from pathlib import Path

from jepa_arkit.data.audit import audit_release
from jepa_arkit.data.withdrawal import plan_withdrawal
from jepa_arkit.demo import create_demo_dataset

ROOT = Path(__file__).parents[1]


def _write_config(path: Path, data: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "release_id: synthetic_candidate_v1",
                "gate: D0A",
                "track: research",
                f"manifest: {data / 'manifest.jsonl'}",
                f"canonical_schema: {ROOT / 'configs/contracts/canonical_arkit_v1.json'}",
                f"rights_registry: {data / 'rights_registry.json'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_demo_release_passes_d0a(tmp_path: Path) -> None:
    data = tmp_path / "demo"
    create_demo_dataset(data, ROOT / "configs/contracts/canonical_arkit_v1.json")
    config = tmp_path / "release.yaml"
    _write_config(config, data)
    report = audit_release(config)
    assert report.passed, report.issues
    assert report.counts["records"] == 25


def test_withdrawal_identifies_shards_and_checkpoints(tmp_path: Path) -> None:
    data = tmp_path / "demo"
    create_demo_dataset(data, ROOT / "configs/contracts/canonical_arkit_v1.json")
    plan = plan_withdrawal(
        data / "manifest.jsonl",
        "withdrawal_000",
        data / "manifest_after.jsonl",
        feature_index={"synthetic/identity_000/clip_000": "shard_00"},
        checkpoint_ancestry={"checkpoint_a": ["synthetic/identity_000/clip_000"]},
    )
    assert plan.affected_feature_shards == ("shard_00",)
    assert plan.revoked_checkpoint_ids == ("checkpoint_a",)

