from pathlib import Path

from jepa_arkit.data.catalog import audit_dataset_catalog

ROOT = Path(__file__).parents[1]


def test_dataset_catalog_is_structurally_valid_and_blocked() -> None:
    report = audit_dataset_catalog(ROOT / "configs/data/dataset_catalog.yaml")
    assert report["datasets"] == 14
    assert report["verified_evidence"] == 3
    assert report["license_evidence_ready"] is True
    assert report["d0a_ready"] is False
    assert "vocaset" in report["priority_research_candidates"]
    assert "ravdess" in report["priority_research_candidates"]
    assert report["product_candidates"] == []
