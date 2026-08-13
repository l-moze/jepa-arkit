from pathlib import Path

from jepa_arkit.failure_analysis import build_failure_report


def test_failure_report_ranks_worst_samples(tmp_path: Path) -> None:
    rows = [
        {"clip_id": "a", "av_sync_error": 0.1, "source_id": "x"},
        {"clip_id": "b", "av_sync_error": 0.9, "source_id": "y"},
    ]
    output = tmp_path / "failure_report.html"
    result = build_failure_report(rows, output, primary_metric="av_sync_error", limit=1)
    assert result["worst"] == 1
    assert "clip_id" in output.read_text(encoding="utf-8")

