from __future__ import annotations

import html
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def build_failure_report(
    rows: Iterable[dict[str, object]],
    output: str | Path,
    *,
    primary_metric: str,
    largest_is_worst: bool = True,
    limit: int = 10,
) -> dict[str, object]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Failure report requires at least one metric row")
    if any(primary_metric not in row for row in materialized):
        raise KeyError(f"Missing primary metric: {primary_metric}")
    ordered = sorted(
        materialized,
        key=lambda row: float(row[primary_metric]),
        reverse=largest_is_worst,
    )
    worst = ordered[:limit]
    groups = {
        field: Counter(str(row.get(field, "unknown")) for row in worst)
        for field in ("source_id", "face_identity_id", "language", "recording_condition")
    }
    header = "".join(f"<th>{html.escape(str(key))}</th>" for key in worst[0])
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values())
        + "</tr>"
        for row in worst
    )
    group_html = "".join(
        f"<h2>{html.escape(field)}</h2><pre>{html.escape(str(dict(counts)))}</pre>"
        for field, counts in groups.items()
    )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Failure report</title>"
        f"<h1>Worst {len(worst)} samples by {html.escape(primary_metric)}</h1>"
        f"<table border='1'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        f"{group_html}"
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return {
        "rows": len(materialized),
        "worst": len(worst),
        "primary_metric": primary_metric,
        "group_counts": {field: dict(counts) for field, counts in groups.items()},
    }

