from __future__ import annotations

from pathlib import Path

import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.errors import ContractError


def load_and_validate_motion(path: str | Path, schema: CanonicalSchema) -> dict[str, np.ndarray]:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "curves",
                "curve_names",
                "head_quaternion",
                "head_translation",
                "timestamps",
                "frame_confidence",
            }
            missing = required - set(archive.files)
            if missing:
                raise ContractError(f"Missing motion arrays in {source}: {sorted(missing)}")
            values = {name: archive[name] for name in required}
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot read motion archive {source}: {exc}") from exc
    curve_names = [str(value) for value in values["curve_names"].tolist()]
    schema.validate_motion(
        curves=values["curves"],
        curve_names=curve_names,
        head_quaternion=values["head_quaternion"],
        head_translation=values["head_translation"],
        timestamps=values["timestamps"],
    )
    confidence = values["frame_confidence"]
    if confidence.shape != (values["curves"].shape[0],) or not np.all(
        (confidence >= 0) & (confidence <= 1)
    ):
        raise ContractError("frame_confidence must have shape [T] and values in [0, 1]")
    return values

