from pathlib import Path

import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.inference import _postprocess_motion
from jepa_arkit.io import load_json

ROOT = Path(__file__).parents[1]


def test_postprocess_motion_enforces_canonical_ranges_and_quaternion() -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    normalization = load_json(ROOT / "data/real/ravdess_pilot_v1/motion_normalization.json")
    prediction = np.zeros((4, schema.motion_dim), dtype=np.float32)
    prediction[:, : len(schema.curves)] = 100
    prediction[:, len(schema.curves) + 3] = 1
    curves, quaternion, translation = _postprocess_motion(
        prediction, np.arange(4, dtype=np.float64) / 30, schema, normalization
    )
    assert curves.max() <= 1
    assert np.allclose(curves[:, schema.curve_names.index("tongueOut")], 0)
    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-6)
    assert translation.shape == (4, 3)
