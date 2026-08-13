from pathlib import Path

import numpy as np
import pytest

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.errors import GateBlocked
from jepa_arkit.solver.mediapipe_face import (
    MediaPipeFaceSolver,
    SolvedMotion,
    _rotation_matrix_to_quaternion,
)

ROOT = Path(__file__).parents[1]


def test_rotation_matrix_identity_to_xyzw() -> None:
    quaternion = _rotation_matrix_to_quaternion(np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(quaternion, [0, 0, 0, 1], atol=1e-6)


def test_solver_requires_model_asset(tmp_path: Path) -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    with pytest.raises(GateBlocked, match="model asset"):
        MediaPipeFaceSolver(model_asset=tmp_path / "missing.task", schema=schema)


def test_explicit_policy_allows_zero_weight_degraded_curve(tmp_path: Path) -> None:
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    solver = MediaPipeFaceSolver(
        model_asset=ROOT / "data/models/face_landmarker.task", schema=schema
    )
    frame_count = 2
    motion = SolvedMotion(
        curves=np.zeros((frame_count, len(schema.curves)), dtype=np.float32),
        curve_names=schema.curve_names,
        head_quaternion=np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (frame_count, 1)),
        head_translation=np.zeros((frame_count, 3), dtype=np.float32),
        frame_confidence=np.ones(frame_count, dtype=np.float32),
        timestamps=np.arange(frame_count, dtype=np.float64) / schema.fps,
        missing_curves=("tongueOut",),
    )
    target = tmp_path / "motion.npz"
    policy = ROOT / "configs/contracts/mediapipe_missing_curve_policy_v1.json"
    solver.save(motion, target, missing_curve_policy=policy)
    with np.load(target) as archive:
        assert archive["label_policy_id"].item() == "mediapipe_face_landmarker_missing_curves_v1"
        assert archive["degraded_curves"].tolist() == ["tongueOut"]
