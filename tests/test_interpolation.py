import numpy as np
import pytest

from jepa_arkit.errors import ContractError
from jepa_arkit.interpolation import interpolate_motion, round_trip_metrics


def _motion(frames: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = np.linspace(0.0, 0.8, frames * 2, dtype=np.float32).reshape(frames, 2)
    translation = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    angle = np.linspace(0.0, np.pi / 2.0, frames)
    quaternion = np.stack(
        [np.zeros(frames), np.zeros(frames), np.sin(angle / 2.0), np.cos(angle / 2.0)], axis=1
    ).astype(np.float32)
    timestamps = np.arange(frames, dtype=np.float64) / 30.0
    return curves, quaternion, translation, timestamps


def test_interpolation_preserves_endpoints_and_round_trip() -> None:
    curves, quaternion, translation, timestamps = _motion()
    result = interpolate_motion(curves, quaternion, translation, timestamps)
    assert result.curves.shape == (7, 2)
    assert np.array_equal(result.timestamps, np.arange(7) / 60.0)
    assert np.allclose(result.curves[[0, -1]], curves[[0, -1]], atol=1e-7)
    assert np.allclose(result.head_translation[[0, -1]], translation[[0, -1]], atol=1e-7)
    assert np.allclose(result.head_quaternion[[0, -1]], quaternion[[0, -1]], atol=1e-7)
    assert np.allclose(np.linalg.norm(result.head_quaternion, axis=1), 1.0, atol=1e-6)
    metrics = round_trip_metrics(curves, quaternion, translation, result)
    assert metrics["passed"] is True
    assert metrics["target_frames"] == 7


def test_interpolation_uses_shortest_quaternion_path() -> None:
    curves, quaternion, translation, timestamps = _motion(2)
    quaternion[1] = -quaternion[0]
    result = interpolate_motion(curves, quaternion, translation, timestamps)
    assert np.allclose(result.head_quaternion[1], result.head_quaternion[0], atol=1e-6)


def test_interpolation_rejects_nonuniform_timestamps() -> None:
    curves, quaternion, translation, timestamps = _motion()
    timestamps[2] += 0.01
    with pytest.raises(ContractError, match="uniformly"):
        interpolate_motion(curves, quaternion, translation, timestamps)
