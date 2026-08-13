from pathlib import Path

import numpy as np

from jepa_arkit.contracts.rights import Track
from jepa_arkit.features.store import FeatureMetadata, FeatureStore, align_features_to_motion
from jepa_arkit.features.wavlm import wavlm_frame_timestamps


def test_feature_store_is_sample_addressable(tmp_path: Path) -> None:
    metadata = FeatureMetadata(
        feature_release_id="wavlm_test",
        model_id="test",
        model_revision="sha256:test",
        layer="last",
        frame_hz=50,
        feature_dim=4,
        dtype="float16",
        normalization="none",
        track=Track.RESEARCH,
        source_data_release_id="synthetic",
    )
    store = FeatureStore.create(tmp_path / "features", metadata)
    timestamps = np.arange(5, dtype=np.float64) / 50
    features = np.arange(20, dtype=np.float32).reshape(5, 4)
    store.write("clip/a", "withdraw/a", features, timestamps)
    restored, restored_timestamps = store.read("clip/a")
    np.testing.assert_allclose(restored, features)
    np.testing.assert_allclose(restored_timestamps, timestamps)
    assert store.affected_by_withdrawal("withdraw/a") == ("clip/a",)


def test_feature_alignment_uses_nearest_timestamp() -> None:
    features = np.asarray([[0], [1], [2]], dtype=np.float32)
    feature_time = np.asarray([0.0, 0.02, 0.04])
    motion_time = np.asarray([0.0, 1 / 30, 2 / 30])
    aligned = align_features_to_motion(features, feature_time, motion_time)
    np.testing.assert_array_equal(aligned[:, 0], [0, 2, 2])


def test_feature_store_batch_write_is_sample_addressable(tmp_path: Path) -> None:
    metadata = FeatureMetadata(
        feature_release_id="batch_test",
        model_id="test",
        model_revision="revision",
        layer="last",
        frame_hz=50,
        feature_dim=2,
        dtype="float16",
        normalization="none",
        track=Track.RESEARCH,
        source_data_release_id="source",
    )
    store = FeatureStore.create(tmp_path / "features", metadata)
    store.write_batch(
        [
            (
                "clip-a",
                "withdraw-a",
                np.ones((3, 2), dtype=np.float32),
                np.arange(3, dtype=np.float64) / 50,
            ),
            (
                "clip-b",
                "withdraw-b",
                np.zeros((2, 2), dtype=np.float32),
                np.arange(2, dtype=np.float64) / 50,
            ),
        ]
    )
    assert store.read("clip-a")[0].shape == (3, 2)
    assert store.affected_by_withdrawal("withdraw-b") == ("clip-b",)


def test_wavlm_timestamps_use_convolution_receptive_field_centers() -> None:
    timestamps = wavlm_frame_timestamps(
        3,
        sample_rate=16_000,
        convolution_kernels=(10, 3, 3, 3, 3, 2, 2),
        convolution_strides=(5, 2, 2, 2, 2, 2, 2),
    )
    np.testing.assert_allclose(timestamps, [199.5 / 16_000, 519.5 / 16_000, 839.5 / 16_000])
