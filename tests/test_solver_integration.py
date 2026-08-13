from pathlib import Path

import cv2
import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.solver import MediaPipeFaceSolver

ROOT = Path(__file__).parents[1]


def test_solver_no_face_video_preserves_30fps_timeline(tmp_path: Path) -> None:
    model = ROOT / "data/models/face_landmarker.task"
    if not model.is_file():
        return
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (64, 64),
    )
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    schema = CanonicalSchema.from_file(ROOT / "configs/contracts/canonical_arkit_v1.json")
    motion = MediaPipeFaceSolver(model_asset=model, schema=schema).solve_video(video)
    assert len(motion.timestamps) == 10
    assert np.all(motion.frame_confidence == 0)
    np.testing.assert_allclose(np.diff(motion.timestamps), 1 / 30, atol=1e-8)

