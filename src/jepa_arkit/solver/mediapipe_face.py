from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jepa_arkit.contracts.canonical import CanonicalSchema
from jepa_arkit.errors import ContractError, GateBlocked
from jepa_arkit.io import file_hash, load_json


@dataclass(frozen=True)
class SolvedMotion:
    curves: np.ndarray
    curve_names: tuple[str, ...]
    head_quaternion: np.ndarray
    head_translation: np.ndarray
    frame_confidence: np.ndarray
    timestamps: np.ndarray
    missing_curves: tuple[str, ...]


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    trace = np.trace(rotation)
    quaternion = np.empty(4, dtype=np.float32)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion[3] = 0.25 * scale
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[1] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[2] = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion[:] = [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ]
        elif axis == 1:
            scale = np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion[:] = [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ]
        else:
            scale = np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion[:] = [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
    return quaternion / np.linalg.norm(quaternion).clip(min=1e-8)


class MediaPipeFaceSolver:
    """Offline video solver using MediaPipe's version-pinned Face Landmarker task."""

    def __init__(
        self,
        *,
        model_asset: str | Path,
        schema: CanonicalSchema,
        minimum_presence_confidence: float = 0.5,
    ) -> None:
        self.model_asset = Path(model_asset).resolve()
        self.schema = schema
        self.minimum_presence_confidence = minimum_presence_confidence
        if not self.model_asset.is_file():
            raise GateBlocked(f"MediaPipe model asset is missing: {self.model_asset}")

    def solve_video(self, video_path: str | Path) -> SolvedMotion:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise GateBlocked("Install the solver extra: uv sync --extra solver") from exc
        source = Path(video_path).resolve()
        if not source.is_file():
            raise GateBlocked(f"Input video is missing: {source}")
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_asset))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_presence_confidence=self.minimum_presence_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ContractError(f"OpenCV cannot open video: {source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            capture.release()
            raise ContractError("Video FPS is unavailable")
        frame_period_ms = 1000.0 / self.schema.fps
        next_output_ms = 0.0
        curve_rows: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []
        confidences: list[float] = []
        timestamps: list[float] = []
        observed_names: set[str] = set()
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                source_ms = frame_index * 1000.0 / source_fps
                frame_index += 1
                if source_ms + 1e-6 < next_output_ms:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(round(source_ms)))
                if not result.face_blendshapes or not result.facial_transformation_matrixes:
                    curve_rows.append(np.zeros(len(self.schema.curves), dtype=np.float32))
                    rotations.append(np.asarray([0, 0, 0, 1], dtype=np.float32))
                    translations.append(np.zeros(3, dtype=np.float32))
                    confidences.append(0.0)
                else:
                    scores = {
                        category.category_name: float(category.score)
                        for category in result.face_blendshapes[0]
                        if category.category_name != "_neutral"
                    }
                    observed_names.update(scores)
                    curve_rows.append(
                        np.asarray([scores.get(name, 0.0) for name in self.schema.curve_names])
                    )
                    transform = np.asarray(result.facial_transformation_matrixes[0])
                    rotations.append(_rotation_matrix_to_quaternion(transform))
                    translations.append(transform[:3, 3].astype(np.float32))
                    confidences.append(1.0)
                timestamps.append(next_output_ms / 1000.0)
                next_output_ms += frame_period_ms
        capture.release()
        if not curve_rows:
            raise ContractError("Video produced no output frames")
        missing = tuple(sorted(set(self.schema.curve_names) - observed_names))
        return SolvedMotion(
            curves=np.stack(curve_rows).astype(np.float32),
            curve_names=self.schema.curve_names,
            head_quaternion=np.stack(rotations).astype(np.float32),
            head_translation=np.stack(translations).astype(np.float32),
            frame_confidence=np.asarray(confidences, dtype=np.float32),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            missing_curves=missing,
        )

    def save(
        self,
        motion: SolvedMotion,
        output_path: str | Path,
        *,
        missing_curve_policy: str | Path | None = None,
    ) -> None:
        policy_id = "none"
        policy_hash = ""
        degraded_curves: tuple[str, ...] = ()
        if motion.missing_curves:
            if missing_curve_policy is None:
                raise GateBlocked(
                    "Solver does not provide every canonical curve; "
                    "define an explicit label policy: "
                    + ", ".join(motion.missing_curves)
                )
            policy_path = Path(missing_curve_policy).resolve()
            policy = load_json(policy_path)
            configured = policy.get("allowed_missing_curves", {})
            if not isinstance(configured, dict):
                raise ContractError("allowed_missing_curves must be an object")
            unknown = set(motion.missing_curves) - configured.keys()
            if unknown:
                raise GateBlocked(
                    "Missing-curve policy does not cover: " + ", ".join(sorted(unknown))
                )
            for curve in motion.missing_curves:
                rule = configured[curve]
                if not isinstance(rule, dict) or rule.get("fill") != "zero":
                    raise ContractError(f"Unsupported missing-curve policy for {curve}")
                if float(rule.get("supervision_weight", -1)) != 0.0:
                    raise ContractError(f"Degraded curve {curve} must have zero supervision weight")
            policy_id = str(policy["policy_id"])
            policy_hash = file_hash(policy_path)
            degraded_curves = motion.missing_curves
        self.schema.validate_motion(
            motion.curves,
            motion.curve_names,
            motion.head_quaternion,
            motion.head_translation,
            motion.timestamps,
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            curves=motion.curves,
            curve_names=np.asarray(motion.curve_names),
            head_quaternion=motion.head_quaternion,
            head_translation=motion.head_translation,
            frame_confidence=motion.frame_confidence,
            timestamps=motion.timestamps,
            label_policy_id=np.asarray(policy_id),
            label_policy_sha256=np.asarray(policy_hash),
            degraded_curves=np.asarray(degraded_curves),
        )
