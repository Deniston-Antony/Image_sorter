"""Per-face measurements for the people sorter.

Given a face box, :class:`FaceAnalyzer` runs MediaPipe's face landmarker on a
crop around it and reports how far the head is turned and how closed the eyes
are. It only *measures*; the good/bad decision lives in :mod:`quality_rules`.

:func:`match_faces_to_persons` pairs the face detector's boxes with the people
found by :mod:`person_detector`, so that a person with no face of their own
(turned back, bent down, face covered) can be recognised as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import cv2
import numpy as np

from detector import Face
from logger import get_logger
from person_detector import Person

_log = get_logger()

LANDMARKER_MODEL_FILENAME: Final[str] = "face_landmarker.task"
LANDMARKER_MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

#: Margins added around a face box before running the landmarker, as
#: fractions of the face size, tried in order until a mesh is found. The
#: landmarker has its own face detector, which is sensitive to how much of the
#: crop the face fills; retrying recovers close to half of the first misses.
_CROP_MARGINS: Final[tuple[float, ...]] = (0.6, 0.25, 1.0, 1.6)
#: Crops are scaled so their side is at least this many pixels.
_MIN_CROP_SIDE: Final[int] = 256

# Landmark indices for the eye aspect ratio: outer corner, two upper-lid
# points, inner corner, two lower-lid points.
_RIGHT_EYE: Final[tuple[int, ...]] = (33, 160, 158, 133, 153, 144)
_LEFT_EYE: Final[tuple[int, ...]] = (362, 385, 387, 263, 373, 380)


class FaceAnalyzerError(Exception):
    """Raised when the face landmarker cannot be created."""


@dataclass(slots=True)
class FaceMetrics:
    """Measurements for one face.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in source-image pixels.
        confidence: Face detector score.
        size: Shorter side of the face box in pixels.
        mesh_found: Whether the landmarker could fit a face mesh. The fields
            below are only meaningful when this is ``True``.
        yaw: Left/right head turn in degrees; 0 is facing the camera.
        pitch: Up/down head tilt in degrees; positive is looking down.
        roll: In-plane head tilt in degrees.
        blink_left: Eye-closed score for the left eye in ``[0, 1]``.
        blink_right: Eye-closed score for the right eye in ``[0, 1]``.
        eye_openness: Eye aspect ratio (lid gap over eye width) of the more
            open eye, so a wink does not read as closed eyes. Open eyes are
            around 0.3; closed eyes fall below about 0.09.
    """

    box: tuple[float, float, float, float]
    confidence: float
    size: float
    mesh_found: bool = False
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    blink_left: float = 0.0
    blink_right: float = 0.0
    eye_openness: float = 0.0


def _eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    """Return the eye aspect ratio (lid gap over eye width) for one eye."""
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    width = float(np.linalg.norm(p1 - p4))
    if width <= 1e-6:
        return 0.0
    gap = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
    return gap / (2.0 * width)


def head_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a face transformation matrix into ``(yaw, pitch, roll)`` degrees.

    Yaw is positive when the head turns towards the image's right, pitch is
    positive when the head tilts down.
    """
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    # Strip the scale the matrix may carry so the angles are well defined.
    rotation = rotation / np.linalg.norm(rotation, axis=0, keepdims=True)
    yaw = math.degrees(math.asin(float(np.clip(rotation[0, 2], -1.0, 1.0))))
    pitch = math.degrees(math.atan2(-rotation[1, 2], rotation[2, 2]))
    roll = math.degrees(math.atan2(-rotation[0, 1], rotation[0, 0]))
    return yaw, pitch, roll


class FaceAnalyzer:
    """Measures head pose and eye state with MediaPipe's face landmarker."""

    def __init__(self, model_path: Path) -> None:
        """Load the landmarker model.

        Raises:
            FaceAnalyzerError: If MediaPipe is missing, the model file is
                absent or the landmarker cannot be created.
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as error:
            raise FaceAnalyzerError(
                "The 'mediapipe' package is required. "
                "Install it with: pip install -r requirements.txt"
            ) from error

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FaceAnalyzerError(
                f"Face landmark model not found: {model_path}\n"
                "Run 'python people_sorter.py --download-models' to fetch it."
            )

        self._mp = mp
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    # CPU is forced: the Metal path aborts the whole process
                    # on macOS when no GPU service is available.
                    base_options=BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
            )
        except Exception as error:  # noqa: BLE001 - surfaced with context
            raise FaceAnalyzerError(
                f"Could not create the face landmarker: {error}"
            ) from error

    def close(self) -> None:
        """Release the landmarker's native resources."""
        self._landmarker.close()

    def measure(self, image: np.ndarray, face: Face) -> FaceMetrics:
        """Measure one face in a BGR image."""
        metrics = FaceMetrics(
            box=tuple(round(float(v), 1) for v in face[:4]),
            confidence=round(float(face.confidence), 3),
            size=round(min(face.width, face.height), 1),
        )
        result, rgb = None, None
        for margin in _CROP_MARGINS:
            crop = self._crop(image, face, margin)
            if crop is None:
                continue
            rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            try:
                result = self._landmarker.detect(
                    self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                )
            except Exception as error:  # noqa: BLE001 - one bad crop is not fatal
                _log.debug("Face landmarker failed on a crop: %s", error)
                continue
            if result.face_landmarks:
                break

        if result is None or not result.face_landmarks:
            return metrics

        metrics.mesh_found = True
        height, width = rgb.shape[:2]
        points = np.array(
            [[p.x * width, p.y * height] for p in result.face_landmarks[0]],
            dtype=np.float32,
        )
        metrics.eye_openness = round(
            max(_eye_aspect_ratio(points, _LEFT_EYE), _eye_aspect_ratio(points, _RIGHT_EYE)),
            3,
        )

        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            metrics.blink_left = round(float(scores.get("eyeBlinkLeft", 0.0)), 3)
            metrics.blink_right = round(float(scores.get("eyeBlinkRight", 0.0)), 3)

        if result.facial_transformation_matrixes:
            yaw, pitch, roll = head_angles(result.facial_transformation_matrixes[0])
            metrics.yaw, metrics.pitch, metrics.roll = (
                round(yaw, 1),
                round(pitch, 1),
                round(roll, 1),
            )
        return metrics

    @staticmethod
    def _crop(image: np.ndarray, face: Face, margin: float) -> np.ndarray | None:
        """Cut a square, padded region around a face, upscaled if tiny."""
        height, width = image.shape[:2]
        side = max(face.width, face.height) * (1.0 + 2.0 * margin)
        cx, cy = (face.x1 + face.x2) / 2.0, (face.y1 + face.y2) / 2.0
        x1, y1 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
        x2, y2 = int(round(cx + side / 2.0)), int(round(cy + side / 2.0))

        # Pad instead of clipping so the face stays centred near image edges.
        pad = max(0, -x1, -y1, x2 - width, y2 - height)
        if pad:
            image = cv2.copyMakeBorder(
                image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.shape[0] < _MIN_CROP_SIDE:
            crop = cv2.resize(
                crop, (_MIN_CROP_SIDE, _MIN_CROP_SIDE), interpolation=cv2.INTER_CUBIC
            )
        return crop


def match_faces_to_persons(
    persons: Sequence[Person], faces: Sequence[Face]
) -> dict[int, Face]:
    """Assign each person at most one face.

    A face can belong to a person only if its centre lies inside the person's
    box. Among the candidates, the face closest to the person's head keypoints
    wins; without head keypoints, the face highest in the box does.

    Returns:
        A mapping from index in ``persons`` to the matched face.
    """
    candidates: list[tuple[float, int, int]] = []
    for person_index, person in enumerate(persons):
        x1, y1, x2, y2 = person.box
        head = person.head_center()
        for face_index, face in enumerate(faces):
            cx, cy = (face.x1 + face.x2) / 2.0, (face.y1 + face.y2) / 2.0
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            face_size = max(face.width, face.height, 1.0)
            if head is not None:
                distance = math.hypot(cx - head[0], cy - head[1]) / face_size
                if distance > 1.5:
                    continue
            else:
                # No head keypoints: fall back to "the face near the top".
                distance = 2.0 + (cy - y1) / max(person.height, 1.0)
            candidates.append((distance, person_index, face_index))

    matches: dict[int, Face] = {}
    used_faces: set[int] = set()
    for _, person_index, face_index in sorted(candidates):
        if person_index in matches or face_index in used_faces:
            continue
        matches[person_index] = faces[face_index]
        used_faces.add(face_index)
    return matches
