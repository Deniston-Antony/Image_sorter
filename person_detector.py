"""Person detection for the people sorter.

Faces alone cannot answer "is there a human in this photo?": someone turned
away, bent over or lying down has no detectable face yet is clearly a person.
This module therefore finds *bodies* with a YOLO pose model, which also yields
head keypoints (nose, eyes, ears) that :mod:`quality_rules` uses to tell a
turned-away head from a face the face detector merely missed.

Background bystanders are separated from the people the photo is actually
about by :func:`mark_main_subjects`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from logger import get_logger

_log = get_logger()

#: Indices into the 17 COCO keypoints returned by the pose model.
KP_NOSE: Final[int] = 0
KP_LEFT_EYE: Final[int] = 1
KP_RIGHT_EYE: Final[int] = 2
KP_LEFT_EAR: Final[int] = 3
KP_RIGHT_EAR: Final[int] = 4
KP_LEFT_SHOULDER: Final[int] = 5
KP_RIGHT_SHOULDER: Final[int] = 6
HEAD_KEYPOINTS: Final[tuple[int, ...]] = (
    KP_NOSE,
    KP_LEFT_EYE,
    KP_RIGHT_EYE,
    KP_LEFT_EAR,
    KP_RIGHT_EAR,
)

POSE_MODEL_FILENAME: Final[str] = "yolo11m-pose.pt"
_PERSON_CLASS: Final[int] = 0


class PersonDetectorError(Exception):
    """Raised when the person detector cannot be created or run."""


@dataclass(slots=True)
class Person:
    """One detected person.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in source-image pixels.
        confidence: Detector score in ``[0.0, 1.0]``.
        keypoints: 17 COCO keypoints as ``[x, y, confidence]`` rows.
        is_main: Whether the person is a main subject rather than a bystander.
    """

    box: tuple[float, float, float, float]
    confidence: float
    keypoints: list[list[float]] = field(default_factory=list)
    is_main: bool = False

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def keypoint_confidence(self, index: int) -> float:
        """Return the confidence of one keypoint, or 0.0 if it is missing."""
        if index >= len(self.keypoints):
            return 0.0
        return float(self.keypoints[index][2])

    def head_center(self, min_confidence: float = 0.3) -> tuple[float, float] | None:
        """Return the mean position of the visible head keypoints, if any."""
        points = [
            self.keypoints[i][:2]
            for i in HEAD_KEYPOINTS
            if self.keypoint_confidence(i) >= min_confidence
        ]
        if not points:
            return None
        mean = np.mean(np.asarray(points, dtype=np.float32), axis=0)
        return float(mean[0]), float(mean[1])

    def estimated_head_size(self) -> float:
        """Roughly estimate the head width in pixels from the body.

        Used only to decide whether a face would be too small to judge.
        """
        left = self.keypoint_confidence(KP_LEFT_SHOULDER)
        right = self.keypoint_confidence(KP_RIGHT_SHOULDER)
        if left >= 0.5 and right >= 0.5:
            dx = self.keypoints[KP_LEFT_SHOULDER][0] - self.keypoints[KP_RIGHT_SHOULDER][0]
            dy = self.keypoints[KP_LEFT_SHOULDER][1] - self.keypoints[KP_RIGHT_SHOULDER][1]
            shoulder_width = float(np.hypot(dx, dy))
            # Shoulders seen side-on collapse to nothing, so keep a floor.
            return max(0.55 * shoulder_width, 0.12 * self.height)
        return min(0.5 * self.width, 0.2 * self.height)


def resolve_torch_device(requested: str = "auto") -> str:
    """Return the torch device to use: ``mps``, ``cuda`` or ``cpu``."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mark_main_subjects(persons: list[Person], *, area_ratio: float = 0.25) -> None:
    """Flag the people a photo is about, in place.

    The largest person is always a main subject. Anyone else counts when their
    box is at least ``area_ratio`` of the largest one, which keeps every row of
    a group photo while dropping passers-by in the background.
    """
    if not persons:
        return
    largest = max(person.area for person in persons)
    for person in persons:
        person.is_main = largest > 0 and person.area >= area_ratio * largest


class PersonDetector:
    """Detects people and their pose keypoints with a YOLO pose model."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.4,
        image_size: int = 640,
        device: str = "auto",
    ) -> None:
        """Load the pose model.

        Args:
            model_path: Weights file. Downloaded by Ultralytics on first use
                if it does not exist yet.
            confidence_threshold: Minimum score for a person to count.
            image_size: Inference size (long side) in pixels.
            device: ``auto``, ``mps``, ``cuda`` or ``cpu``.

        Raises:
            PersonDetectorError: If Ultralytics is missing or the model fails
                to load.
        """
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise PersonDetectorError(
                "The 'ultralytics' package is required. "
                "Install it with: pip install -r requirements.txt"
            ) from error

        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = resolve_torch_device(device)
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self._model = YOLO(str(model_path))
        except Exception as error:  # noqa: BLE001 - surfaced with context
            raise PersonDetectorError(
                f"Could not load pose model {model_path}: {error}"
            ) from error
        _log.info(
            "Person detector ready: model=%s device=%s confidence=%.2f",
            Path(model_path).name,
            self.device,
            self.confidence_threshold,
        )

    def detect(self, image: np.ndarray) -> list[Person]:
        """Detect every person in a BGR image, largest first."""
        try:
            result = self._model.predict(
                image,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                classes=[_PERSON_CLASS],
                verbose=False,
            )[0]
        except Exception as error:  # noqa: BLE001 - surfaced with context
            raise PersonDetectorError(f"Person detection failed: {error}") from error

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = (
            result.keypoints.data.cpu().numpy()
            if result.keypoints is not None
            else np.zeros((len(boxes), 17, 3), dtype=np.float32)
        )

        persons = [
            Person(
                box=tuple(round(float(v), 1) for v in box),
                confidence=round(float(score), 3),
                keypoints=[[round(float(v), 2) for v in kp] for kp in kps],
            )
            for box, score, kps in zip(boxes, scores, keypoints)
        ]
        return sorted(persons, key=lambda person: person.area, reverse=True)
