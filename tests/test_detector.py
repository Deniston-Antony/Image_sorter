"""Tests for the face detector.

Covers the pure geometry helpers, the backend-selection logic and real
detection against photographs in ``tests/assets/``: one face, many faces and
no face.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
import detector
from detector import Face, FaceDetector, merge_detections
from image_processor import ImageProcessingError


@pytest.fixture
def face_detector(detector_available: bool) -> FaceDetector:
    """A detector built with default settings."""
    return FaceDetector(confidence_threshold=config.CONFIDENCE_THRESHOLD)


# --------------------------------------------------------------------------- #
# Face value object
# --------------------------------------------------------------------------- #


def test_face_unpacks_as_a_five_tuple() -> None:
    x1, y1, x2, y2, confidence = Face(1.0, 2.0, 11.0, 22.0, 0.75)

    assert (x1, y1, x2, y2, confidence) == (1.0, 2.0, 11.0, 22.0, 0.75)


def test_face_geometry_properties() -> None:
    face = Face(10.0, 20.0, 40.0, 60.0, 0.9)

    assert face.width == 30.0
    assert face.height == 40.0
    assert face.area == 1200.0


def test_face_as_int_box_rounds() -> None:
    assert Face(1.4, 2.6, 10.5, 20.4, 0.9).as_int_box() == (1, 3, 10, 20)


def test_face_clipped_to_image_bounds() -> None:
    clipped = Face(-10.0, -5.0, 200.0, 300.0, 0.9).clipped(100, 80)

    assert clipped.as_int_box() == (0, 0, 100, 80)
    assert clipped.confidence == 0.9


def test_face_scaled_and_offset_keep_confidence() -> None:
    face = Face(10.0, 10.0, 20.0, 20.0, 0.42)

    assert face.scaled(2.0) == Face(20.0, 20.0, 40.0, 40.0, 0.42)
    assert face.offset(5.0, -5.0) == Face(15.0, 5.0, 25.0, 15.0, 0.42)


def test_face_width_is_never_negative() -> None:
    assert Face(50.0, 50.0, 10.0, 10.0, 0.9).width == 0.0


# --------------------------------------------------------------------------- #
# merge_detections
# --------------------------------------------------------------------------- #


def test_merge_deduplicates_overlapping_boxes() -> None:
    faces = [
        Face(10.0, 10.0, 60.0, 60.0, 0.95),
        Face(12.0, 11.0, 62.0, 61.0, 0.80),  # same face, another pass
    ]

    merged = merge_detections(faces, nms_threshold=0.3)

    assert len(merged) == 1
    assert merged[0].confidence == 0.95


def test_merge_keeps_distinct_faces() -> None:
    faces = [
        Face(0.0, 0.0, 50.0, 50.0, 0.9),
        Face(200.0, 200.0, 250.0, 250.0, 0.9),
        Face(400.0, 0.0, 450.0, 50.0, 0.9),
    ]

    assert len(merge_detections(faces)) == 3


def test_merge_drops_low_confidence_candidates() -> None:
    faces = [
        Face(0.0, 0.0, 50.0, 50.0, 0.90),
        Face(200.0, 200.0, 250.0, 250.0, 0.10),
    ]

    merged = merge_detections(faces, confidence_threshold=0.5)

    assert len(merged) == 1


def test_merge_sorts_by_descending_confidence() -> None:
    faces = [
        Face(0.0, 0.0, 40.0, 40.0, 0.50),
        Face(100.0, 0.0, 140.0, 40.0, 0.99),
        Face(200.0, 0.0, 240.0, 40.0, 0.75),
    ]

    scores = [face.confidence for face in merge_detections(faces)]

    assert scores == sorted(scores, reverse=True)


def test_merge_handles_empty_and_single_inputs() -> None:
    assert merge_detections([]) == []
    single = [Face(0.0, 0.0, 10.0, 10.0, 0.9)]
    assert merge_detections(single) == single


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #


def test_tiling_disabled_for_grid_of_one(face_detector: FaceDetector) -> None:
    face_detector.tile_grid = 1

    assert face_detector._tiles(4000, 3000) == []


def test_tiling_skipped_for_small_images(face_detector: FaceDetector) -> None:
    assert face_detector._tiles(320, 240) == []


def test_tiling_produces_grid_squared_tiles(face_detector: FaceDetector) -> None:
    face_detector.tile_grid = 3

    tiles = face_detector._tiles(1800, 1800)

    assert len(tiles) == 9


def test_tiles_stay_inside_the_image(face_detector: FaceDetector) -> None:
    width, height = 1600, 1200

    for x, y, w, h in face_detector._tiles(width, height):
        assert x >= 0 and y >= 0
        assert x + w <= width
        assert y + h <= height


def test_tiles_overlap_as_configured(face_detector: FaceDetector) -> None:
    face_detector.tile_grid = 2
    face_detector.tile_overlap = 0.2

    tiles = face_detector._tiles(1000, 1000)
    first, second = tiles[0], tiles[1]

    # The second tile must start before the first one ends.
    assert second[0] < first[0] + first[2]


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def test_auto_backend_initialises(face_detector: FaceDetector) -> None:
    assert face_detector.backend.name in config.VALID_BACKENDS
    assert face_detector.backend.description


def test_missing_model_file_raises_model_file_error(tmp_path: Path) -> None:
    with pytest.raises(detector.ModelFileError, match="not found"):
        detector.YuNetBackend(tmp_path / "absent.onnx")


def test_explicit_unknown_backend_raises() -> None:
    with pytest.raises(detector.DetectorInitializationError):
        FaceDetector(backend="nonexistent-backend")


def test_auto_selection_fails_cleanly_without_any_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(detector.HaarBackend, "is_available", staticmethod(lambda: False))
    monkeypatch.setattr(detector.SsdBackend, "is_available", staticmethod(lambda: False))

    with pytest.raises(detector.DetectorInitializationError, match="download_model"):
        FaceDetector(backend="auto", models_dir=tmp_path)


def test_cuda_request_falls_back_to_cpu_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, detector_available: bool
) -> None:
    monkeypatch.setattr(detector, "cuda_is_available", lambda: False)

    instance = FaceDetector(device=config.DEVICE_CUDA)

    assert instance.device == config.DEVICE_CPU


def test_cpu_is_the_default_device(face_detector: FaceDetector) -> None:
    assert face_detector.device == config.DEVICE_CPU


def test_cuda_availability_probe_never_raises() -> None:
    assert isinstance(detector.cuda_is_available(), bool)


# --------------------------------------------------------------------------- #
# Real detection
# --------------------------------------------------------------------------- #


def test_image_with_one_face(
    face_detector: FaceDetector, one_face_image: np.ndarray
) -> None:
    faces = face_detector.detect_faces(one_face_image)

    assert len(faces) == 1
    assert face_detector.has_face(one_face_image) is True


def test_detected_box_is_inside_the_image(
    face_detector: FaceDetector, one_face_image: np.ndarray
) -> None:
    height, width = one_face_image.shape[:2]
    face = face_detector.detect_faces(one_face_image)[0]

    assert 0 <= face.x1 < face.x2 <= width
    assert 0 <= face.y1 < face.y2 <= height
    assert 0.0 < face.confidence <= 1.0


def test_image_with_multiple_faces(
    face_detector: FaceDetector, multi_face_image: np.ndarray
) -> None:
    faces = face_detector.detect_faces(multi_face_image)

    assert len(faces) == 3, "detection must not stop after the first face"


def test_multiple_faces_are_distinct_regions(
    face_detector: FaceDetector, multi_face_image: np.ndarray
) -> None:
    faces = face_detector.detect_faces(multi_face_image)
    centres = sorted((face.x1 + face.x2) / 2 for face in faces)

    for left, right in zip(centres, centres[1:]):
        assert right - left > 100


def test_large_group_photo_finds_many_faces(
    face_detector: FaceDetector, group_image: np.ndarray
) -> None:
    faces = face_detector.detect_faces(group_image)

    assert len(faces) > 20, f"expected a crowd, found {len(faces)}"
    assert all(0.0 < face.confidence <= 1.0 for face in faces)


def test_group_photo_contains_a_range_of_face_sizes(
    face_detector: FaceDetector, group_image: np.ndarray
) -> None:
    areas = sorted(face.area for face in face_detector.detect_faces(group_image))

    assert areas, "no faces detected"
    assert areas[-1] > areas[0], "detector should find faces at different scales"


def test_image_without_a_face(
    face_detector: FaceDetector, no_face_image: np.ndarray
) -> None:
    assert face_detector.detect_faces(no_face_image) == []
    assert face_detector.has_face(no_face_image) is False


def test_blank_image_has_no_face(
    face_detector: FaceDetector, blank_image: np.ndarray
) -> None:
    assert face_detector.has_face(blank_image) is False


def test_random_noise_has_no_face(face_detector: FaceDetector) -> None:
    rng = np.random.default_rng(seed=1234)
    noise = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    assert face_detector.has_face(noise) is False


def test_higher_confidence_never_finds_more_faces(
    detector_available: bool, group_image: np.ndarray
) -> None:
    lenient = FaceDetector(confidence_threshold=0.3).detect_faces(group_image)
    strict = FaceDetector(confidence_threshold=0.9).detect_faces(group_image)

    assert len(strict) <= len(lenient)


def test_grayscale_input_is_accepted(
    face_detector: FaceDetector, one_face_image: np.ndarray
) -> None:
    import cv2

    gray = cv2.cvtColor(one_face_image, cv2.COLOR_BGR2GRAY)

    assert face_detector.has_face(gray) is True


def test_tiny_image_does_not_crash(face_detector: FaceDetector) -> None:
    tiny = np.full((8, 8, 3), 120, dtype=np.uint8)

    assert face_detector.detect_faces(tiny) == []


def test_invalid_array_raises_image_processing_error(
    face_detector: FaceDetector,
) -> None:
    with pytest.raises(ImageProcessingError):
        face_detector.detect_faces(np.zeros((0, 0, 3), dtype=np.uint8))


def test_detection_is_deterministic(
    face_detector: FaceDetector, one_face_image: np.ndarray
) -> None:
    first = face_detector.detect_faces(one_face_image)
    second = face_detector.detect_faces(one_face_image)

    assert [f.as_int_box() for f in first] == [f.as_int_box() for f in second]


def test_very_large_image_is_handled(
    face_detector: FaceDetector, one_face_image: np.ndarray
) -> None:
    import cv2

    huge = cv2.resize(one_face_image, (3000, 3000), interpolation=cv2.INTER_LINEAR)

    faces = face_detector.detect_faces(huge)

    assert len(faces) >= 1
    # Coordinates must be reported against the original resolution.
    assert max(face.x2 for face in faces) > 1000


def test_detections_respect_minimum_face_size(
    face_detector: FaceDetector, group_image: np.ndarray
) -> None:
    for face in face_detector.detect_faces(group_image):
        assert face.width >= face_detector.min_face_size
        assert face.height >= face_detector.min_face_size
