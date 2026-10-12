"""Tests for similar-photo grouping."""

from __future__ import annotations

import numpy as np

from similar_groups import group_similar, read_capture_time


def _unit(*rows: list[float]) -> np.ndarray:
    vectors = np.asarray(rows, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_near_identical_vectors_are_grouped() -> None:
    vectors = _unit([1, 0, 0], [0, 1, 0], [1, 0.05, 0], [0, 0, 1], [0.02, 1, 0])
    assert group_similar(vectors, threshold=0.95) == [[0, 2], [1, 4]]


def test_distinct_vectors_form_no_group() -> None:
    assert group_similar(_unit([1, 0, 0], [0, 1, 0], [0, 0, 1]), threshold=0.9) == []


def test_fewer_than_two_images() -> None:
    assert group_similar(_unit([1, 0, 0]), threshold=0.9) == []


def test_chunking_does_not_change_the_result() -> None:
    vectors = _unit([1, 0, 0], [0, 1, 0], [1, 0.05, 0], [0, 0, 1], [0.02, 1, 0])
    assert group_similar(vectors, threshold=0.95, chunk_size=2) == [[0, 2], [1, 4]]


def test_capture_time_separates_lookalikes() -> None:
    vectors = _unit([1, 0, 0], [1, 0.01, 0], [1, 0.02, 0])
    groups = group_similar(
        vectors, threshold=0.95, capture_times=[0.0, 3.0, 500.0], max_seconds_apart=10.0
    )
    assert groups == [[0, 1]]


def test_missing_capture_time_does_not_block_grouping() -> None:
    vectors = _unit([1, 0, 0], [1, 0.01, 0])
    groups = group_similar(
        vectors, threshold=0.95, capture_times=[0.0, None], max_seconds_apart=10.0
    )
    assert groups == [[0, 1]]


def test_read_capture_time_without_exif(tmp_path) -> None:
    import cv2

    path = tmp_path / "plain.jpg"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    assert read_capture_time(path) is None
    assert read_capture_time(tmp_path / "missing.jpg") is None
