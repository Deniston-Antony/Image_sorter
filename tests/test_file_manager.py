"""Tests for output directory creation and non-destructive file copying."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import config
import file_manager
from detector import Face


def write_image_file(path: Path, image: np.ndarray) -> Path:
    """Write ``image`` to ``path``, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


# --------------------------------------------------------------------------- #
# Output directory creation
# --------------------------------------------------------------------------- #


def test_create_output_directories_makes_the_three_core_folders(
    run_config: config.Config,
) -> None:
    file_manager.create_output_directories(run_config)

    assert run_config.faces_dir.is_dir()
    assert run_config.no_faces_dir.is_dir()
    assert run_config.failed_dir.is_dir()


def test_optional_folders_are_not_created_by_default(
    run_config: config.Config,
) -> None:
    file_manager.create_output_directories(run_config)

    assert not run_config.debug_dir.exists()
    assert not run_config.face_crops_dir.exists()


def test_visualize_creates_debug_folder(run_config: config.Config) -> None:
    run_config.visualize = True

    file_manager.create_output_directories(run_config)

    assert run_config.debug_dir.is_dir()


def test_crop_faces_creates_crops_folder(run_config: config.Config) -> None:
    run_config.crop_faces = True

    file_manager.create_output_directories(run_config)

    assert run_config.face_crops_dir.is_dir()


def test_create_output_directories_is_idempotent(run_config: config.Config) -> None:
    file_manager.create_output_directories(run_config)
    file_manager.create_output_directories(run_config)

    assert run_config.faces_dir.is_dir()


def test_create_output_directories_creates_nested_paths(tmp_path: Path) -> None:
    cfg = config.Config(
        input_dir=tmp_path,
        output_dir=tmp_path / "deeply" / "nested" / "output",
    )

    file_manager.create_output_directories(cfg)

    assert cfg.faces_dir.is_dir()


def test_create_output_directories_reports_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(*args: object, **kwargs: object) -> None:
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "mkdir", deny)
    cfg = config.Config(input_dir=tmp_path, output_dir=tmp_path / "out")

    with pytest.raises(file_manager.FileManagerError, match="Permission denied"):
        file_manager.create_output_directories(cfg)


# --------------------------------------------------------------------------- #
# Duplicate filenames
# --------------------------------------------------------------------------- #


def test_unique_destination_returns_plain_name_when_free(tmp_path: Path) -> None:
    assert file_manager.unique_destination(tmp_path, "photo.jpg").name == "photo.jpg"


def test_unique_destination_appends_incrementing_suffixes(tmp_path: Path) -> None:
    names: list[str] = []
    for _ in range(4):
        destination = file_manager.unique_destination(tmp_path, "photo.jpg")
        destination.write_bytes(b"x")
        names.append(destination.name)

    assert names == ["photo.jpg", "photo_1.jpg", "photo_2.jpg", "photo_3.jpg"]


def test_unique_destination_preserves_extension_case(tmp_path: Path) -> None:
    (tmp_path / "image.JPG").write_bytes(b"x")

    assert file_manager.unique_destination(tmp_path, "image.JPG").name == "image_1.JPG"


def test_unique_destination_handles_multi_dot_names(tmp_path: Path) -> None:
    (tmp_path / "my.holiday.photo.jpeg").write_bytes(b"x")

    destination = file_manager.unique_destination(tmp_path, "my.holiday.photo.jpeg")

    assert destination.name == "my.holiday.photo_1.jpeg"


def test_duplicate_names_from_different_folders_do_not_overwrite(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    first = write_image_file(tmp_path / "folder1" / "photo.jpg", blank_image)
    second = write_image_file(tmp_path / "folder2" / "photo.jpg", blank_image)
    destination_dir = tmp_path / "out"

    copied_first = file_manager.copy_image(first, destination_dir)
    copied_second = file_manager.copy_image(second, destination_dir)

    assert copied_first != copied_second
    assert copied_first.exists() and copied_second.exists()
    assert len(list(destination_dir.iterdir())) == 2


def test_relative_name_keeps_folder_structure(tmp_path: Path) -> None:
    source = tmp_path / "family" / "reunion" / "photo.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x")

    assert file_manager.relative_name(source, tmp_path) == "family__reunion__photo.jpg"


def test_relative_name_for_top_level_file(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")

    assert file_manager.relative_name(source, tmp_path) == "photo.jpg"


def test_relative_name_falls_back_for_unrelated_paths(tmp_path: Path) -> None:
    source = tmp_path / "elsewhere" / "photo.jpg"
    source.parent.mkdir()
    source.write_bytes(b"x")

    assert file_manager.relative_name(source, tmp_path / "other") == "photo.jpg"


def test_relative_name_makes_duplicates_distinct(tmp_path: Path) -> None:
    names = []
    for folder in ("folder1", "folder2"):
        source = tmp_path / folder / "photo.jpg"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"x")
        names.append(file_manager.relative_name(source, tmp_path))

    assert names == ["folder1__photo.jpg", "folder2__photo.jpg"]
    assert len(set(names)) == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("normal", "normal"),
        ("with space", "with_space"),
        ("sl/ash", "sl_ash"),
        ("../escape", "escape"),
        ("", "image"),
        ("...", "image"),
    ],
)
def test_sanitize_stem(raw: str, expected: str) -> None:
    assert file_manager.sanitize_stem(raw) == expected


# --------------------------------------------------------------------------- #
# Copying
# --------------------------------------------------------------------------- #


def test_copy_preserves_the_original(tmp_path: Path, blank_image: np.ndarray) -> None:
    source = write_image_file(tmp_path / "input" / "photo.png", blank_image)
    original_bytes = source.read_bytes()

    file_manager.copy_image(source, tmp_path / "out")

    assert source.exists()
    assert source.read_bytes() == original_bytes


def test_copy_produces_byte_identical_file(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    source = write_image_file(tmp_path / "photo.png", blank_image)

    destination = file_manager.copy_image(source, tmp_path / "out")

    assert destination.read_bytes() == source.read_bytes()


def test_copy_preserves_uppercase_extension(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    source = write_image_file(tmp_path / "image.JPG", blank_image)

    destination = file_manager.copy_image(source, tmp_path / "out")

    assert destination.name == "image.JPG"
    assert destination.suffix == ".JPG"


@pytest.mark.parametrize("extension", [".jpg", ".JPEG", ".png", ".BMP", ".tiff"])
def test_copy_never_converts_format(
    tmp_path: Path, blank_image: np.ndarray, extension: str
) -> None:
    source = write_image_file(tmp_path / f"photo{extension}", blank_image)

    destination = file_manager.copy_image(source, tmp_path / "out")

    assert destination.suffix == extension


def test_copy_creates_missing_destination_directory(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    source = write_image_file(tmp_path / "photo.png", blank_image)

    destination = file_manager.copy_image(source, tmp_path / "brand" / "new")

    assert destination.exists()


def test_copy_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(file_manager.FileManagerError):
        file_manager.copy_image(tmp_path / "ghost.jpg", tmp_path / "out")


def test_classification_helpers_target_the_right_folders(
    run_config: config.Config, blank_image: np.ndarray
) -> None:
    file_manager.create_output_directories(run_config)
    source = write_image_file(run_config.input_dir / "photo.png", blank_image)

    assert (
        file_manager.copy_to_faces(source, run_config).parent == run_config.faces_dir
    )
    assert (
        file_manager.copy_to_no_faces(source, run_config).parent
        == run_config.no_faces_dir
    )
    assert (
        file_manager.copy_to_failed(source, run_config).parent == run_config.failed_dir
    )


def test_copy_with_explicit_filename(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    source = write_image_file(tmp_path / "photo.png", blank_image)

    destination = file_manager.copy_image(
        source, tmp_path / "out", filename="renamed.png"
    )

    assert destination.name == "renamed.png"


# --------------------------------------------------------------------------- #
# Drawing, cropping and writing
# --------------------------------------------------------------------------- #


def test_draw_detections_does_not_modify_the_source(blank_image: np.ndarray) -> None:
    before = blank_image.copy()
    faces = [Face(10.0, 10.0, 60.0, 70.0, 0.91)]

    annotated = file_manager.draw_detections(blank_image, faces)

    assert np.array_equal(blank_image, before)
    assert not np.array_equal(annotated, before)
    assert annotated.shape == blank_image.shape


def test_draw_detections_handles_boxes_at_the_border(
    blank_image: np.ndarray,
) -> None:
    height, width = blank_image.shape[:2]
    faces = [
        Face(0.0, 0.0, 30.0, 30.0, 0.9),
        Face(width - 20.0, height - 20.0, width + 50.0, height + 50.0, 0.8),
    ]

    annotated = file_manager.draw_detections(blank_image, faces)

    assert annotated.shape == blank_image.shape


def test_draw_detections_with_no_faces_returns_a_copy(
    blank_image: np.ndarray,
) -> None:
    annotated = file_manager.draw_detections(blank_image, [])

    assert np.array_equal(annotated, blank_image)
    assert annotated is not blank_image


def test_narrow_boxes_are_drawn_without_a_text_label() -> None:
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    narrow = [Face(100.0, 100.0, 130.0, 140.0, 0.93)]

    labelled = file_manager.draw_detections(image, narrow, min_label_width=10)
    unlabelled = file_manager.draw_detections(image, narrow, min_label_width=200)

    # Both draw the rectangle, but only the first adds label pixels.
    assert not np.array_equal(labelled, image)
    assert not np.array_equal(unlabelled, image)
    assert np.count_nonzero(labelled != image) > np.count_nonzero(
        unlabelled != image
    )


def test_label_threshold_default_labels_a_large_face() -> None:
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    large = [Face(50.0, 100.0, 250.0, 300.0, 0.93)]

    with_label = file_manager.draw_detections(image, large)
    without_label = file_manager.draw_detections(image, large, min_label_width=10_000)

    assert np.count_nonzero(with_label != image) > np.count_nonzero(
        without_label != image
    )


def test_crop_face_returns_region_with_margin(blank_image: np.ndarray) -> None:
    face = Face(50.0, 40.0, 90.0, 80.0, 0.9)

    crop = file_manager.crop_face(blank_image, face, margin=0.0)

    assert crop is not None
    assert crop.shape[:2] == (40, 40)


def test_crop_face_clamps_to_image_bounds(blank_image: np.ndarray) -> None:
    height, width = blank_image.shape[:2]
    face = Face(-30.0, -30.0, width + 30.0, height + 30.0, 0.9)

    crop = file_manager.crop_face(blank_image, face)

    assert crop is not None
    assert crop.shape[:2] == (height, width)


def test_crop_face_returns_none_for_degenerate_box(blank_image: np.ndarray) -> None:
    assert file_manager.crop_face(blank_image, Face(10.0, 10.0, 10.0, 10.0, 0.9)) is None


def test_crop_face_returns_none_for_out_of_bounds_box(
    blank_image: np.ndarray,
) -> None:
    face = Face(5000.0, 5000.0, 5100.0, 5100.0, 0.9)

    assert file_manager.crop_face(blank_image, face) is None


def test_save_face_crops_writes_one_file_per_face(
    run_config: config.Config, blank_image: np.ndarray
) -> None:
    run_config.crop_faces = True
    file_manager.create_output_directories(run_config)
    source = write_image_file(run_config.input_dir / "group.jpg", blank_image)
    faces = [
        Face(5.0, 5.0, 45.0, 45.0, 0.95),
        Face(55.0, 10.0, 95.0, 50.0, 0.90),
        Face(100.0, 20.0, 140.0, 60.0, 0.85),
    ]

    written = file_manager.save_face_crops(blank_image, faces, source, run_config)

    assert [p.name for p in written] == [
        "group_face_001.jpg",
        "group_face_002.jpg",
        "group_face_003.jpg",
    ]
    assert all(p.exists() for p in written)


def test_save_face_crops_skips_unusable_boxes(
    run_config: config.Config, blank_image: np.ndarray
) -> None:
    run_config.crop_faces = True
    file_manager.create_output_directories(run_config)
    source = write_image_file(run_config.input_dir / "mixed.png", blank_image)
    faces = [Face(5.0, 5.0, 45.0, 45.0, 0.95), Face(900.0, 900.0, 950.0, 950.0, 0.9)]

    written = file_manager.save_face_crops(blank_image, faces, source, run_config)

    assert len(written) == 1


def test_save_debug_image_writes_annotated_copy(
    run_config: config.Config, blank_image: np.ndarray
) -> None:
    run_config.visualize = True
    file_manager.create_output_directories(run_config)
    source = write_image_file(run_config.input_dir / "photo.png", blank_image)

    destination = file_manager.save_debug_image(
        blank_image, [Face(10.0, 10.0, 60.0, 60.0, 0.9)], source, run_config
    )

    assert destination.exists()
    assert destination.parent == run_config.debug_dir
    assert cv2.imread(str(destination)) is not None


def test_write_image_rejects_unknown_extension(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    with pytest.raises(file_manager.FileManagerError):
        file_manager.write_image(blank_image, tmp_path / "out.unsupported")
