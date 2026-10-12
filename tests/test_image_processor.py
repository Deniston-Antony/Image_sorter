"""Tests for image discovery, loading, validation and pre-processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import config
import image_processor
from image_processor import ImageProcessingError


def write_png(path: Path, image: np.ndarray) -> Path:
    """Write ``image`` to ``path``, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


# --------------------------------------------------------------------------- #
# Extension handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "photo.jpg",
        "photo.JPG",
        "photo.jpeg",
        "photo.JPEG",
        "photo.png",
        "photo.PNG",
        "photo.bmp",
        "photo.BMP",
        "photo.webp",
        "photo.WEBP",
        "photo.tiff",
        "photo.TIFF",
        "photo.tif",
        "photo.TIF",
        "photo.JpEg",
    ],
)
def test_supported_extensions_are_case_insensitive(name: str) -> None:
    assert image_processor.is_supported_image(Path(name))


@pytest.mark.parametrize(
    "name", ["notes.txt", "clip.mp4", "archive.zip", "raw.cr2", "noextension"]
)
def test_unsupported_extensions_are_rejected(name: str) -> None:
    assert not image_processor.is_supported_image(Path(name))


# --------------------------------------------------------------------------- #
# find_images
# --------------------------------------------------------------------------- #


def test_find_images_is_recursive_by_default(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    write_png(tmp_path / "top.png", blank_image)
    write_png(tmp_path / "vacation" / "photo1.png", blank_image)
    write_png(tmp_path / "family" / "nested" / "photo2.PNG", blank_image)
    (tmp_path / "notes.txt").write_text("ignored")

    found = image_processor.find_images(tmp_path)

    assert {p.name for p in found} == {"top.png", "photo1.png", "photo2.PNG"}


def test_find_images_non_recursive_skips_subdirectories(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    write_png(tmp_path / "top.png", blank_image)
    write_png(tmp_path / "sub" / "deep.png", blank_image)

    found = image_processor.find_images(tmp_path, recursive=False)

    assert [p.name for p in found] == ["top.png"]


def test_find_images_on_empty_folder_returns_empty_list(empty_input_dir: Path) -> None:
    assert image_processor.find_images(empty_input_dir) == []


def test_find_images_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        image_processor.find_images(tmp_path / "does_not_exist")


def test_find_images_on_file_path_raises(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    path = write_png(tmp_path / "single.png", blank_image)
    with pytest.raises(NotADirectoryError):
        image_processor.find_images(path)


def test_find_images_excludes_requested_directories(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    write_png(tmp_path / "keep.png", blank_image)
    output_dir = tmp_path / "output"
    write_png(output_dir / "faces" / "previous_run.png", blank_image)

    found = image_processor.find_images(tmp_path, exclude_dirs=(output_dir,))

    assert [p.name for p in found] == ["keep.png"]


def test_find_images_results_are_sorted(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    for name in ("c.png", "a.png", "b.png"):
        write_png(tmp_path / name, blank_image)

    found = image_processor.find_images(tmp_path)

    assert found == sorted(found)


def test_find_images_honours_custom_extension_set(
    tmp_path: Path, blank_image: np.ndarray
) -> None:
    write_png(tmp_path / "a.png", blank_image)
    write_png(tmp_path / "b.bmp", blank_image)

    found = image_processor.find_images(tmp_path, extensions=frozenset({".bmp"}))

    assert [p.name for p in found] == ["b.bmp"]


# --------------------------------------------------------------------------- #
# validate_image
# --------------------------------------------------------------------------- #


def test_validate_accepts_colour_and_grayscale(blank_image: np.ndarray) -> None:
    image_processor.validate_image(blank_image)
    image_processor.validate_image(blank_image[:, :, 0])


def test_validate_rejects_none() -> None:
    with pytest.raises(ImageProcessingError, match="could not be decoded"):
        image_processor.validate_image(None, "x.jpg")


def test_validate_rejects_empty_array() -> None:
    with pytest.raises(ImageProcessingError, match="no pixel data"):
        image_processor.validate_image(np.array([], dtype=np.uint8))


def test_validate_rejects_zero_dimension_image() -> None:
    with pytest.raises(ImageProcessingError):
        image_processor.validate_image(np.zeros((0, 10, 3), dtype=np.uint8))


def test_validate_rejects_wrong_dimensionality() -> None:
    with pytest.raises(ImageProcessingError, match="Unsupported image shape"):
        image_processor.validate_image(np.zeros((2, 2, 2, 3), dtype=np.uint8))


def test_validate_rejects_odd_channel_count() -> None:
    with pytest.raises(ImageProcessingError, match="channel count"):
        image_processor.validate_image(np.zeros((8, 8, 5), dtype=np.uint8))


def test_validate_rejects_non_array() -> None:
    with pytest.raises(ImageProcessingError, match="not an array"):
        image_processor.validate_image("definitely not an image")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# load_image
# --------------------------------------------------------------------------- #


def test_load_valid_image_round_trips(tmp_path: Path, blank_image: np.ndarray) -> None:
    path = write_png(tmp_path / "valid.png", blank_image)

    loaded = image_processor.load_image(path)

    assert loaded.shape == blank_image.shape
    assert np.array_equal(loaded, blank_image)


@pytest.mark.parametrize("extension", [".jpg", ".png", ".bmp", ".webp", ".tiff"])
def test_load_every_supported_format(
    tmp_path: Path, blank_image: np.ndarray, extension: str
) -> None:
    path = write_png(tmp_path / f"sample{extension}", blank_image)

    loaded = image_processor.load_image(path)

    assert loaded.shape[:2] == blank_image.shape[:2]


def test_load_uppercase_extension(tmp_path: Path, blank_image: np.ndarray) -> None:
    path = write_png(tmp_path / "UPPER.JPG", blank_image)

    assert image_processor.load_image(path).size > 0


def test_load_corrupted_image_raises(corrupted_image_file: Path) -> None:
    with pytest.raises(ImageProcessingError):
        image_processor.load_image(corrupted_image_file)


def test_load_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jpg"
    path.touch()

    with pytest.raises(ImageProcessingError, match="File is empty"):
        image_processor.load_image(path)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageProcessingError, match="File not found"):
        image_processor.load_image(tmp_path / "ghost.jpg")


def test_load_text_file_with_image_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_text("I am a text file wearing a PNG costume" * 40)

    with pytest.raises(ImageProcessingError):
        image_processor.load_image(path)


def test_load_handles_unicode_filename(tmp_path: Path, blank_image: np.ndarray) -> None:
    path = write_png(tmp_path / "家族写真_ünïcode.png", blank_image)

    assert image_processor.load_image(path).size > 0


# --------------------------------------------------------------------------- #
# to_bgr / preprocess_image
# --------------------------------------------------------------------------- #


def test_to_bgr_converts_grayscale() -> None:
    gray = np.full((10, 12), 90, dtype=np.uint8)

    result = image_processor.to_bgr(gray)

    assert result.shape == (10, 12, 3)


def test_to_bgr_converts_bgra() -> None:
    bgra = np.zeros((10, 12, 4), dtype=np.uint8)

    assert image_processor.to_bgr(bgra).shape == (10, 12, 3)


def test_to_bgr_passes_through_bgr(blank_image: np.ndarray) -> None:
    assert image_processor.to_bgr(blank_image).shape == blank_image.shape


def test_preprocess_leaves_normal_image_untouched(blank_image: np.ndarray) -> None:
    processed, scale = image_processor.preprocess_image(blank_image)

    assert scale == 1.0
    assert processed.shape == blank_image.shape


def test_preprocess_downscales_oversized_image() -> None:
    huge = np.zeros((400, 1200, 3), dtype=np.uint8)

    processed, scale = image_processor.preprocess_image(huge, max_long_side=600)

    assert max(processed.shape[:2]) == 600
    assert scale == pytest.approx(0.5)


def test_preprocess_converts_grayscale_to_three_channels() -> None:
    gray = np.full((50, 50), 10, dtype=np.uint8)

    processed, _ = image_processor.preprocess_image(gray)

    assert processed.ndim == 3 and processed.shape[2] == 3


def test_preprocess_rejects_invalid_image() -> None:
    with pytest.raises(ImageProcessingError):
        image_processor.preprocess_image(np.zeros((0, 0, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# letterbox
# --------------------------------------------------------------------------- #


def test_letterbox_produces_exact_requested_size() -> None:
    image = np.full((100, 400, 3), 255, dtype=np.uint8)

    result = image_processor.letterbox(image, 320, 320)

    assert result.image.shape == (320, 320, 3)


def test_letterbox_preserves_aspect_ratio() -> None:
    image = np.full((100, 400, 3), 255, dtype=np.uint8)

    result = image_processor.letterbox(image, 320, 320)

    # 400x100 scaled by 0.8 is 320x80, centred vertically in a 320-tall canvas.
    assert result.scale == pytest.approx(0.8)
    assert result.pad_x == 0
    assert result.pad_y == 120


def test_letterbox_inverse_transform_recovers_coordinates() -> None:
    image = np.zeros((200, 500, 3), dtype=np.uint8)
    result = image_processor.letterbox(image, 640, 640)

    x, y = result.to_source(
        result.pad_x + 100 * result.scale, result.pad_y + 50 * result.scale
    )

    assert (x, y) == pytest.approx((100.0, 50.0))


def test_letterbox_upscales_small_images() -> None:
    tiny = np.full((16, 16, 3), 200, dtype=np.uint8)

    result = image_processor.letterbox(tiny, 128, 128)

    assert result.scale > 1.0
    assert result.image.shape == (128, 128, 3)


def test_supported_extension_set_matches_specification() -> None:
    assert config.SUPPORTED_EXTENSIONS == frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    )
