"""Image discovery, loading, validation and pre-processing.

This module is the only place that touches image bytes on the way in. It is
deliberately defensive: every function either returns a usable image or raises
:class:`ImageProcessingError`, so the caller can route the offending file to
``output/failed/`` and carry on with the next image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

import config
from logger import get_logger

_log = get_logger()


class ImageProcessingError(Exception):
    """Raised when an image cannot be read, decoded or validated."""


class LetterboxResult(NamedTuple):
    """A padded, fixed-size image plus the transform back to source pixels.

    Attributes:
        image: The letterboxed image of exactly the requested size.
        scale: Factor the source image was multiplied by.
        pad_x: Horizontal padding added on the left edge, in pixels.
        pad_y: Vertical padding added on the top edge, in pixels.
    """

    image: np.ndarray
    scale: float
    pad_x: int
    pad_y: int

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map a coordinate from the letterboxed image back to the source."""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale


def is_supported_image(path: Path, extensions: frozenset[str] | None = None) -> bool:
    """Return whether ``path`` has one of the supported image extensions.

    The comparison is case-insensitive, so ``photo.JPG``, ``photo.jpg`` and
    ``photo.JPeG`` are all accepted.
    """
    allowed = config.SUPPORTED_EXTENSIONS if extensions is None else extensions
    return path.suffix.lower() in allowed


def find_images(
    input_dir: Path,
    *,
    recursive: bool = True,
    extensions: frozenset[str] | None = None,
    exclude_dirs: tuple[Path, ...] = (),
) -> list[Path]:
    """Collect every supported image file under ``input_dir``.

    Args:
        input_dir: Folder to search.
        recursive: Search sub-directories as well as ``input_dir`` itself.
        extensions: Override the set of accepted (lower-case, dotted) suffixes.
        exclude_dirs: Directories to skip. Used to avoid re-ingesting a
            previous run's output when the output folder lives inside the
            input folder.

    Returns:
        A sorted list of file paths. Empty if nothing matched.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
        NotADirectoryError: If ``input_dir`` is not a directory.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    resolved_excludes = tuple(d.resolve() for d in exclude_dirs)
    candidates: Iterator[Path] = (
        input_dir.rglob("*") if recursive else input_dir.glob("*")
    )

    images: list[Path] = []
    for path in candidates:
        try:
            if not path.is_file() or not is_supported_image(path, extensions):
                continue
            if _is_inside_any(path, resolved_excludes):
                continue
        except OSError as error:  # broken symlink, permission denied, ...
            _log.warning("Skipping unreadable path %s: %s", path, error)
            continue
        images.append(path)

    return sorted(images)


def _is_inside_any(path: Path, directories: tuple[Path, ...]) -> bool:
    """Return whether ``path`` lives inside any of ``directories``."""
    if not directories:
        return False
    resolved = path.resolve()
    return any(directory in resolved.parents for directory in directories)


def validate_image(image: np.ndarray | None, source: Path | str = "<memory>") -> None:
    """Verify that a decoded array is a usable image.

    Args:
        image: Candidate image array, possibly ``None``.
        source: Path or label used in the error message.

    Raises:
        ImageProcessingError: If the array is missing, empty, has a zero
            dimension, or does not look like a 2-D grayscale / 3-D colour image.
    """
    if image is None:
        raise ImageProcessingError(f"Image could not be decoded: {source}")
    if not isinstance(image, np.ndarray):
        raise ImageProcessingError(f"Decoded object is not an array: {source}")
    if image.size == 0:
        raise ImageProcessingError(f"Image contains no pixel data: {source}")
    if image.ndim not in (2, 3):
        raise ImageProcessingError(
            f"Unsupported image shape {image.shape}: {source}"
        )
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageProcessingError(
            f"Image has zero dimensions ({width}x{height}): {source}"
        )
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ImageProcessingError(
            f"Unsupported channel count {image.shape[2]}: {source}"
        )


def load_image(path: Path) -> np.ndarray:
    """Read an image from disk as a BGR array.

    Bytes are read with :mod:`pathlib` and decoded with :func:`cv2.imdecode`
    rather than :func:`cv2.imread` so that non-ASCII paths work identically on
    every platform. The result is validated before being returned.

    Args:
        path: Image file to read.

    Returns:
        The decoded image.

    Raises:
        ImageProcessingError: If the file is missing, unreadable, empty,
            corrupted or in a format OpenCV cannot decode.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise ImageProcessingError(f"File not found: {path}") from error
    except PermissionError as error:
        raise ImageProcessingError(f"Permission denied reading {path}") from error
    except OSError as error:
        raise ImageProcessingError(f"Could not read {path}: {error}") from error

    if not raw:
        raise ImageProcessingError(f"File is empty: {path}")

    try:
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ImageProcessingError(f"OpenCV failed to decode {path}: {error}") from error

    validate_image(image, path)
    return image


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR view of a grayscale, BGR or BGRA image."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def preprocess_image(
    image: np.ndarray,
    *,
    max_long_side: int = config.MAX_IMAGE_LONG_SIDE,
) -> tuple[np.ndarray, float]:
    """Normalise an image for detection.

    Converts to 3-channel BGR and, if the image is enormous, shrinks it so its
    long side is at most ``max_long_side``. Shrinking is a safety valve against
    memory exhaustion on very large inputs; the returned scale lets the caller
    map detections back to original pixel coordinates.

    Args:
        image: Validated input image.
        max_long_side: Upper bound for the longest side, in pixels.

    Returns:
        A tuple of ``(processed_image, scale)`` where ``scale`` is the factor
        the original was multiplied by (``1.0`` when no resize happened).
    """
    validate_image(image)
    processed = to_bgr(image)

    height, width = processed.shape[:2]
    long_side = max(height, width)
    if max_long_side > 0 and long_side > max_long_side:
        scale = max_long_side / long_side
        processed = cv2.resize(
            processed,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        _log.debug(
            "Downscaled oversized image from %dx%d to %dx%d",
            width,
            height,
            processed.shape[1],
            processed.shape[0],
        )
        return processed, scale

    return processed, 1.0


def letterbox(image: np.ndarray, width: int, height: int) -> LetterboxResult:
    """Resize an image into a fixed ``width`` x ``height`` canvas.

    The aspect ratio is preserved and the remaining space is padded, which
    keeps faces undistorted. Up-scaling is allowed on purpose: feeding a small
    crop into a larger detector input is what makes tiny faces detectable.

    Args:
        image: Source BGR image.
        width: Target canvas width.
        height: Target canvas height.

    Returns:
        A :class:`LetterboxResult` carrying the canvas and the inverse transform.
    """
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    new_width = max(1, min(width, int(round(source_width * scale))))
    new_height = max(1, min(height, int(round(source_height * scale))))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=image.dtype)
    pad_x = (width - new_width) // 2
    pad_y = (height - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized

    return LetterboxResult(image=canvas, scale=scale, pad_x=pad_x, pad_y=pad_y)
