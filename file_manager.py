"""Output directory management and non-destructive file copying.

Every function here treats the input tree as read-only. Images are **copied**
into the output folders with :func:`shutil.copy2`, which also preserves
timestamps; originals are never moved, renamed, modified or deleted.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

import config
from detector import Face
from logger import get_logger

_log = get_logger()

#: Upper bound on the numeric suffix tried when resolving a name collision.
_MAX_NAME_ATTEMPTS = 10_000


class FileManagerError(Exception):
    """Raised when an output directory or file operation cannot be completed."""


def create_output_directories(cfg: config.Config) -> list[Path]:
    """Create every output folder required by the current configuration.

    ``faces/``, ``no_faces/`` and ``failed/`` are always created. ``debug/``
    and ``face_crops/`` are created only when the corresponding option is on,
    so a plain run leaves no empty optional folders behind.

    Args:
        cfg: The run configuration.

    Returns:
        The directories that now exist.

    Raises:
        FileManagerError: If a directory could not be created.
    """
    directories = [cfg.faces_dir, cfg.no_faces_dir, cfg.failed_dir]
    if cfg.visualize:
        directories.append(cfg.debug_dir)
    if cfg.crop_faces:
        directories.append(cfg.face_crops_dir)

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError as error:
            raise FileManagerError(
                f"Permission denied creating output folder {directory}"
            ) from error
        except OSError as error:
            raise FileManagerError(
                f"Could not create output folder {directory}: {error}"
            ) from error

    _log.debug("Output directories ready: %s", ", ".join(map(str, directories)))
    return directories


def sanitize_stem(stem: str) -> str:
    """Make a filename stem safe to use as a single path component."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in stem
    )
    return cleaned.strip("._") or "image"


def unique_destination(directory: Path, filename: str) -> Path:
    """Return a path inside ``directory`` that does not collide with an existing file.

    The original extension is always preserved exactly as given, including its
    case, so ``image.JPG`` stays ``image.JPG``. When the name is already taken
    a numeric suffix is appended:

    ``photo.jpg`` -> ``photo_1.jpg`` -> ``photo_2.jpg`` -> ...

    Args:
        directory: Target folder.
        filename: Desired file name, extension included.

    Returns:
        A free path within ``directory``.

    Raises:
        FileManagerError: If no free name was found within a sane number of tries.
    """
    candidate = Path(filename)
    stem = sanitize_stem(candidate.stem)
    suffix = candidate.suffix

    destination = directory / f"{stem}{suffix}"
    if not destination.exists():
        return destination

    for counter in range(1, _MAX_NAME_ATTEMPTS):
        destination = directory / f"{stem}_{counter}{suffix}"
        if not destination.exists():
            return destination

    raise FileManagerError(
        f"Could not find a free filename for {filename} in {directory}"
    )


def relative_name(source: Path, input_dir: Path) -> str:
    """Build a flat filename that keeps the source's relative folder structure.

    ``input/family/photo.jpg`` relative to ``input`` becomes ``family__photo.jpg``,
    which keeps provenance visible and makes collisions between identically
    named files in different folders far less likely.

    Args:
        source: The image file.
        input_dir: Root of the input tree.

    Returns:
        A single-component filename.
    """
    try:
        relative = source.resolve().relative_to(input_dir.resolve())
    except ValueError:
        return source.name

    parts = [sanitize_stem(part) for part in relative.parts[:-1]]
    if not parts:
        return source.name
    return "__".join([*parts, source.name])


def copy_image(
    source: Path,
    destination_dir: Path,
    *,
    filename: str | None = None,
) -> Path:
    """Copy an image into ``destination_dir`` without touching the original.

    Args:
        source: File to copy.
        destination_dir: Target folder, created if missing.
        filename: Override the destination file name. Defaults to the source's
            own name. The extension is preserved either way.

    Returns:
        The path the file was written to.

    Raises:
        FileManagerError: If the copy failed.
    """
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(destination_dir, filename or source.name)
        shutil.copy2(source, destination)
    except FileManagerError:
        raise
    except PermissionError as error:
        raise FileManagerError(f"Permission denied copying {source}: {error}") from error
    except OSError as error:
        raise FileManagerError(f"Could not copy {source}: {error}") from error

    _log.debug("Copied %s -> %s", source, destination)
    return destination


def copy_to_faces(source: Path, cfg: config.Config, *, filename: str | None = None) -> Path:
    """Copy an image that contains at least one face into ``output/faces/``."""
    return copy_image(source, cfg.faces_dir, filename=filename)


def copy_to_no_faces(
    source: Path, cfg: config.Config, *, filename: str | None = None
) -> Path:
    """Copy an image with no detected face into ``output/no_faces/``."""
    return copy_image(source, cfg.no_faces_dir, filename=filename)


def copy_to_failed(
    source: Path, cfg: config.Config, *, filename: str | None = None
) -> Path:
    """Copy an image that could not be processed into ``output/failed/``."""
    return copy_image(source, cfg.failed_dir, filename=filename)


# --------------------------------------------------------------------------- #
# Optional outputs
# --------------------------------------------------------------------------- #


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_label(canvas: np.ndarray, label: str, box: tuple[int, int, int, int]) -> None:
    """Draw ``label`` on a filled background just above ``box``.

    The font is sized to the box so that a label never dwarfs the face it
    describes, and the text is nudged inside the frame when the box sits
    against an edge.
    """
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = box

    font_scale = float(np.clip((x2 - x1) / 160.0, 0.4, 0.8))
    thickness = max(1, int(round(font_scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, _FONT, font_scale, thickness
    )

    # Prefer above the box; drop below it when there is no headroom.
    if y1 - text_height - baseline >= 0:
        text_baseline = y1 - baseline
    else:
        text_baseline = y2 + text_height
    text_baseline = int(np.clip(text_baseline, text_height, height - baseline - 1))
    text_left = int(np.clip(x1, 0, max(0, width - text_width)))

    cv2.rectangle(
        canvas,
        (text_left, text_baseline - text_height - baseline),
        (text_left + text_width, text_baseline + baseline),
        config.LABEL_BG_COLOR,
        cv2.FILLED,
    )
    cv2.putText(
        canvas,
        label,
        (text_left, text_baseline),
        _FONT,
        font_scale,
        config.LABEL_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    image: np.ndarray,
    faces: list[Face],
    *,
    min_label_width: int = config.MIN_LABEL_BOX_WIDTH,
) -> np.ndarray:
    """Return a **copy** of the image annotated with face boxes and scores.

    Every detected face gets a rectangle. Faces wide enough to carry readable
    text also get a ``Face N: 0.92`` label; narrower ones are left unlabelled
    so that a crowd photograph does not turn into a wall of overlapping text.
    Their confidences are still written to the log file.

    Args:
        image: Source image. Never modified.
        faces: Faces to draw, in source pixel coordinates.
        min_label_width: Minimum box width, in pixels, that gets a text label.

    Returns:
        A new annotated image.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for index, face in enumerate(faces, start=1):
        box = face.clipped(width, height).as_int_box()
        x1, y1, x2, y2 = box
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS
        )
        if x2 - x1 >= min_label_width:
            _draw_label(canvas, f"Face {index}: {face.confidence:.2f}", box)

    return canvas


def write_image(image: np.ndarray, destination: Path) -> Path:
    """Encode and write an image, preserving the extension of ``destination``.

    Args:
        image: Image to write.
        destination: Target path; its suffix selects the encoder.

    Returns:
        The path written.

    Raises:
        FileManagerError: If encoding or writing failed.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        success, buffer = cv2.imencode(destination.suffix, image)
    except (cv2.error, OSError) as error:
        raise FileManagerError(f"Could not encode {destination}: {error}") from error

    if not success:
        raise FileManagerError(f"OpenCV could not encode {destination}")

    try:
        destination.write_bytes(buffer.tobytes())
    except OSError as error:
        raise FileManagerError(f"Could not write {destination}: {error}") from error

    return destination


def save_debug_image(
    image: np.ndarray, faces: list[Face], source: Path, cfg: config.Config
) -> Path:
    """Write an annotated copy of ``source`` into ``output/debug/``.

    Args:
        image: The loaded source image. Not modified.
        faces: Faces to annotate.
        source: Original file, used for naming.
        cfg: Run configuration.

    Returns:
        The path of the annotated image.

    Raises:
        FileManagerError: If the image could not be written.
    """
    annotated = draw_detections(image, faces)
    destination = unique_destination(cfg.debug_dir, source.name)
    return write_image(annotated, destination)


def crop_face(
    image: np.ndarray, face: Face, *, margin: float = config.CROP_MARGIN
) -> np.ndarray | None:
    """Extract a single face as a new image, with a little context margin.

    Args:
        image: Source image. Not modified.
        face: Face box to extract.
        margin: Extra context around the box as a fraction of its size.

    Returns:
        The cropped region, or ``None`` if the box lies outside the image or is
        degenerate.
    """
    height, width = image.shape[:2]
    pad_x = face.width * margin
    pad_y = face.height * margin

    x1 = int(round(max(0.0, face.x1 - pad_x)))
    y1 = int(round(max(0.0, face.y1 - pad_y)))
    x2 = int(round(min(float(width), face.x2 + pad_x)))
    y2 = int(round(min(float(height), face.y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def save_face_crops(
    image: np.ndarray, faces: list[Face], source: Path, cfg: config.Config
) -> list[Path]:
    """Write one image per detected face into ``output/face_crops/``.

    Crops are named ``<stem>_face_001.<ext>``, ``<stem>_face_002.<ext>``, ...
    A failure on an individual crop is logged and skipped rather than aborting
    the rest, because cropping is an optional extra that must never influence
    the face / no-face classification.

    Args:
        image: The loaded source image. Not modified.
        faces: Faces to crop.
        source: Original file, used for naming.
        cfg: Run configuration.

    Returns:
        Paths of the crops that were written successfully.
    """
    stem = sanitize_stem(source.stem)
    suffix = source.suffix or ".jpg"
    written: list[Path] = []

    for index, face in enumerate(faces, start=1):
        region = crop_face(image, face)
        if region is None or region.size == 0:
            _log.warning("Skipped empty crop %d for %s", index, source.name)
            continue
        destination = unique_destination(
            cfg.face_crops_dir, f"{stem}_face_{index:03d}{suffix}"
        )
        try:
            written.append(write_image(region, destination))
        except FileManagerError as error:
            _log.warning("Could not write crop %s: %s", destination.name, error)

    return written
