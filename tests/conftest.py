"""Shared pytest fixtures for the face image sorter test suite.

The project modules are top-level (``main.py`` is run directly, not as a
package), so the project root is prepended to ``sys.path`` here. Real face
photographs live in ``tests/assets/`` so the detection tests run offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (import after sys.path setup)
import cv2  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

#: Single-face portrait (OpenCV's ``lena.jpg`` sample).
ONE_FACE_ASSET = ASSETS_DIR / "one_face.jpg"

#: Large class photograph with dozens of faces at small scale
#: (OpenCV's ``class57.png`` cascade test image).
GROUP_ASSET = ASSETS_DIR / "group.png"


def _require_asset(path: Path) -> Path:
    """Return ``path`` or skip the test when the asset is missing."""
    if not path.is_file():
        pytest.skip(
            f"Test asset missing: {path.name}. "
            "See tests/assets/README.md for download instructions."
        )
    return path


def _model_available() -> bool:
    """Return whether at least one detection backend can be constructed."""
    import detector

    if (config.MODELS_DIR / config.YUNET_MODEL_FILENAME).is_file():
        return True
    if detector.HaarBackend.is_available():
        return True
    ssd_files = (
        config.MODELS_DIR / config.SSD_PROTO_FILENAME,
        config.MODELS_DIR / config.SSD_WEIGHTS_FILENAME,
    )
    return detector.SsdBackend.is_available() and all(p.is_file() for p in ssd_files)


# --------------------------------------------------------------------------- #
# Synthetic images
# --------------------------------------------------------------------------- #


@pytest.fixture
def blank_image() -> np.ndarray:
    """A plain mid-grey 120x160 BGR image."""
    return np.full((120, 160, 3), 128, dtype=np.uint8)


@pytest.fixture
def no_face_image() -> np.ndarray:
    """A synthetic texture that contains no face.

    A vertical gradient plus geometric shapes, which stands in for the kind of
    landscape or object photo that must land in ``no_faces/``.
    """
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(20, 200, height, dtype=np.uint8)
    image[:, :] = gradient[:, None, None]
    cv2.rectangle(image, (40, 300), (280, 440), (30, 90, 160), cv2.FILLED)
    cv2.circle(image, (480, 120), 70, (200, 200, 60), cv2.FILLED)
    cv2.line(image, (0, 260), (width, 290), (10, 10, 10), 6)
    return image


@pytest.fixture
def one_face_image() -> np.ndarray:
    """A real photograph containing exactly one face."""
    import image_processor

    return image_processor.load_image(_require_asset(ONE_FACE_ASSET))


@pytest.fixture
def group_image() -> np.ndarray:
    """A real photograph containing many faces at a range of scales."""
    import image_processor

    return image_processor.load_image(_require_asset(GROUP_ASSET))


@pytest.fixture
def multi_face_image(one_face_image: np.ndarray) -> np.ndarray:
    """A composite image holding three copies of a real face side by side.

    Built from the single-face asset so the expected face count is known
    exactly. This is the fixture that proves detection does not stop after the
    first face.
    """
    face = cv2.resize(one_face_image, (256, 256))
    canvas = np.full((300, 3 * 280, 3), 40, dtype=np.uint8)
    for index in range(3):
        left = index * 280 + 12
        canvas[22:278, left : left + 256] = face
    return canvas


# --------------------------------------------------------------------------- #
# Filesystem fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def empty_input_dir(tmp_path: Path) -> Path:
    """An existing but empty input folder."""
    directory = tmp_path / "empty_input"
    directory.mkdir()
    return directory


@pytest.fixture
def corrupted_image_file(tmp_path: Path) -> Path:
    """A ``.jpg`` file whose bytes are not a valid image."""
    path = tmp_path / "corrupted.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 this is not actually a JPEG payload")
    return path


@pytest.fixture
def run_config(tmp_path: Path) -> config.Config:
    """A :class:`config.Config` pointing entirely inside ``tmp_path``."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    return config.Config(
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        logs_dir=tmp_path / "logs",
    )


@pytest.fixture
def detector_available() -> bool:
    """Skip the test unless a detection backend can be built."""
    if not _model_available():
        pytest.skip(
            "No detection model available. Run 'python download_model.py' first."
        )
    return True
