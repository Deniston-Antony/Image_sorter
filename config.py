"""Central configuration for the face image sorter.

Every tunable value lives here so that the rest of the code never needs to
hard-code a path, threshold or magic number. Command line flags parsed in
``main.py`` override the defaults by building a :class:`Config` instance.

All paths are derived from ``Path(__file__).parent`` so the project can be
copied or moved anywhere without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
DEFAULT_INPUT_DIR: Final[Path] = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"
#: Default output folder of ``people_sorter.py``.
DEFAULT_SORTED_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output_sorted"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# Sub-folders created inside the output directory.
FACES_DIRNAME: Final[str] = "faces"
NO_FACES_DIRNAME: Final[str] = "no_faces"
FAILED_DIRNAME: Final[str] = "failed"
DEBUG_DIRNAME: Final[str] = "debug"
FACE_CROPS_DIRNAME: Final[str] = "face_crops"

LOG_FILENAME: Final[str] = "processing.log"

# --------------------------------------------------------------------------- #
# Model files
# --------------------------------------------------------------------------- #

# Primary detector: OpenCV YuNet (cv2.FaceDetectorYN), an accurate and very
# small ONNX model shipped by the OpenCV Zoo.
YUNET_MODEL_FILENAME: Final[str] = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL: Final[str] = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Secondary detector: the classic OpenCV DNN ResNet-10 SSD face detector.
SSD_PROTO_FILENAME: Final[str] = "deploy.prototxt"
SSD_WEIGHTS_FILENAME: Final[str] = "res10_300x300_ssd_iter_140000.caffemodel"
SSD_PROTO_URL: Final[str] = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/"
    "face_detector/deploy.prototxt"
)
SSD_WEIGHTS_URL: Final[str] = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)

# --------------------------------------------------------------------------- #
# Detection defaults
# --------------------------------------------------------------------------- #

#: Minimum detector confidence for a candidate to count as a face.
#: Lower -> more faces found but more false positives.
#: Higher -> fewer false positives but small/hard faces get missed.
CONFIDENCE_THRESHOLD: Final[float] = 0.5

#: Non-maximum-suppression IoU threshold used both inside the detector and when
#: merging the results of several detection passes.
NMS_THRESHOLD: Final[float] = 0.3

#: Detector input resolution. Images are letterboxed into this size, so larger
#: values find smaller faces at the cost of time and memory.
DETECTOR_WIDTH: Final[int] = 640
DETECTOR_HEIGHT: Final[int] = 640

#: Maximum number of candidates YuNet keeps before NMS.
TOP_K: Final[int] = 5000

#: Multi-scale tiling: split the image into an overlapping grid and run the
#: detector on every tile. This is what makes tiny faces in large group photos
#: detectable. Set to 1 to disable tiling and use a single full-image pass.
TILE_GRID: Final[int] = 2

#: Fractional overlap between neighbouring tiles so faces on a tile border are
#: still fully contained in at least one tile.
TILE_OVERLAP: Final[float] = 0.2

#: Only run the tiled passes when the image is at least this many pixels on its
#: long side; small images gain nothing from tiling.
MIN_SIDE_FOR_TILING: Final[int] = 700

#: Hard safety cap on the long side of an image before detection. Protects
#: against absurdly large inputs exhausting memory. Detection coordinates are
#: always mapped back to the original resolution.
MAX_IMAGE_LONG_SIDE: Final[int] = 6000

#: Reject degenerate detections smaller than this (pixels, on the original image).
MIN_FACE_SIZE_PX: Final[int] = 8

# --------------------------------------------------------------------------- #
# Input / output defaults
# --------------------------------------------------------------------------- #

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)

RECURSIVE: Final[bool] = True

#: Margin added around a face box when cropping, as a fraction of box size.
CROP_MARGIN: Final[float] = 0.2

#: Drawing style for ``--visualize``.
BOX_COLOR: Final[tuple[int, int, int]] = (0, 220, 0)  # BGR
BOX_THICKNESS: Final[int] = 2
LABEL_COLOR: Final[tuple[int, int, int]] = (0, 0, 0)
LABEL_BG_COLOR: Final[tuple[int, int, int]] = (0, 220, 0)

#: Boxes narrower than this get no text label in debug images. In a crowd
#: photograph the labels of dozens of small faces overlap into an unreadable
#: mass, so below this width only the rectangle is drawn; the per-face
#: confidence is still recorded in logs/processing.log.
MIN_LABEL_BOX_WIDTH: Final[int] = 52

DEVICE_CPU: Final[str] = "cpu"
DEVICE_CUDA: Final[str] = "cuda"
VALID_DEVICES: Final[tuple[str, ...]] = (DEVICE_CPU, DEVICE_CUDA)

VALID_BACKENDS: Final[tuple[str, ...]] = ("auto", "yunet", "ssd", "haar")


@dataclass(slots=True)
class Config:
    """Resolved, validated runtime settings for a single run.

    Instances are normally created by :func:`main.build_config` from parsed
    command line arguments, which is why every field has a sensible default.
    """

    input_dir: Path = DEFAULT_INPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR

    confidence_threshold: float = CONFIDENCE_THRESHOLD
    nms_threshold: float = NMS_THRESHOLD
    detector_width: int = DETECTOR_WIDTH
    detector_height: int = DETECTOR_HEIGHT

    backend: str = "auto"
    device: str = DEVICE_CPU

    tile_grid: int = TILE_GRID
    tile_overlap: float = TILE_OVERLAP

    recursive: bool = RECURSIVE
    visualize: bool = False
    crop_faces: bool = False
    quiet: bool = False
    verbose: bool = False

    extensions: frozenset[str] = field(default_factory=lambda: SUPPORTED_EXTENSIONS)

    # ------------------------------------------------------------------ #
    # Derived paths
    # ------------------------------------------------------------------ #

    @property
    def faces_dir(self) -> Path:
        """Destination for images containing at least one face."""
        return self.output_dir / FACES_DIRNAME

    @property
    def no_faces_dir(self) -> Path:
        """Destination for images with no detected face."""
        return self.output_dir / NO_FACES_DIRNAME

    @property
    def failed_dir(self) -> Path:
        """Destination for images that could not be processed."""
        return self.output_dir / FAILED_DIRNAME

    @property
    def debug_dir(self) -> Path:
        """Destination for annotated images (``--visualize``)."""
        return self.output_dir / DEBUG_DIRNAME

    @property
    def face_crops_dir(self) -> Path:
        """Destination for cropped faces (``--crop-faces``)."""
        return self.output_dir / FACE_CROPS_DIRNAME

    @property
    def log_file(self) -> Path:
        """Path of the run's log file."""
        return self.logs_dir / LOG_FILENAME

    def validate(self) -> None:
        """Check the configuration and raise on anything unusable.

        Raises:
            ValueError: If a numeric option is out of range or an enum-like
                option (device, backend) has an unknown value.
            FileNotFoundError: If the input directory does not exist.
            NotADirectoryError: If the input path is not a directory.
        """
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "--confidence must be greater than 0.0 and at most 1.0, "
                f"got {self.confidence_threshold}"
            )
        if not 0.0 < self.nms_threshold <= 1.0:
            raise ValueError(
                f"--nms must be in (0.0, 1.0], got {self.nms_threshold}"
            )
        if self.detector_width < 32 or self.detector_height < 32:
            raise ValueError("--width and --height must each be at least 32 pixels")
        if self.tile_grid < 1:
            raise ValueError("--tile-grid must be 1 (disabled) or greater")
        if not 0.0 <= self.tile_overlap < 1.0:
            raise ValueError("--tile-overlap must be in [0.0, 1.0)")
        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"--device must be one of {', '.join(VALID_DEVICES)}, got {self.device!r}"
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"--backend must be one of {', '.join(VALID_BACKENDS)}, "
                f"got {self.backend!r}"
            )
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a folder: {self.input_dir}")
