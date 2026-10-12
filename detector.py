"""Face **detection** (not recognition) built on OpenCV.

The module answers exactly one question for a given image: *does it contain one
or more human faces, and where are they?* No identity, age, gender, emotion or
person-level information is computed anywhere.

Architecture
------------
``FaceDetector`` is the public facade. It owns a *backend* that knows how to run
one neural network over one image, and adds the logic that matters for group
photographs:

1. A full-image pass catches large and medium faces.
2. Optional overlapping tile passes re-run the backend on sub-regions, which
   effectively raises the resolution seen by the network and is what makes
   small faces in large group shots detectable.
3. All candidate boxes from all passes are merged with non-maximum suppression.

Three backends are provided and selected by capability at runtime:

=========  ==================================================================
``yunet``  OpenCV Zoo YuNet ONNX model. Accurate, tiny, the recommended default.
``ssd``    Classic ResNet-10 SSD Caffe model. Requires OpenCV 4.x.
``haar``   Bundled Haar cascade. No download needed, requires OpenCV 4.x.
=========  ==================================================================

OpenCV 5 removed the Caffe importer and ``cv2.CascadeClassifier``, so on that
version only ``yunet`` is available. The selection logic reports clearly which
backend it ended up using.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final, NamedTuple, Sequence

import cv2
import numpy as np

import config
import image_processor
from logger import get_logger

_log = get_logger()

#: Synthetic confidence assigned to Haar cascade detections, which have no
#: probabilistic score of their own. Scaled from the cascade's neighbour count.
_HAAR_BASE_CONFIDENCE: Final[float] = 0.55


class Face(NamedTuple):
    """A single detected face.

    Unpacks as ``(x1, y1, x2, y2, confidence)`` so callers can treat the
    detector output as a plain list of tuples.

    Attributes:
        x1: Left edge in source-image pixels.
        y1: Top edge in source-image pixels.
        x2: Right edge in source-image pixels.
        y2: Bottom edge in source-image pixels.
        confidence: Detector score in ``[0.0, 1.0]``.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def as_int_box(self) -> tuple[int, int, int, int]:
        """Return the box as integer ``(x1, y1, x2, y2)`` for drawing/cropping."""
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )

    def clipped(self, width: int, height: int) -> "Face":
        """Return a copy clamped to the bounds of a ``width`` x ``height`` image."""
        return Face(
            x1=float(np.clip(self.x1, 0, width)),
            y1=float(np.clip(self.y1, 0, height)),
            x2=float(np.clip(self.x2, 0, width)),
            y2=float(np.clip(self.y2, 0, height)),
            confidence=self.confidence,
        )

    def scaled(self, factor: float) -> "Face":
        """Return a copy with all coordinates multiplied by ``factor``."""
        return Face(
            x1=self.x1 * factor,
            y1=self.y1 * factor,
            x2=self.x2 * factor,
            y2=self.y2 * factor,
            confidence=self.confidence,
        )

    def offset(self, dx: float, dy: float) -> "Face":
        """Return a copy translated by ``(dx, dy)`` pixels."""
        return Face(
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
            confidence=self.confidence,
        )


class DetectorError(Exception):
    """Base class for detector problems."""


class ModelFileError(DetectorError):
    """Raised when a required model file is missing or unreadable."""


class DetectorInitializationError(DetectorError):
    """Raised when no usable detection backend could be constructed."""


def cuda_is_available() -> bool:
    """Return whether this OpenCV build exposes at least one CUDA device."""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (AttributeError, cv2.error):
        return False


def _dnn_ids(device: str) -> tuple[int, int]:
    """Map a device name to OpenCV DNN ``(backend_id, target_id)``.

    CPU uses the default backend rather than explicitly selecting the OpenCV
    one: on OpenCV 5 an explicit target request makes the new graph engine emit
    a warning for what is already the default behaviour.
    """
    if device == config.DEVICE_CUDA:
        return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class DetectionBackend(ABC):
    """Runs one face-detection network over one whole image.

    A backend is responsible for feeding the image to its network at the
    network's expected input size and returning boxes in the coordinate space
    of the image it was given. Tiling and merging live in :class:`FaceDetector`.
    """

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Face]:
        """Detect faces in a BGR image.

        Args:
            image: BGR image of any size.

        Returns:
            Faces in ``image`` pixel coordinates.
        """

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Return whether this OpenCV build supports the backend's API."""

    @property
    def description(self) -> str:
        """Human-readable backend summary for logs and terminal output."""
        return self.name


class YuNetBackend(DetectionBackend):
    """OpenCV Zoo YuNet ONNX detector.

    YuNet is an anchor-based CNN that emits every face in one forward pass with
    a real confidence score, which is exactly what group-photo classification
    needs. The model is ~230 KB.
    """

    name = "yunet"

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.NMS_THRESHOLD,
        top_k: int = config.TOP_K,
        input_size: tuple[int, int] = (config.DETECTOR_WIDTH, config.DETECTOR_HEIGHT),
        device: str = config.DEVICE_CPU,
    ) -> None:
        """Load the YuNet model.

        Args:
            model_path: Path to ``face_detection_yunet_*.onnx``.
            confidence_threshold: Score below which candidates are discarded.
            nms_threshold: IoU threshold for the model's internal NMS.
            top_k: Candidates kept before NMS.
            input_size: ``(width, height)`` the image is letterboxed into.
            device: ``"cpu"`` or ``"cuda"``.

        Raises:
            ModelFileError: If the ONNX file is missing or cannot be loaded.
        """
        if not Path(model_path).is_file():
            raise ModelFileError(
                f"YuNet model not found: {model_path}\n"
                "Run 'python download_model.py' to fetch it, or download it "
                f"manually from {config.YUNET_MODEL_URL}"
            )

        self._input_size = input_size
        backend_id, target_id = _dnn_ids(device)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(model_path),
                config="",
                input_size=input_size,
                score_threshold=float(confidence_threshold),
                nms_threshold=float(nms_threshold),
                top_k=int(top_k),
                backend_id=backend_id,
                target_id=target_id,
            )
        except cv2.error as error:
            raise ModelFileError(
                f"OpenCV could not load the YuNet model {model_path}: {error}"
            ) from error

    @staticmethod
    def is_available() -> bool:
        """YuNet exists in OpenCV 4.5.4+ and 5.x."""
        return hasattr(cv2, "FaceDetectorYN")

    @property
    def description(self) -> str:
        """Backend summary including the input resolution."""
        return f"yunet (ONNX, input {self._input_size[0]}x{self._input_size[1]})"

    def detect(self, image: np.ndarray) -> list[Face]:
        """Letterbox the image into the model input size and run one pass."""
        box = image_processor.letterbox(image, *self._input_size)

        try:
            self._detector.setInputSize(self._input_size)
            _, raw = self._detector.detect(box.image)
        except cv2.error as error:
            _log.warning("YuNet forward pass failed: %s", error)
            return []

        if raw is None or len(raw) == 0:
            return []

        faces: list[Face] = []
        for row in raw:
            x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            confidence = float(row[-1])
            left, top = box.to_source(x, y)
            right, bottom = box.to_source(x + w, y + h)
            faces.append(Face(left, top, right, bottom, confidence))
        return faces


class SsdBackend(DetectionBackend):
    """Classic OpenCV DNN ResNet-10 SSD face detector (Caffe).

    Kept as a fallback for environments where the YuNet model file is absent.
    Requires an OpenCV build with the Caffe importer, i.e. OpenCV 4.x.
    """

    name = "ssd"

    #: Mean values the Caffe model was trained with.
    MEAN: Final[tuple[float, float, float]] = (104.0, 177.0, 123.0)

    def __init__(
        self,
        proto_path: Path,
        weights_path: Path,
        *,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        input_size: tuple[int, int] = (300, 300),
        device: str = config.DEVICE_CPU,
    ) -> None:
        """Load the SSD network.

        Args:
            proto_path: ``deploy.prototxt``.
            weights_path: ``res10_300x300_ssd_iter_140000.caffemodel``.
            confidence_threshold: Score below which candidates are discarded.
            input_size: Network input ``(width, height)``.
            device: ``"cpu"`` or ``"cuda"``.

        Raises:
            ModelFileError: If a model file is missing or the net cannot load.
            DetectorInitializationError: If this OpenCV build has no Caffe importer.
        """
        if not self.is_available():
            raise DetectorInitializationError(
                "This OpenCV build has no Caffe importer "
                f"(cv2 {cv2.__version__}); the 'ssd' backend is unavailable. "
                "Use the 'yunet' backend, or install opencv-python<5."
            )

        missing = [str(p) for p in (proto_path, weights_path) if not Path(p).is_file()]
        if missing:
            raise ModelFileError(
                "SSD model file(s) not found: "
                + ", ".join(missing)
                + "\nRun 'python download_model.py --all' to fetch them."
            )

        self._confidence_threshold = float(confidence_threshold)
        self._input_size = input_size

        try:
            self._net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
        except cv2.error as error:
            raise ModelFileError(f"Could not load SSD model: {error}") from error

        if device == config.DEVICE_CUDA:
            backend_id, target_id = _dnn_ids(device)
            self._net.setPreferableBackend(backend_id)
            self._net.setPreferableTarget(target_id)

    @staticmethod
    def is_available() -> bool:
        """The Caffe importer was removed in OpenCV 5."""
        return hasattr(cv2.dnn, "readNetFromCaffe")

    @property
    def description(self) -> str:
        """Backend summary including the input resolution."""
        return f"ssd (ResNet-10 Caffe, input {self._input_size[0]}x{self._input_size[1]})"

    def detect(self, image: np.ndarray) -> list[Face]:
        """Run one SSD forward pass and scale relative boxes to pixels."""
        height, width = image.shape[:2]
        try:
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=1.0,
                size=self._input_size,
                mean=self.MEAN,
                swapRB=False,
                crop=False,
            )
            self._net.setInput(blob)
            raw = self._net.forward()
        except cv2.error as error:
            _log.warning("SSD forward pass failed: %s", error)
            return []

        faces: list[Face] = []
        for index in range(raw.shape[2]):
            confidence = float(raw[0, 0, index, 2])
            if confidence < self._confidence_threshold:
                continue
            x1, y1, x2, y2 = raw[0, 0, index, 3:7] * np.array(
                [width, height, width, height], dtype=np.float32
            )
            faces.append(Face(float(x1), float(y1), float(x2), float(y2), confidence))
        return faces


class HaarBackend(DetectionBackend):
    """Haar cascade detector using the cascade bundled with OpenCV.

    This is the last-resort backend: it needs no downloaded model at all, but
    it is noticeably weaker than the DNN backends, especially on rotated or
    partially occluded faces. Requires OpenCV 4.x, which still ships
    ``cv2.CascadeClassifier`` and the cascade XML data files.
    """

    name = "haar"

    CASCADE_FILENAME: Final[str] = "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        *,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (24, 24),
        cascade_path: Path | None = None,
    ) -> None:
        """Load the cascade classifier.

        Args:
            confidence_threshold: Threshold applied to the synthetic score.
            scale_factor: Image pyramid step. Smaller finds more, slower.
            min_neighbors: Neighbour votes required to keep a detection.
            min_size: Smallest face to look for, in pixels.
            cascade_path: Override the cascade XML location.

        Raises:
            DetectorInitializationError: If this OpenCV build has no cascade
                support or the XML file cannot be loaded.
        """
        if not self.is_available():
            raise DetectorInitializationError(
                "This OpenCV build has no cv2.CascadeClassifier "
                f"(cv2 {cv2.__version__}); the 'haar' backend is unavailable. "
                "Use the 'yunet' backend, or install opencv-python<5."
            )

        path = Path(cascade_path) if cascade_path else self._default_cascade_path()
        if path is None or not path.is_file():
            raise DetectorInitializationError(
                f"Haar cascade file not found: {path}"
            )

        self._classifier = cv2.CascadeClassifier(str(path))
        if self._classifier.empty():
            raise DetectorInitializationError(
                f"Haar cascade could not be loaded from {path}"
            )

        self._confidence_threshold = float(confidence_threshold)
        self._scale_factor = float(scale_factor)
        self._min_neighbors = int(min_neighbors)
        self._min_size = min_size

    @staticmethod
    def is_available() -> bool:
        """``cv2.CascadeClassifier`` was removed in OpenCV 5."""
        return hasattr(cv2, "CascadeClassifier")

    @staticmethod
    def _default_cascade_path() -> Path | None:
        """Locate the frontal-face cascade shipped inside the cv2 package."""
        data_dir = getattr(cv2, "data", None)
        if data_dir is None:
            return None
        return Path(data_dir.haarcascades) / HaarBackend.CASCADE_FILENAME

    @property
    def description(self) -> str:
        """Backend summary."""
        return "haar (bundled cascade, no model download required)"

    def detect(self, image: np.ndarray) -> list[Face]:
        """Run the cascade and convert neighbour counts into pseudo-confidences.

        Cascades return no probability, so a score is synthesised from the
        number of neighbouring detections that voted for each box. This keeps
        the ``--confidence`` flag meaningful across all backends.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            boxes, _, weights = self._classifier.detectMultiScale3(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
                outputRejectLevels=True,
            )
        except cv2.error as error:
            _log.warning("Haar cascade failed: %s", error)
            return []

        if len(boxes) == 0:
            return []

        faces: list[Face] = []
        for (x, y, w, h), weight in zip(boxes, np.ravel(weights)):
            confidence = min(1.0, _HAAR_BASE_CONFIDENCE + float(weight) / 20.0)
            if confidence < self._confidence_threshold:
                continue
            faces.append(
                Face(float(x), float(y), float(x + w), float(y + h), confidence)
            )
        return faces


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #


def merge_detections(
    faces: Sequence[Face],
    *,
    nms_threshold: float = config.NMS_THRESHOLD,
    confidence_threshold: float = 0.0,
) -> list[Face]:
    """Deduplicate overlapping boxes from multiple detection passes.

    Args:
        faces: Candidate boxes, possibly containing near-duplicates of the same
            face found by different passes.
        nms_threshold: Boxes overlapping an already-kept box by more than this
            IoU are suppressed.
        confidence_threshold: Candidates scoring below this are dropped first.

    Returns:
        Surviving faces sorted by descending confidence.
    """
    candidates = [f for f in faces if f.confidence >= confidence_threshold]
    if len(candidates) <= 1:
        return list(candidates)

    boxes = [
        [int(round(f.x1)), int(round(f.y1)), int(round(f.width)), int(round(f.height))]
        for f in candidates
    ]
    scores = [float(f.confidence) for f in candidates]

    try:
        keep = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=0.0, nms_threshold=float(nms_threshold)
        )
    except cv2.error as error:
        _log.warning("NMS failed (%s); keeping all candidates", error)
        return sorted(candidates, key=lambda f: f.confidence, reverse=True)

    indices = np.array(keep).ravel().astype(int) if keep is not None else np.array([])
    merged = [candidates[i] for i in indices if 0 <= i < len(candidates)]
    return sorted(merged, key=lambda f: f.confidence, reverse=True)


class FaceDetector:
    """Detects every human face in an image.

    The detector performs a full-image pass plus, for sufficiently large
    images, a grid of overlapping tile passes. Results from all passes are
    merged with non-maximum suppression, so a face found by several passes is
    reported once while small faces that only the tile passes can resolve are
    still reported.

    Example:
        >>> detector = FaceDetector(confidence_threshold=0.5)
        >>> faces = detector.detect_faces(image)
        >>> if detector.has_face(image):
        ...     ...
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.NMS_THRESHOLD,
        detector_width: int = config.DETECTOR_WIDTH,
        detector_height: int = config.DETECTOR_HEIGHT,
        backend: str = "auto",
        device: str = config.DEVICE_CPU,
        tile_grid: int = config.TILE_GRID,
        tile_overlap: float = config.TILE_OVERLAP,
        models_dir: Path = config.MODELS_DIR,
        min_face_size: int = config.MIN_FACE_SIZE_PX,
    ) -> None:
        """Build a detector, selecting and loading a backend.

        Args:
            confidence_threshold: Minimum score for a face to count.
            nms_threshold: IoU threshold used when merging passes.
            detector_width: Detector input width in pixels.
            detector_height: Detector input height in pixels.
            backend: ``"auto"``, ``"yunet"``, ``"ssd"`` or ``"haar"``.
            device: ``"cpu"`` or ``"cuda"``. Falls back to CPU if CUDA is
                unavailable, reporting the fallback.
            tile_grid: Tiles per axis for the multi-scale passes; 1 disables them.
            tile_overlap: Fractional overlap between neighbouring tiles.
            models_dir: Folder holding the model files.
            min_face_size: Discard detections smaller than this, in pixels.

        Raises:
            DetectorInitializationError: If no backend could be initialised.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.tile_grid = int(tile_grid)
        self.tile_overlap = float(tile_overlap)
        self.min_face_size = int(min_face_size)
        self._models_dir = Path(models_dir)
        self._input_size = (int(detector_width), int(detector_height))

        self.device = self._resolve_device(device)
        self.backend = self._create_backend(backend)
        _log.info(
            "Face detector ready: backend=%s device=%s confidence=%.2f tiles=%dx%d",
            self.backend.description,
            self.device,
            self.confidence_threshold,
            self.tile_grid,
            self.tile_grid,
        )

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Return the device actually used, warning on a CUDA fallback."""
        if requested != config.DEVICE_CUDA:
            return config.DEVICE_CPU
        if cuda_is_available():
            _log.info("CUDA device found; running detection on GPU.")
            return config.DEVICE_CUDA
        message = "CUDA unavailable. Falling back to CPU."
        print(message)
        _log.warning(message)
        return config.DEVICE_CPU

    def _create_backend(self, requested: str) -> DetectionBackend:
        """Instantiate the requested backend, or try each in order for ``auto``.

        Raises:
            DetectorInitializationError: If every candidate backend failed.
        """
        order = (
            ("yunet", "ssd", "haar") if requested == "auto" else (requested,)
        )

        errors: list[str] = []
        for name in order:
            try:
                return self._build_one(name)
            except DetectorError as error:
                errors.append(f"{name}: {error}")
                if requested != "auto":
                    raise
                _log.debug("Backend %s unavailable: %s", name, error)

        raise DetectorInitializationError(
            "No usable face-detection backend could be initialised.\n"
            + "\n".join(f"  - {message}" for message in errors)
            + "\n\nMost likely fix: run 'python download_model.py'."
        )

    def _build_one(self, name: str) -> DetectionBackend:
        """Construct a single backend by name."""
        if name == "yunet":
            return YuNetBackend(
                self._models_dir / config.YUNET_MODEL_FILENAME,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                input_size=self._input_size,
                device=self.device,
            )
        if name == "ssd":
            return SsdBackend(
                self._models_dir / config.SSD_PROTO_FILENAME,
                self._models_dir / config.SSD_WEIGHTS_FILENAME,
                confidence_threshold=self.confidence_threshold,
                device=self.device,
            )
        if name == "haar":
            return HaarBackend(confidence_threshold=self.confidence_threshold)
        raise DetectorInitializationError(f"Unknown backend: {name!r}")

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def detect_faces(self, image: np.ndarray) -> list[Face]:
        """Detect **all** faces in an image.

        Every detection pass runs to completion; the method never returns early
        after the first hit, which is what allows large group photographs to
        report every face they contain.

        Args:
            image: Image array as loaded by :mod:`image_processor`.

        Returns:
            A list of :class:`Face` tuples ``(x1, y1, x2, y2, confidence)`` in
            original-image pixel coordinates, sorted by descending confidence.
            Empty if no face was found.

        Raises:
            image_processor.ImageProcessingError: If the array is not a usable
                image.
        """
        prepared, scale = image_processor.preprocess_image(image)
        height, width = prepared.shape[:2]

        candidates: list[Face] = list(self.backend.detect(prepared))
        for tile in self._tiles(width, height):
            candidates.extend(self._detect_in_tile(prepared, tile))

        merged = merge_detections(
            candidates,
            nms_threshold=self.nms_threshold,
            confidence_threshold=self.confidence_threshold,
        )

        # Map back to the original resolution if preprocessing shrank the image.
        inverse = 1.0 / scale if scale not in (0.0, 1.0) else 1.0
        original_height = int(round(height * inverse))
        original_width = int(round(width * inverse))

        faces = [
            face.scaled(inverse).clipped(original_width, original_height)
            for face in merged
        ]
        return [
            face
            for face in faces
            if face.width >= self.min_face_size and face.height >= self.min_face_size
        ]

    def has_face(self, image: np.ndarray) -> bool:
        """Return whether the image contains at least one detectable face."""
        return len(self.detect_faces(image)) > 0

    # ------------------------------------------------------------------ #
    # Tiling
    # ------------------------------------------------------------------ #

    def _tiles(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Compute overlapping tile rectangles for the multi-scale passes.

        Tiling is skipped for small images, where the full-image pass already
        sees every face at adequate resolution.

        Returns:
            A list of ``(x, y, w, h)`` tiles. Empty when tiling is disabled.
        """
        if self.tile_grid <= 1:
            return []
        if max(width, height) < config.MIN_SIDE_FOR_TILING:
            return []

        grid = self.tile_grid
        overlap = self.tile_overlap
        span = grid - (grid - 1) * overlap
        tile_width = width / span
        tile_height = height / span
        step_x = tile_width * (1.0 - overlap)
        step_y = tile_height * (1.0 - overlap)

        tiles: list[tuple[int, int, int, int]] = []
        for row in range(grid):
            for column in range(grid):
                x = int(round(column * step_x))
                y = int(round(row * step_y))
                w = min(int(round(tile_width)), width - x)
                h = min(int(round(tile_height)), height - y)
                if w >= self.min_face_size and h >= self.min_face_size:
                    tiles.append((x, y, w, h))
        return tiles

    def _detect_in_tile(
        self, image: np.ndarray, tile: tuple[int, int, int, int]
    ) -> list[Face]:
        """Run the backend on one tile and translate boxes to image coordinates."""
        x, y, w, h = tile
        region = image[y : y + h, x : x + w]
        if region.size == 0:
            return []
        return [face.offset(x, y) for face in self.backend.detect(region)]
