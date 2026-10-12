"""People sorter: categorise photos of humans and separate the bad ones.

For every image under the input folder this program:

1. finds the people in it and files it by how many main subjects there are
   (``solo``, ``couple``, ``small_group``, ``large_group``), or as
   ``no_human`` / ``artwork``;
2. checks each main subject's face and marks the photo ``good``, ``review``
   or ``bad`` (eyes closed, face turned away, face not visible);
3. groups near-identical photos and picks the best of each group.

By default originals are left untouched and every result is a copy.
``--transfer symlink`` links instead of copying, and ``--transfer move`` moves
the originals out of the input folder; ``--restore`` puts moved files back.

Usage::

    python people_sorter.py                       # sort ./input into ./output_sorted
    python people_sorter.py --limit 300 --shuffle # try a random sample first
    python people_sorter.py --clean --yaw 60      # re-sort with a new threshold
    python people_sorter.py --transfer move       # move the originals, do not copy
    python people_sorter.py --restore             # put moved originals back

The slow part (running the models) is cached in ``<output>/.analysis``, so
re-sorting with different thresholds takes seconds.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

import cv2
import numpy as np
from tqdm import tqdm

import config
import file_manager
import image_processor
import quality_rules as rules
from detector import FaceDetector
from face_quality import (
    LANDMARKER_MODEL_FILENAME,
    LANDMARKER_MODEL_URL,
    FaceAnalyzer,
    match_faces_to_persons,
)
from logger import configure_opencv_logging, setup_logger
from person_detector import (
    KP_LEFT_EAR,
    KP_LEFT_EYE,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_EYE,
    KP_RIGHT_SHOULDER,
    POSE_MODEL_FILENAME,
    PersonDetector,
)
from scene_classifier import SceneClassifier
from similar_groups import group_similar, read_capture_time

DEFAULT_OUTPUT_DIR: Final[Path] = config.DEFAULT_SORTED_OUTPUT_DIR
LOG_FILENAME: Final[str] = "people_sorter.log"

FAILED_DIRNAME: Final[str] = "failed"
#: Holds every photo whose class was left out with ``--verdicts``, ``--people``
#: or ``--other``.
OTHERS_DIRNAME: Final[str] = "others"
SIMILAR_DIRNAME: Final[str] = "similar_groups"
SHEETS_DIRNAME: Final[str] = "review_sheets"
CACHE_DIRNAME: Final[str] = ".analysis"
CACHE_FILENAME: Final[str] = "analysis.jsonl"
REPORT_FILENAME: Final[str] = "report.csv"
#: Records how the last run placed its files, so that ``--clean`` never deletes
#: moved originals and ``--restore`` knows where they came from.
STATE_FILENAME: Final[str] = "last_run.json"

#: Everything a run writes into the output folder, removed by ``--clean``.
MANAGED_DIRS: Final[tuple[str, ...]] = (
    *rules.VERDICTS,
    rules.CATEGORY_NO_HUMAN,
    rules.CATEGORY_ARTWORK,
    FAILED_DIRNAME,
    OTHERS_DIRNAME,
    SIMILAR_DIRNAME,
    SHEETS_DIRNAME,
)

#: The classes with a folder of their own that is not split by verdict.
OTHER_CATEGORIES: Final[tuple[str, ...]] = (rules.CATEGORY_NO_HUMAN, rules.CATEGORY_ARTWORK)

#: Faces are measured for anyone at least this fraction of the largest
#: person's area. Looser than any sensible main-subject ratio, so that ratio
#: can be changed later without re-running the models.
_MEASURE_AREA_RATIO: Final[float] = 0.1
_MAX_MEASURED_PERSONS: Final[int] = 40

_HEAD_KEYPOINT_NAMES: Final[dict[str, int]] = {
    "nose": KP_NOSE,
    "left_eye": KP_LEFT_EYE,
    "right_eye": KP_RIGHT_EYE,
    "left_ear": KP_LEFT_EAR,
    "right_ear": KP_RIGHT_EAR,
    "left_shoulder": KP_LEFT_SHOULDER,
    "right_shoulder": KP_RIGHT_SHOULDER,
}

_SHEET_COLUMNS: Final[int] = 8
_SHEET_ROWS: Final[int] = 5
_SHEET_TILE: Final[int] = 240
_LEVEL_COLORS: Final[dict[str, tuple[int, int, int]]] = {  # BGR
    rules.LEVEL_OK: (0, 200, 0),
    rules.LEVEL_SOFT: (0, 165, 255),
    rules.LEVEL_HARD: (0, 0, 255),
}


# ---------------------------------------------------------------------- #
# Command line
# ---------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    defaults = rules.Settings()
    parser = argparse.ArgumentParser(
        prog="people_sorter.py",
        description=(
            "Categorise photos by the people in them, separate bad photos "
            "(eyes closed, face turned away or hidden) and group similar ones."
        ),
    )
    parser.add_argument("-i", "--input", type=Path, default=config.DEFAULT_INPUT_DIR,
                        help="Folder containing the images (default: ./input).")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Folder for the sorted copies (default: ./output_sorted).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many images.")
    parser.add_argument("--shuffle", action="store_true",
                        help="With --limit, take a random sample instead of the first N.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for --shuffle and the review sheets.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the previous sorted output first (the analysis "
                             "cache is kept).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore the analysis cache and run the models again.")
    parser.add_argument("--transfer", choices=("copy", "symlink", "move"), default="copy",
                        help="Copy files (default), create symlinks, or move the "
                             "originals out of the input folder.")
    parser.add_argument("--restore", action="store_true",
                        help="Move files from an earlier '--transfer move' run back to "
                             "where they came from, then exit.")
    parser.add_argument("--device", default="auto",
                        help="Torch device: auto, mps, cuda or cpu (default: auto).")
    parser.add_argument("--person-confidence", type=float, default=0.4,
                        help="Minimum person detection score (default: 0.4).")
    parser.add_argument("--no-sheets", action="store_true",
                        help="Do not write the review contact sheets.")
    parser.add_argument("--download-models", action="store_true",
                        help="Download the model files and exit.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Print only the final summary.")

    tuning = parser.add_argument_group("sorting rules")
    tuning.add_argument("--main-ratio", type=float, default=defaults.main_subject_area_ratio,
                        help="A person is a main subject when their box is at least this "
                             "fraction of the largest person's (default: %(default)s).")
    tuning.add_argument("--artwork", type=float, default=defaults.artwork_threshold,
                        help="Artwork score at which an image is filed as artwork "
                             "(default: %(default)s).")
    tuning.add_argument("--yaw", type=float, default=defaults.yaw_hard,
                        help="Head turn in degrees that makes a photo bad "
                             "(default: %(default)s).")
    tuning.add_argument("--pitch-down", type=float, default=defaults.pitch_down_hard,
                        help="Looking-down angle in degrees that makes a photo bad "
                             "(default: %(default)s).")
    tuning.add_argument("--eye-openness", type=float, default=defaults.eye_closed_hard,
                        help="Eye openness (lid gap / eye width) at or below which eyes "
                             "count as closed; raise to flag more (default: %(default)s).")
    tuning.add_argument("--group-bad-fraction", type=float,
                        default=defaults.group_bad_fraction,
                        help="In groups of 3+, fraction of failing subjects that makes "
                             "the photo bad rather than review (default: %(default)s).")
    tuning.add_argument("--candid-lenient", action="store_true",
                        help="Send failing candid (not camera-facing) photos to review "
                             "instead of bad. Suited to wedding candids.")

    classes = parser.add_argument_group(
        "classes to keep",
        "Photos of any other class are placed together in 'others' and take no "
        "part in the similar groups or review sheets.",
    )
    classes.add_argument("--verdicts", nargs="*", choices=rules.VERDICTS,
                         default=list(rules.VERDICTS), metavar="VERDICT",
                         help="Verdicts that get their own folder (default: all of "
                              "%(choices)s).")
    classes.add_argument("--people", nargs="*", choices=rules.PEOPLE_CATEGORIES,
                         default=list(rules.PEOPLE_CATEGORIES), metavar="CATEGORY",
                         help="People categories that get their own folder (default: "
                              "all of %(choices)s).")
    classes.add_argument("--other", nargs="*", choices=OTHER_CATEGORIES,
                         default=list(OTHER_CATEGORIES), metavar="CATEGORY",
                         help="Further classes that get their own folder (default: "
                              "all of %(choices)s).")

    similar = parser.add_argument_group("similar photos")
    similar.add_argument("--no-similar", action="store_true",
                         help="Skip grouping of similar photos.")
    similar.add_argument("--similar-threshold", type=float, default=0.93,
                         help="Similarity (0-1) at which two photos are grouped; "
                              "lower groups more loosely (default: %(default)s).")
    similar.add_argument("--max-seconds-apart", type=float, default=None,
                         help="Only group photos taken within this many seconds, when "
                              "both carry a capture time.")
    return parser


def build_settings(args: argparse.Namespace) -> rules.Settings:
    """Build the rule thresholds from parsed arguments.

    Each soft (review) limit keeps its default distance below the hard limit.
    """
    defaults = rules.Settings()
    return rules.Settings(
        main_subject_area_ratio=max(args.main_ratio, _MEASURE_AREA_RATIO),
        artwork_threshold=args.artwork,
        yaw_hard=args.yaw,
        yaw_soft=args.yaw - (defaults.yaw_hard - defaults.yaw_soft),
        pitch_down_hard=args.pitch_down,
        pitch_down_soft=args.pitch_down
        - (defaults.pitch_down_hard - defaults.pitch_down_soft),
        eye_closed_hard=args.eye_openness,
        eye_closed_soft=args.eye_openness
        + (defaults.eye_closed_soft - defaults.eye_closed_hard),
        group_bad_fraction=args.group_bad_fraction,
        candid_lenient=args.candid_lenient,
    )


# ---------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------- #


def download_models(models_dir: Path) -> None:
    """Fetch the model files that are not downloaded automatically."""
    import download_model

    for spec in (download_model.YUNET,):
        ok, message = download_model.ensure_model(spec, models_dir)
        print(f"  {spec.filename}: {message}")
        if not ok:
            raise RuntimeError(message)

    landmarker = models_dir / LANDMARKER_MODEL_FILENAME
    if landmarker.exists():
        print(f"  {landmarker.name}: already present")
    else:
        download_model.download_file(LANDMARKER_MODEL_URL, landmarker)
        print(f"  {landmarker.name}: downloaded")


class ImageAnalyzer:
    """Runs every model on an image and records plain measurements."""

    def __init__(self, models_dir: Path, *, device: str, person_confidence: float) -> None:
        self.persons = PersonDetector(
            models_dir / POSE_MODEL_FILENAME,
            confidence_threshold=person_confidence,
            device=device,
        )
        self.faces = FaceDetector(models_dir=models_dir)
        self.face_analyzer = FaceAnalyzer(models_dir / LANDMARKER_MODEL_FILENAME)
        self.scene = SceneClassifier(device=device)

    def close(self) -> None:
        """Release native model resources."""
        self.face_analyzer.close()

    def analyze(self, image: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        """Measure one BGR image.

        Returns:
            ``(record, embedding)``: a JSON-serialisable dictionary of
            measurements and the image's similarity embedding.
        """
        image = image_processor.to_bgr(image)
        height, width = image.shape[:2]
        persons = self.persons.detect(image)
        embedding, image_artwork, subject_artwork, artwork_label = self.scene.analyze(
            image, persons[0].box if persons else None
        )
        records: list[dict[str, Any]] = []
        if persons:
            matches = match_faces_to_persons(persons, self.faces.detect_faces(image))
            largest = persons[0].area
            for index, person in enumerate(persons):
                measured = (
                    index < _MAX_MEASURED_PERSONS
                    and person.area >= _MEASURE_AREA_RATIO * largest
                )
                face = matches.get(index) if measured else None
                records.append(
                    {
                        "box": list(person.box),
                        "confidence": person.confidence,
                        "head_keypoints": {
                            name: round(person.keypoint_confidence(kp), 2)
                            for name, kp in _HEAD_KEYPOINT_NAMES.items()
                        },
                        "head_size": round(person.estimated_head_size(), 1),
                        "face": (
                            asdict(self.face_analyzer.measure(image, face))
                            if face is not None
                            else None
                        ),
                    }
                )

        record = {
            "width": width,
            "height": height,
            "persons": records,
            # Both the whole frame and the main person must look like artwork;
            # a real person next to a painting then stays a photo.
            "artwork_score": min(image_artwork, subject_artwork),
            "artwork_score_image": image_artwork,
            "artwork_score_subject": subject_artwork,
            "artwork_label": artwork_label,
        }
        return record, embedding


# ---------------------------------------------------------------------- #
# Analysis cache
# ---------------------------------------------------------------------- #


def _encode_embedding(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.astype(np.float16).tobytes()).decode("ascii")


def _decode_embedding(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float16).astype(np.float32)


def load_cache(cache_file: Path) -> dict[str, dict[str, Any]]:
    """Read previously analysed records, keyed by file path relative to input."""
    cached: dict[str, dict[str, Any]] = {}
    if not cache_file.is_file():
        return cached
    with cache_file.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by an interrupted run
            cached[record["file"]] = record
    return cached


def _prefetch(paths: Sequence[Path], *, workers: int = 4, ahead: int = 8
              ) -> Iterator[tuple[Path, Future]]:
    """Decode images on background threads, a bounded number at a time."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[tuple[Path, Future]] = deque()
        iterator = iter(paths)
        while True:
            while len(pending) < ahead:
                path = next(iterator, None)
                if path is None:
                    break
                pending.append((path, pool.submit(image_processor.load_image, path)))
            if not pending:
                return
            yield pending.popleft()


def analyze_images(
    images: Sequence[Path],
    args: argparse.Namespace,
    log: logging.Logger,
) -> list[dict[str, Any]]:
    """Return one analysis record per image, running models only where needed."""
    cache_file = args.output / CACHE_DIRNAME / CACHE_FILENAME
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and cache_file.exists():
        cache_file.unlink()
    cached = load_cache(cache_file)

    keys = {path: path.resolve().relative_to(args.input.resolve()).as_posix()
            for path in images}
    todo = [path for path in images if keys[path] not in cached]
    log.info("Analysis: %d cached, %d to run", len(images) - len(todo), len(todo))

    if todo:
        if not args.quiet:
            print("Loading models (first run downloads them)...")
        analyzer = ImageAnalyzer(
            config.MODELS_DIR, device=args.device, person_confidence=args.person_confidence
        )
        try:
            with cache_file.open("a", encoding="utf-8") as handle:
                progress = tqdm(_prefetch(todo), total=len(todo), unit="img",
                                desc="Analysing", disable=args.quiet)
                for path, future in progress:
                    record: dict[str, Any] = {"file": keys[path]}
                    try:
                        measurements, embedding = analyzer.analyze(future.result())
                        record.update(measurements)
                        record["embedding"] = _encode_embedding(embedding)
                        record["capture_time"] = read_capture_time(path)
                    except Exception as error:  # noqa: BLE001 - never abort the run
                        log.error("Failed: %s (%s)", path, error)
                        record["error"] = f"{type(error).__name__}: {error}"
                    cached[record["file"]] = record
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
        finally:
            analyzer.close()

    return [cached[keys[path]] for path in images]


# ---------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------- #


def _state_file(output_dir: Path) -> Path:
    return output_dir / CACHE_DIRNAME / STATE_FILENAME


def read_state(output_dir: Path) -> dict[str, Any]:
    """Return what the last run recorded about itself, or an empty mapping."""
    try:
        return json.loads(_state_file(output_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(output_dir: Path, *, transfer_mode: str, input_dir: Path) -> None:
    """Record how this run places its files."""
    path = _state_file(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"transfer": transfer_mode, "input": str(input_dir.resolve())}),
        encoding="utf-8",
    )


def prepare_output(output_dir: Path, *, clean: bool) -> None:
    """Make sure the output folder holds no results from an earlier run.

    Raises:
        FileExistsError: If earlier results exist and ``clean`` is ``False``,
            or if they are moved originals, which are never deleted.
    """
    existing = [output_dir / name for name in MANAGED_DIRS if (output_dir / name).exists()]
    if existing and read_state(output_dir).get("transfer") == "move":
        raise FileExistsError(
            f"{output_dir} holds originals moved there by an earlier run, so it "
            "will not be cleaned.\nRun 'python people_sorter.py --restore' first "
            "to put them back, or choose another --output folder."
        )
    if existing and not clean:
        raise FileExistsError(
            f"{output_dir} already contains sorted results "
            f"({', '.join(path.name for path in existing)}).\n"
            "Pass --clean to replace them, or choose another --output folder."
        )
    for path in existing:
        shutil.rmtree(path)
    output_dir.mkdir(parents=True, exist_ok=True)


def transfer(source: Path, directory: Path, filename: str, mode: str) -> Path:
    """Place ``source`` into ``directory`` by copy, symlink or move."""
    if mode == "copy":
        return file_manager.copy_image(source, directory, filename=filename)
    directory.mkdir(parents=True, exist_ok=True)
    destination = file_manager.unique_destination(directory, filename)
    if mode == "move":
        shutil.move(str(source), str(destination))
    else:
        os.symlink(source.resolve(), destination)
    return destination


def link_into(target: Path, directory: Path, filename: str) -> Path:
    """Create a relative symlink to ``target`` inside ``directory``.

    Relative, so the link survives the output folder being moved as a whole.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = file_manager.unique_destination(directory, filename)
    os.symlink(os.path.relpath(target.resolve(), directory.resolve()), destination)
    return destination


def restore(output_dir: Path) -> int:
    """Undo a ``--transfer move`` run using its report.

    Every moved file goes back to its original path. The sorted folders, which
    then hold only links and contact sheets, are removed; the report and the
    analysis cache are kept. Returns a process exit code.
    """
    state = read_state(output_dir)
    report_path = output_dir / REPORT_FILENAME
    if state.get("transfer") != "move" or not report_path.is_file():
        print(f"Nothing to restore: {output_dir} does not hold a '--transfer move' run.")
        return 1

    input_dir = Path(state["input"])
    restored, skipped = 0, 0
    with report_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["destination"]:
                continue
            current = output_dir / row["destination"]
            original = input_dir / row["file"]
            if current.is_symlink() or not current.is_file() or original.exists():
                skipped += 1
                continue
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current), str(original))
            restored += 1

    print(f"Restored {restored} files to {input_dir}.")
    if skipped:
        # Something is not where the report says; leave everything in place
        # rather than risk deleting a file that was not put back.
        print(f"{skipped} files were missing or already present and were left "
              f"alone; the sorted folders in {output_dir} were not removed.")
        return 1
    for name in MANAGED_DIRS:
        shutil.rmtree(output_dir / name, ignore_errors=True)
    write_state(output_dir, transfer_mode="restored", input_dir=input_dir)
    return 0


def is_selected(result: rules.ImageResult, args: argparse.Namespace) -> bool:
    """Whether the image's class is one the user asked to keep.

    A photo of people needs both its verdict and its people category selected.
    """
    if result.category in OTHER_CATEGORIES:
        return result.category in args.other
    return result.verdict in args.verdicts and result.category in args.people


def _max_or_blank(values: list[float]) -> str:
    return f"{max(values):.2f}" if values else ""


def report_row(
    record: dict[str, Any],
    result: rules.ImageResult | None,
    settings: rules.Settings,
    destination: Path | None,
    group: tuple[int, int, bool] | None,
) -> dict[str, Any]:
    """Build one ``report.csv`` row."""
    row: dict[str, Any] = {
        "file": record["file"],
        "category": result.category if result else FAILED_DIRNAME,
        "verdict": result.verdict if result else "",
        "reason": result.reason if result else record.get("error", ""),
        "style": result.style if result else "",
        "people_total": result.people_total if result else "",
        "main_subjects": result.main_subjects if result else "",
        "subjects_failing": result.subjects_failing if result else "",
        "subjects_borderline": result.subjects_borderline if result else "",
        "artwork_score": record.get("artwork_score", ""),
        "artwork_label": record.get("artwork_label", ""),
        "max_abs_yaw": "",
        "max_pitch_down": "",
        "min_eye_openness": "",
        "group_id": group[0] if group else "",
        "group_rank": group[1] if group else "",
        "is_best_in_group": group[2] if group else "",
        "destination": destination.as_posix() if destination else "",
    }
    faces = [
        person["face"]
        for person in rules.main_subjects(record.get("persons") or [], settings)
        if person.get("face") and person["face"]["mesh_found"]
    ]
    if faces:
        row["max_abs_yaw"] = _max_or_blank([abs(face["yaw"]) for face in faces])
        row["max_pitch_down"] = _max_or_blank([face["pitch"] for face in faces])
        row["min_eye_openness"] = f"{min(face['eye_openness'] for face in faces):.2f}"
    return row


def _thumbnail(
    path: Path, record: dict[str, Any], result: rules.ImageResult, settings: rules.Settings
) -> np.ndarray | None:
    """Render one annotated, letterboxed tile for a contact sheet."""
    try:
        image = image_processor.to_bgr(image_processor.load_image(path))
    except image_processor.ImageProcessingError:
        return None

    thickness = max(2, round(max(image.shape[:2]) / 250))
    subjects = rules.main_subjects(record.get("persons") or [], settings)
    for subject, outcome in zip(subjects, result.subjects):
        box = (subject.get("face") or subject)["box"]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), _LEVEL_COLORS[outcome.level], thickness)

    height, width = image.shape[:2]
    scale = _SHEET_TILE / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    tile = np.full((_SHEET_TILE, _SHEET_TILE, 3), 30, dtype=np.uint8)
    top = (_SHEET_TILE - resized.shape[0]) // 2
    left = (_SHEET_TILE - resized.shape[1]) // 2
    tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return tile


def write_review_sheets(
    entries: Sequence[tuple[Path, dict[str, Any], rules.ImageResult]],
    output_dir: Path,
    settings: rules.Settings,
    *,
    seed: int,
) -> int:
    """Write one contact sheet per outcome so results can be checked at a glance.

    Boxes are green for a subject that passed, orange for borderline and red
    for a clear failure.
    """
    buckets: dict[str, list[tuple[Path, dict[str, Any], rules.ImageResult]]] = {}
    for entry in entries:
        result = entry[2]
        name = "__".join(part for part in (result.verdict, result.reason) if part)
        buckets.setdefault(name or result.category, []).append(entry)

    sheets_dir = output_dir / SHEETS_DIRNAME
    sheets_dir.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    capacity = _SHEET_COLUMNS * _SHEET_ROWS
    written = 0
    for name, bucket in sorted(buckets.items()):
        sample = randomizer.sample(bucket, min(capacity, len(bucket)))
        tiles = [tile for tile in (_thumbnail(*entry, settings) for entry in sample)
                 if tile is not None]
        if not tiles:
            continue
        blank = np.full_like(tiles[0], 30)
        tiles += [blank] * (-len(tiles) % _SHEET_COLUMNS)
        rows = [np.hstack(tiles[i : i + _SHEET_COLUMNS])
                for i in range(0, len(tiles), _SHEET_COLUMNS)]
        cv2.imwrite(str(sheets_dir / f"{name}.jpg"), np.vstack(rows))
        written += 1
    return written


# ---------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    """Execute a full sorting run. Returns a process exit code."""
    log = setup_logger(config.LOGS_DIR / LOG_FILENAME, console_level=logging.CRITICAL)
    configure_opencv_logging()
    settings = build_settings(args)
    started = time.perf_counter()

    images = image_processor.find_images(args.input, exclude_dirs=(args.output,))
    if args.limit is not None:
        if args.shuffle:
            random.Random(args.seed).shuffle(images)
        images = sorted(images[: args.limit])
    if not images:
        print(f"No supported images found in {args.input}.")
        return 1

    prepare_output(args.output, clean=args.clean)
    write_state(args.output, transfer_mode=args.transfer, input_dir=args.input)
    log.info("=" * 60)
    log.info("People sorter started: %d images from %s", len(images), args.input)
    log.info("Settings: %s", {f.name: getattr(settings, f.name) for f in fields(settings)})
    if not args.quiet:
        print(f"Input : {args.input}\nOutput: {args.output}\nImages: {len(images)}\n")

    records = analyze_images(images, args, log)
    results = [
        None if record.get("error") else rules.assess_image(record, settings)
        for record in records
    ]
    selected = [result is not None and is_selected(result, args) for result in results]

    # Similar photos: group, then rank each group best-first.
    group_of: dict[int, tuple[int, int, bool]] = {}
    if not args.no_similar:
        usable = [i for i, record in enumerate(records)
                  if record.get("embedding") and selected[i]]
        if len(usable) > 1:
            groups = group_similar(
                np.stack([_decode_embedding(records[i]["embedding"]) for i in usable]),
                threshold=args.similar_threshold,
                capture_times=[records[i].get("capture_time") for i in usable],
                max_seconds_apart=args.max_seconds_apart,
            )
            for group_id, group in enumerate(groups, start=1):
                ranked = sorted(
                    (usable[member] for member in group),
                    key=lambda i: rules.group_rank_key(results[i], records[i]),
                )
                for rank, index in enumerate(ranked, start=1):
                    group_of[index] = (group_id, rank, rank == 1)

    # Place the files and build the report.
    rows: list[dict[str, Any]] = []
    sheet_entries: list[tuple[Path, dict[str, Any], rules.ImageResult]] = []
    folders: Counter[str] = Counter()
    iterator = tqdm(zip(images, records, results), total=len(images), unit="img",
                    desc="Sorting", disable=args.quiet)
    for index, (path, record, result) in enumerate(iterator):
        name = file_manager.relative_name(path, args.input)
        if result is None:
            folder = FAILED_DIRNAME
        else:
            folder = result.folder if selected[index] else OTHERS_DIRNAME
        destination: Path | None = None
        try:
            destination = transfer(path, args.output / folder, name, args.transfer)
            group = group_of.get(index)
            if group:
                group_dir = args.output / SIMILAR_DIRNAME / f"group_{group[0]:04d}"
                group_name = f"best__{name}" if group[2] else name
                if args.transfer == "move":
                    # A moved file exists once, so its group entry is a link.
                    link_into(destination, group_dir, group_name)
                else:
                    transfer(path, group_dir, group_name, args.transfer)
        except (file_manager.FileManagerError, OSError) as error:
            log.error("Could not place %s: %s", path, error)
        folders[folder] += 1
        if result and selected[index]:
            moved = args.transfer == "move" and destination is not None
            sheet_entries.append((destination if moved else path, record, result))
        rows.append(report_row(
            record, result, settings,
            destination.relative_to(args.output) if destination else None,
            group_of.get(index),
        ))
        log.info("%s -> %s%s", record["file"], folder,
                 f" ({result.reason})" if result and result.reason else "")

    report_path = args.output / REPORT_FILENAME
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sheets = 0
    if not args.no_sheets:
        sheets = write_review_sheets(sheet_entries, args.output, settings, seed=args.seed)

    print_summary(folders, results, group_of, len(images), sheets,
                  args.output, time.perf_counter() - started)
    log.info("Finished: %s", dict(folders))
    return 0


def print_summary(
    folders: Counter[str],
    results: Sequence[rules.ImageResult | None],
    group_of: dict[int, tuple[int, int, bool]],
    total: int,
    sheets: int,
    output_dir: Path,
    seconds: float,
) -> None:
    """Print the end-of-run summary table."""
    reasons = Counter(
        (result.verdict, result.reason)
        for result in results
        if result and result.verdict in (rules.VERDICT_REVIEW, rules.VERDICT_BAD)
    )
    print("\n" + "=" * 52)
    print(f"Images processed : {total}")
    for folder in sorted(folders):
        print(f"  {folder:<24} {folders[folder]:>6}")
    if reasons:
        print("Reasons")
        for (verdict, reason), count in sorted(reasons.items()):
            print(f"  {verdict + ' / ' + reason:<32} {count:>6}")
    group_count = len({group[0] for group in group_of.values()})
    print(f"Similar groups   : {group_count} ({len(group_of)} photos)")
    if sheets:
        print(f"Review sheets    : {sheets} in {output_dir / SHEETS_DIRNAME}")
    print(f"Report           : {output_dir / REPORT_FILENAME}")
    print(f"Time             : {seconds:.0f} s")
    print("=" * 52)


def main(argv: list[str] | None = None) -> int:
    """Program entry point."""
    args = build_parser().parse_args(argv)
    try:
        if args.download_models:
            download_models(config.MODELS_DIR)
            return 0
        if args.restore:
            return restore(args.output)
        if not (config.MODELS_DIR / LANDMARKER_MODEL_FILENAME).exists():
            download_models(config.MODELS_DIR)
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, NotADirectoryError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - last-resort, reported cleanly
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
