# Face Image Sorter

Sort a folder of images into two piles: **images that contain at least one
human face** and **images that do not**. Files that cannot be read at all go
into a third pile so nothing is silently lost.

This is **face detection**, not face recognition. The program answers exactly
one question per image:

> Does this image contain one or more faces?

It never computes identity, name, age, gender, emotion or any other
person-level attribute, and it never compares one face against another.

Originals are never moved, renamed, modified or deleted — every result is a
**copy**.

---

## Table of contents

- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Model installation](#model-installation)
- [Basic usage](#basic-usage)
- [Advanced usage](#advanced-usage)
- [Output structure](#output-structure)
- [Detection logic](#detection-logic)
- [How multi-face detection works](#how-multi-face-detection-works)
- [Command line reference](#command-line-reference)
- [Tuning the confidence threshold](#tuning-the-confidence-threshold)
- [Example output](#example-output)
- [Logging](#logging)
- [Testing](#testing)
- [Detection backends](#detection-backends)
- [GPU support](#gpu-support)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [People sorter](#people-sorter)
- [Future extensibility](#future-extensibility)

---

## Features

- **Face / no-face sorting** driven by a single rule: one or more detected
  faces means the image is a face image.
- **Group photographs are first-class.** All faces are detected, never just
  the first one. The bundled demo class photo yields 58 faces.
- **Small-face detection** via overlapping multi-scale tile passes, which
  roughly doubled the face count on a stress-test crowd image (50 → 118).
- **Recursive input search**, on by default, with `--no-recursive` to disable.
- **Case-insensitive extension matching**: `.jpg .jpeg .png .bmp .webp .tiff
  .tif`, so `photo.JPG` and `photo.jpg` behave identically.
- **Original files untouched.** Everything is copied, with timestamps
  preserved; nothing in the input tree is written to.
- **Collision-safe output naming** that keeps the source folder in the name
  (`family/photo.jpg` → `family__photo.jpg`) and falls back to numeric
  suffixes (`photo.jpg`, `photo_1.jpg`, …).
- **Extensions preserved exactly**, including case. Nothing is transcoded to
  JPEG.
- **Robust error handling.** A corrupted, empty, zero-dimension or unreadable
  image is logged, copied to `failed/`, and the run continues.
- **Optional bounding-box debug images** (`--visualize`).
- **Optional face crops** (`--crop-faces`), which never influence the
  classification.
- **Detailed log file** plus a live terminal progress bar and a final
  statistics summary.
- **Optional CUDA acceleration** with automatic, announced CPU fallback.
- **Three detection backends** selected automatically by capability.

---

## Project structure

```text
face_image_sorter/
│
├── old_files/            The original face / no-face program (`main.py`)
│                         and its tests, kept for reference. Not used by
│                         `people_sorter.py`. Run it from the project root
│                         with `PYTHONPATH=. python old_files/main.py`.
├── config.py             Every default and path in one place, plus the
│                         `Config` dataclass and its validation rules.
├── detector.py           Face detection. The `FaceDetector` facade, the three
│                         backends (YuNet / SSD / Haar), the `Face` box type,
│                         multi-scale tiling and non-maximum suppression.
├── image_processor.py    Finding, loading, validating and pre-processing
│                         images. The only module that reads image bytes.
├── file_manager.py       Output folders, collision-free copying, debug
│                         overlays and face crops. Never writes to the input.
├── logger.py             Logging setup for `logs/processing.log` and the
│                         console.
├── download_model.py     One-shot downloader for the model files.
│
├── people_sorter.py      Second entry point: people categories, good /
│                         review / bad, similar-photo groups. See
│                         "People sorter" below.
├── person_detector.py    YOLO pose person detection and keypoints.
├── face_quality.py       Head angle and eye state per face; face-to-person
│                         matching.
├── scene_classifier.py   CLIP photo-vs-artwork score and image embeddings.
├── similar_groups.py     Grouping of near-identical photos.
├── quality_rules.py      The sorting rules and their thresholds. Pure
│                         functions, no model access.
│
├── requirements.txt      Runtime dependencies.
├── requirements-dev.txt  Runtime dependencies plus pytest.
├── pytest.ini            Test configuration.
├── README.md             This file.
│
├── models/               Model files live here. Empty except for `.gitkeep`
│                         until you run `download_model.py`.
│
├── input/                Default input folder, pre-filled with a small demo
│                         set you can safely delete.
│
├── output/               Created automatically.
│   ├── faces/            Images with one or more detected faces.
│   ├── no_faces/         Images with no detected face.
│   ├── failed/           Images that could not be processed.
│   ├── debug/            Annotated copies (only with `--visualize`).
│   └── face_crops/       One image per face (only with `--crop-faces`).
│
├── logs/                 `processing.log` is written here.
│
└── tests/                pytest suite.
    ├── conftest.py              Shared fixtures and synthetic images.
    ├── test_image_processor.py  Discovery, loading, validation, letterboxing.
    ├── test_file_manager.py     Folders, duplicate names, copies, crops.
    ├── test_detector.py         Geometry, backends and real detection.
    ├── test_quality_rules.py    People sorter verdict rules.
    ├── test_similar_groups.py   Similar-photo grouping.
    └── assets/                  Two real photographs used by the tests.
```

---

## Installation

Python 3.9 or newer is required (3.10+ recommended).

```bash
cd face_image_sorter

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

To also install the test dependencies:

```bash
pip install -r requirements-dev.txt
```

### Dependencies

| Package | Why it is needed |
| --- | --- |
| `opencv-python` | Image I/O and the face-detection networks. |
| `numpy` | Array handling for image and box maths. |
| `tqdm` | Terminal progress bar. |
| `certifi` | Root certificates, used **only** by `download_model.py` so the HTTPS model download works on Python builds without a configured trust store. |
| `pytest` | Test suite only (`requirements-dev.txt`). |

Both OpenCV 4.x and OpenCV 5.x are supported. See
[Detection backends](#detection-backends) for the one behavioural difference.

---

## Model installation

The recommended detector is **YuNet**, a small (~230 KB) ONNX model from the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo).

### Automatic download

```bash
python download_model.py
```

**This command contacts `github.com`** and writes into `models/`. Nothing is
ever downloaded unless you run this script yourself; the main program only
ever reads from disk.

To additionally fetch the optional ResNet-10 SSD fallback (~10 MB):

```bash
python download_model.py --all
```

### Required files

| File | Backend | Required? | Size |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | `yunet` | Recommended | ~230 KB |
| `deploy.prototxt` | `ssd` | Optional | ~27 KB |
| `res10_300x300_ssd_iter_140000.caffemodel` | `ssd` | Optional | ~10 MB |

All of them go directly inside `models/`:

```text
face_image_sorter/models/face_detection_yunet_2023mar.onnx
```

### Manual download

If the script cannot reach the network, fetch the files with a browser or
`curl` and drop them in `models/`:

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Optional SSD fallback
curl -L -o models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt
curl -L -o models/res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### How the program verifies the model

Paths are resolved relative to the project, never hard-coded:

```python
MODELS_DIR = Path(__file__).resolve().parent / "models"
```

At startup `FaceDetector` checks that the file exists and that OpenCV can load
it. If it cannot, you get an actionable message naming the missing file and the
command that fixes it, rather than a stack trace:

```text
Error: YuNet model not found: .../models/face_detection_yunet_2023mar.onnx
Run 'python download_model.py' to fetch it, or download it manually from ...
```

To check what is installed without running a sort:

```bash
python -c "import download_model; print(download_model.verify_models())"
# {'yunet': True, 'ssd-proto': True, 'ssd-weights': True}
```

---

## Basic usage

```bash
python main.py --input ./input
```

That is all that is needed. The output folders are created automatically, and
paths with spaces work as expected:

```bash
python main.py --input "/home/user/My Pictures"
```

The bundled `input/` folder holds a small demo set that exercises every code
path: a 58-face class photograph, a single portrait, a three-face composite, a
small off-centre face, face-free images in five different formats, two
identically named files in different folders, a `.txt` file that must be
ignored, and two deliberately broken image files. Because of those two broken
files the demo run exits with code `1` — that is the documented "finished with
failures" status, not a crash. Delete the folder's contents and drop in your
own photos whenever you like.

---

## Advanced usage

**Stricter detection** (fewer false positives, may miss hard faces):

```bash
python main.py --input ./input --confidence 0.6
```

**More sensitive detection** (finds more faces, risks false positives):

```bash
python main.py --input ./input --confidence 0.4
```

**Bounding-box debug images** in `output/debug/`:

```bash
python main.py --input ./input --visualize
```

**One cropped image per detected face** in `output/face_crops/`:

```bash
python main.py --input ./input --crop-faces
```

**Higher detector resolution** to pick up smaller faces:

```bash
python main.py --input ./input --width 960 --height 960
```

**Custom output folder:**

```bash
python main.py --input ./input --output /data/sorted_photos
```

**Disable recursion** and process only the top-level folder:

```bash
python main.py --input ./input --no-recursive
```

**Choose a device.** CUDA falls back to CPU automatically if unavailable:

```bash
python main.py --input ./input --device cpu
python main.py --input ./input --device cuda
```

**Maximum recall on large group photographs:**

```bash
python main.py --input ./input --confidence 0.4 --width 960 --height 960 --tile-grid 3
```

**Fastest throughput**, at the cost of small faces:

```bash
python main.py --input ./input --tile-grid 1 --width 320 --height 320
```

**Everything at once:**

```bash
python main.py \
    --input ./input \
    --output ./output \
    --confidence 0.5 \
    --visualize \
    --crop-faces \
    --width 640 \
    --height 640 \
    --device cpu
```

---

## Output structure

Created automatically if missing. `debug/` and `face_crops/` appear only when
their flag is used, so a plain run leaves no empty folders behind.

```text
output/
├── faces/        Images with one or more detected faces
├── no_faces/     Images with no detected face
├── failed/       Images that could not be read or decoded
├── debug/        --visualize: copies with boxes and confidence labels
└── face_crops/   --crop-faces: one image per detected face
```

### File naming

The source folder is folded into the filename so that provenance stays visible
and same-named files from different folders cannot collide:

```text
input/family/photo.jpg      ->  output/faces/family__photo.jpg
input/vacation/photo.jpg    ->  output/faces/vacation__photo.jpg
input/photo.jpg             ->  output/faces/photo.jpg
```

If a name is still taken, a numeric suffix is appended: `photo.jpg`,
`photo_1.jpg`, `photo_2.jpg`, … Nothing is ever overwritten.

Extensions are preserved byte-for-byte, including case, so `image.JPG` stays
`image.JPG`. Face crops inherit the extension of their source image:

```text
group.jpg with 5 faces ->  group_face_001.jpg
                           group_face_002.jpg
                           group_face_003.jpg
                           group_face_004.jpg
                           group_face_005.jpg
```

---

## Detection logic

```text
1 or more detected faces  ->  output/faces/
0 detected faces          ->  output/no_faces/
could not be processed    ->  output/failed/
```

The entire decision is this, and nothing else:

```python
if len(detected_faces) > 0:
    destination = faces_folder
else:
    destination = no_faces_folder
```

Classification deliberately does **not** consider the number of people, body
or person detection, identity, gender, age or emotion. Only face presence
matters. An image with one face and an image with fifty faces are both simply
face images.

An image lands in `failed/` when OpenCV cannot decode it, the file is empty or
truncated, the decoded image has a zero dimension, the file cannot be read
because of permissions, or any unexpected error occurs while handling it. One
bad file never stops the run.

---

## How multi-face detection works

Finding every face in a crowd is the hardest requirement here, and it is
handled in three stages inside `detector.py`.

### 1. The backend returns all faces at once

YuNet is an anchor-based convolutional detector. A single forward pass emits
every candidate box with its own confidence score, so there is no loop to exit
early from and no "first face wins" shortcut. The image is letterboxed into
the detector's input size (640×640 by default), which preserves aspect ratio so
faces are never squashed.

### 2. Overlapping tile passes find the small faces

A face occupying 30 pixels in a 4000-pixel-wide group photo becomes about 5
pixels after the image is scaled down to 640 — far too small to detect. So
after the full-image pass, the image is split into an `N × N` grid of
**overlapping** tiles (2×2 by default) and the detector runs again on each
tile. Because each tile is letterboxed up to the full detector input size, the
network effectively sees the image at roughly double the resolution, and the
small faces become findable.

Tiles overlap by 20% by default so that a face straddling a tile boundary is
still wholly contained in at least one tile. Tiling is skipped for images under
700 pixels on the long side, where the full-image pass already resolves every
face.

### 3. Non-maximum suppression merges the passes

A face near a tile boundary may be found by the full-image pass and by two
tiles. All candidates from all passes are pooled and reduced with NMS: boxes
overlapping an already-kept box by more than the IoU threshold (0.3 by default)
are dropped, keeping the highest-scoring copy. Survivors are mapped back to
original-image pixel coordinates, clipped to the image bounds, filtered against
a minimum face size, and sorted by descending confidence.

The result is a list of plain tuples:

```python
faces = [
    (x1, y1, x2, y2, confidence),
    (x1, y1, x2, y2, confidence),
    ...
]
```

`Face` is a `NamedTuple`, so it unpacks exactly like the tuple above while also
offering `.width`, `.height`, `.area`, `.confidence` and `.as_int_box()`.

### Measured effect

On a 2560×2048 stress-test image built from two pasted copies of a class
photograph:

| `--tile-grid` | Faces found | Time |
| --- | --- | --- |
| 1 (tiling off) | 50 | 0.011 s |
| 2 (default) | **118** | 0.064 s |
| 3 | 112 | 0.115 s |

More tiles is not monotonically better: at 3×3 the individual tiles become
small enough that some faces are cut across two boundaries. 2×2 is a good
default, and `--tile-grid 3` is worth trying on very large images.

---

## Command line reference

```text
usage: main.py [-h] --input INPUT [--output OUTPUT] [--confidence CONFIDENCE]
               [--visualize] [--crop-faces] [--width WIDTH] [--height HEIGHT]
               [--recursive | --no-recursive] [--device {cpu,cuda}]
               [--backend {auto,yunet,ssd,haar}] [--tile-grid N]
               [--tile-overlap FRACTION] [--nms IOU] [--quiet] [--verbose]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | *required* | Folder of images to sort. |
| `--output` | `./output` | Where results are written. |
| `--confidence` | `0.5` | Minimum detection score, in (0.0, 1.0]. |
| `--visualize` | off | Write annotated copies to `output/debug/`. |
| `--crop-faces` | off | Write one crop per face to `output/face_crops/`. |
| `--width` | `640` | Detector input width. |
| `--height` | `640` | Detector input height. |
| `--recursive` / `--no-recursive` | recursive | Search sub-folders or not. |
| `--device` | `cpu` | `cpu` or `cuda`; `cuda` falls back to CPU if absent. |
| `--backend` | `auto` | `auto`, `yunet`, `ssd` or `haar`. |
| `--tile-grid` | `2` | `N×N` extra tile passes; `1` disables tiling. |
| `--tile-overlap` | `0.2` | Overlap between neighbouring tiles. |
| `--nms` | `0.3` | IoU threshold for merging duplicate boxes. |
| `--quiet` | off | Print only the final summary. |
| `--verbose` | off | Echo log records to the console too. |

Every default lives in `config.py` if you would rather change it permanently.

**Exit codes:** `0` all images processed; `1` the run could not start, or it
finished with at least one image in `failed/`; `2` invalid arguments.

---

## Tuning the confidence threshold

Each detection carries a score between 0 and 1. `--confidence` is the cutoff
below which a candidate is discarded.

- **Lower threshold (e.g. 0.3–0.4)** → more faces are detected, including
  small, blurred, dark and partially turned ones. The cost is more **false
  positives**: patterned fabric, foliage, car headlights, clock faces and
  faces on posters or screens can be reported as faces, which moves genuinely
  face-free images into `faces/`.
- **Higher threshold (e.g. 0.7–0.9)** → almost everything reported really is a
  face, but difficult faces get missed, which moves genuine face images into
  `no_faces/`.

The default of **0.5** is a deliberate middle ground. Which way to lean depends
on which mistake costs you more: if `no_faces/` must not contain any faces,
lower it; if `faces/` must be clean, raise it. Running with `--visualize` at
two settings and comparing `output/debug/` is the quickest way to calibrate for
your own photos.

---

## Example output

```text
========================================
           FACE IMAGE SORTER
========================================

Input folder:
  input

Images found: 11

Detector: yunet (ONNX, input 640x640) on cpu

Processing images...

[ 1/11] beach_sunset.jpg                 -> NO FACE
[ 2/11] class_photo.jpg                  -> FACE (58 faces)
[ 3/11] hills.bmp                        -> NO FACE
[ 4/11] texture.tif                      -> NO FACE
[ 5/11] portrait.JPG                     -> FACE (1 face)
[ 6/11] small_face.webp                  -> FACE (1 face)
[ 7/11] trio.png                         -> FACE (3 faces)
[ 8/11] corrupted.jpg                    -> FAILED (Image could not be decoded)
[ 9/11] empty.png                        -> FAILED (File is empty)
[10/11] photo.jpg                        -> NO FACE
[11/11] photo.jpg                        -> NO FACE

----------------------------------------
Processing Complete
----------------------------------------

Total images:                11
Images with faces:            4
Images without faces:         5
Failed images:                2
Total faces detected:        63
Average faces/face image:  15.75
Processing time:            0.29 s
Throughput:                37.52 images/s

Output:
  Faces:     .../output/faces
  No faces:  .../output/no_faces
  Failed:    .../output/failed
  Log:       .../logs/processing.log

Failures (2):
  - corrupted.jpg: Image could not be decoded: input/random/corrupted.jpg
  - empty.png: File is empty: input/random/empty.png
```

Statistics reported at the end: total images, images with faces, images without
faces, failed images, total faces detected, average faces per face-containing
image, processing time and throughput.

---

## Logging

Every run appends to `logs/processing.log`:

```text
2026-10-01 16:30:21 | INFO    | Processing started
2026-10-01 16:30:21 | INFO    | Input folder: input
2026-10-01 16:30:21 | INFO    | Output folder: output
2026-10-01 16:30:21 | INFO    | Processing: input/family/class_photo.jpg
2026-10-01 16:30:21 | INFO    | Faces detected: 58
2026-10-01 16:30:21 | INFO    | Classification: FACE
2026-10-01 16:30:21 | INFO    | Copied to output/faces/family__class_photo.jpg (0.073 s)
2026-10-01 16:30:21 | ERROR   | Failed: input/random/corrupted.jpg | Image could not be decoded
2026-10-01 16:30:22 | INFO    | === Final statistics ===
2026-10-01 16:30:22 | INFO    | Total faces detected: 63
```

The log holds the run settings, every filename, every face count and
classification, per-face boxes and confidences at debug level, every error, the
per-image duration and the final statistics. The console stays clean by
default because the progress lines and summary already carry the outcomes; use
`--verbose` to mirror log records to the terminal as well.

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Useful variations:

```bash
python -m pytest -v                       # one line per test
python -m pytest tests/test_detector.py   # a single file
python -m pytest -k "duplicate"           # tests matching a name
```

The suite is 190 tests across four files and covers all of the required cases:

| Required case | Where |
| --- | --- |
| Valid image | `test_image_processor.py::test_load_valid_image_round_trips` |
| Invalid / corrupted image | `test_load_corrupted_image_raises`, `test_load_empty_file_raises` |
| Empty folder | `test_find_images_on_empty_folder_returns_empty_list`, `test_empty_input_folder_runs_without_error` |
| Image with one face | `test_detector.py::test_image_with_one_face` |
| Image with multiple faces | `test_image_with_multiple_faces`, `test_large_group_photo_finds_many_faces` |
| Image without a face | `test_image_without_a_face`, `test_random_noise_has_no_face` |
| Duplicate filenames | `test_duplicate_names_from_different_folders_do_not_overwrite`, `test_duplicate_filenames_are_all_preserved` |
| Output directory creation | `test_create_output_directories_makes_the_three_core_folders` |

It also covers originals remaining untouched, extension-case preservation,
every supported image format, zero-dimension and oversized images, the CUDA
fallback, the tiling geometry, NMS merging, argument validation and full
end-to-end runs.

Detection tests use two real photographs in `tests/assets/` (see the README
there). If those files or the model are missing, the affected tests are
**skipped** rather than failed, so the suite still runs on a fresh checkout.

---

## Detection backends

`--backend auto` (the default) tries each in order and uses the first that
works, reporting its choice on startup and in the log.

| Backend | Model needed | Strengths | Requires |
| --- | --- | --- | --- |
| `yunet` | `face_detection_yunet_2023mar.onnx` (~230 KB) | Accurate, fast, real confidence scores, strong on small faces. **Recommended.** | OpenCV ≥ 4.5.4 |
| `ssd` | `deploy.prototxt` + `res10_...caffemodel` (~10 MB) | Well-proven, good on large frontal faces. | OpenCV 4.x only |
| `haar` | none (ships inside OpenCV) | Works with no download at all. | OpenCV 4.x only |

**OpenCV 5 removed the Caffe importer and `cv2.CascadeClassifier`**, so on
OpenCV 5 only `yunet` is available and the model download is mandatory. Asking
for an unavailable backend produces a clear message rather than a crash:

```text
Error: This OpenCV build has no Caffe importer (cv2 5.0.0); the 'ssd' backend
is unavailable. Use the 'yunet' backend, or install opencv-python<5.
```

Measured on the bundled demo folder, which is why YuNet is the default:

| Backend | Images classified as faces | Total faces found |
| --- | --- | --- |
| `yunet` | 4 | 63 |
| `ssd` | 3 | 14 |
| `haar` | 4 | 64 (includes false positives) |

Haar's raw count looks competitive but it has no genuine confidence score — one
is synthesised from the cascade's neighbour count — and it produces noticeably
more false positives on cluttered backgrounds.

---

## GPU support

```bash
python main.py --input ./input --device cuda
```

This requires an OpenCV build compiled with CUDA support. The standard
`opencv-python` wheel from PyPI is **not** CUDA-enabled, so on a default
install this prints:

```text
CUDA unavailable. Falling back to CPU.
```

and continues on the CPU. CUDA is never mandatory and the fallback is always
announced on the terminal and in the log.

If you do have a CUDA-enabled OpenCV build, `--device cuda` passes
`DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA` through to the network. Device
selection is isolated in `detector._dnn_ids()`, so adding another accelerator
means editing one small function.

CPU performance is good enough that this is rarely the bottleneck: the bundled
demo folder of 11 images sorts in about 0.3 seconds, and a 1280×1024 class
photograph with 58 faces takes roughly 60 ms including the tile passes.

---

## Limitations

Face detection is statistical, so mistakes happen in both directions.

**Faces that may be missed (ending up in `no_faces/`):**

- **Very small faces** — distant faces in large crowds, especially with tiling
  disabled. Try `--tile-grid 3`, `--width 960 --height 960`, or a lower
  `--confidence`.
- **Extremely blurred faces** — motion blur or a badly out-of-focus subject.
- **Heavy occlusion** — a face mostly hidden behind a hand, mask, scarf or
  another person. Partial visibility usually still works; most of the face
  being hidden usually does not.
- **Faces turned completely away** — the back of a head contains no face to
  detect. Profiles are detected much less reliably than frontal faces.
- **Faces rotated far from upright** — strong in-plane rotation, such as
  someone lying down or a sideways-held camera, degrades detection sharply.
- **Very dark or very low-contrast images** — night shots, heavy backlighting
  and silhouettes.
- **Very low-resolution images** — a thumbnail may simply not contain enough
  pixels of face.
- **Unusual appearance** — heavy makeup, face paint, costumes or strong
  colour casts.

**False positives (face-free images ending up in `faces/`):**

The detector reports face-*like* patterns, so these are sometimes flagged:

- Patterned fabric, foliage, rock and tree bark.
- Car front ends, clock faces, electrical sockets and other "two eyes and a
  mouth" arrangements.
- **Faces in pictures count as faces.** A photo of a poster, painting,
  magazine cover, statue, mannequin, doll or a face on a phone or TV screen
  will be classified as a face image. The program cannot tell a real person
  from a depicted one, and by design it does not try.

Raising `--confidence` reduces false positives; lowering it reduces misses. You
cannot eliminate both at once. Treat the output as a very good first pass
rather than ground truth, and use `--visualize` to see exactly what was
detected.

**Other constraints:**

- Images are capped at 6000 pixels on the long side before detection, as a
  memory safeguard. Boxes are still reported in original-image coordinates.
  The cap is `MAX_IMAGE_LONG_SIDE` in `config.py`.
- Only still images are handled; there is no video support.
- Animated GIFs and multi-page TIFFs are read as a single frame.
- Processing is single-threaded. Throughput scales with image count, not cores.

---

## Troubleshooting

**`Error: YuNet model not found`** — run `python download_model.py`.

**`CERTIFICATE_VERIFY_FAILED` while downloading the model** — your Python has
no configured certificate store. Either `pip install certifi` (already in
`requirements.txt`), or on a macOS python.org install run
`/Applications/Python 3.x/Install Certificates.command`, or download the model
manually as shown above. `python download_model.py --insecure` exists as a last
resort.

**`No supported images found`** — check the extension is in the supported list
and, if your images are in sub-folders, that you have not passed
`--no-recursive`.

**Everything lands in `no_faces/`** — confirm the detector line on startup
reports the backend you expect, then try `--confidence 0.4` and `--visualize`
to see what is being detected.

**Too many false positives** — raise `--confidence` to 0.6–0.7.

**Small faces in group photos are missed** — use `--tile-grid 3` and
`--width 960 --height 960`.

**It is too slow on a large library** — use `--tile-grid 1` and
`--width 320 --height 320`. Expect to lose the smallest faces.

**The output folder keeps growing between runs** — `output/` is never cleared
automatically, and re-runs add numbered copies rather than overwriting. Delete
it between runs if you want a clean slate.

---

## People sorter

`people_sorter.py` is a second, separate program. Where `main.py` answers "is
there a face?", it answers three further questions:

1. **Who is in the photo?** People are found by body, not by face, so someone
   turned away still counts as a human. Photos are filed by the number of main
   subjects; small background bystanders are not counted.
2. **Is it a good photo of them?** Each main subject's face is checked for
   closed eyes, a turned-away head and a face that is not visible at all.
3. **Which photos are near-identical?** Similar photos are grouped and the
   best one in each group is marked.

It does no face recognition. By default originals are only copied;
`--transfer move` moves them instead (see below).

### Usage

```bash
pip install -r requirements.txt
python people_sorter.py                        # sort ./input into ./output_sorted
python people_sorter.py --limit 300 --shuffle  # try a random sample first
python people_sorter.py --clean --yaw 60       # re-sort with a different threshold
python people_sorter.py --transfer move        # move the originals instead of copying
python people_sorter.py --restore              # put moved originals back
python people_sorter.py --help                 # every option
```

`python sorter_gui.py` opens a window for the same program: choose the photos
folder and where to save the sorted copies, then press Start. It always copies
and never moves the originals.

The first run downloads three models (YOLO pose ~40 MB, MediaPipe face
landmarker ~4 MB, CLIP ~350 MB). Expect roughly a quarter of a second per
image on an Apple-silicon Mac.

The model results are cached in `output_sorted/.analysis/`. An interrupted run
resumes where it stopped, and re-sorting with different thresholds takes
seconds plus the time to copy files. Use `--fresh` to discard the cache.

### Moving instead of copying

`--transfer move` moves each original out of the input folder into its sorted
folder, so no second copy is made. Three things follow from a file then
existing only once:

- Entries in `similar_groups/` are links to the moved file, not files.
- The input folder is left empty, so a re-sort needs the files back first:
  `python people_sorter.py --restore` returns every file to its original
  path using `report.csv`, after which you can run again.
- `--clean` refuses to delete a folder of moved originals; restore first.

### Output

```
output_sorted/
  good/    solo/ couple/ small_group/ large_group/
  review/  solo/ couple/ small_group/ large_group/
  bad/     solo/ couple/ small_group/ large_group/
  no_human/           no person found
  artwork/            paintings, drawings, statues, dolls (never judged good/bad)
  failed/             files that could not be read
  others/             only with --verdicts / --people / --other: every photo
                      whose class was left out
  similar_groups/     group_0001/ best__a.jpg, b.jpg, ...
  review_sheets/      one contact sheet per outcome, for checking at a glance
  report.csv          one row per image: category, verdict, reason, scores
```

Grouped photos appear both in their category folder and in `similar_groups/`.

To keep only some classes, name them: `--verdicts good review --people couple
small_group --other` gives `good/couple`, `good/small_group`, `review/couple`
and `review/small_group`, and places every other photo in `others/`. A photo
of people needs both its verdict and its people category named. Photos in
`others/` are left out of the similar groups and review sheets but are still
listed in `report.csv`.
On the review sheets a green box is a subject that passed, orange is
borderline and red is a clear failure.

### Rules

| Category | Main subjects |
|---|---|
| `solo` | 1 |
| `couple` | 2 |
| `small_group` | 3-5 |
| `large_group` | 6 or more |

| Reason | Meaning |
|---|---|
| `face_not_visible` | a person with no detectable face: turned back, bent down, face covered |
| `face_turned_away` | head turned more than `--yaw` degrees, looking down more than `--pitch-down`, or a profile so strong no face mesh fits |
| `eyes_closed` | both eyes closed (a wink does not count) |
| `face_unclear` | a face seems to be there but could not be measured (dark, blurred, partly hidden); always `review` |
| `couple_moment` | two close faces with closed eyes or turned heads, such as a kiss; always `review` |

How the per-subject checks become a verdict:

- **One or two subjects:** a clear failure makes the photo `bad`, a borderline
  one makes it `review`.
- **Three or more:** the photo is `bad` once `--group-bad-fraction` (default
  one third) of the subjects clearly fail. Fewer clear failures mean `review`,
  and a single borderline subject is ignored.
- **`--candid-lenient`:** photos where most subjects are not facing the camera
  go to `review` instead of `bad`. Intended for wedding candids, where looking
  down or away is often deliberate.

Blur and darkness are deliberately not checked.

### Limits

- Eye state cannot be judged behind sunglasses or on faces smaller than about
  40 pixels; those subjects pass.
- Someone looking down at the camera's level usually reads as closed eyes.
- A hard laugh that squeezes the eyes shut reads as closed eyes.
- A person's shadow or reflection is occasionally detected as a second,
  faceless person.
- The artwork check is a similarity score. Some stylised photos are filed as
  artwork and some realistic artwork as photos; `report.csv` has the score and
  `--artwork` moves the threshold.

---

## Future extensibility

The current program does exactly one thing: `IMAGE → FACE DETECTION → FACE /
NO FACE`. The structure anticipates the pipeline below, but **none of these are
implemented**, by design:

```text
Face detection   (implemented)
      ↓
Face images      (implemented: output/faces/)
      ↓
Face cropping    (implemented: --crop-faces)
      ↓
Expression detection   (not implemented)
      ↓
Person clustering      (not implemented)
      ↓
Face recognition       (not implemented)
```

What makes those additions straightforward later:

- `DetectionBackend` is an abstract base class, so a new detector is a new
  subclass plus one branch in `FaceDetector._build_one`. Nothing else changes.
- `Face` already carries the box and confidence that any downstream stage would
  need, and `--crop-faces` already produces per-face images.
- Classification is isolated in `main.classify_image`, so additional outputs
  can be added without touching the face / no-face rule.
- `config.Config` centralises settings, so new options need no changes to the
  modules that consume them.

Face recognition, identity matching and person clustering are deliberately
absent from the current code.
