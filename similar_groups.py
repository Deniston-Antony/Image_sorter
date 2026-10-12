"""Grouping of near-identical photos.

Photos are compared by the cosine similarity of their CLIP embeddings (see
:mod:`scene_classifier`). Two photos are linked when they are similar enough
and, if both carry a capture time, were taken close together. Linked photos
form a group; a photo with no link stays ungrouped.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME = 306
_EXIF_IFD_POINTER = 34665


def read_capture_time(path: Path) -> float | None:
    """Return a photo's EXIF capture time as a POSIX timestamp, if it has one."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif = image.getexif()
            raw = exif.get_ifd(_EXIF_IFD_POINTER).get(_EXIF_DATETIME_ORIGINAL) or exif.get(
                _EXIF_DATETIME
            )
        if not raw:
            return None
        return datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:  # noqa: BLE001 - missing or malformed EXIF is normal
        return None


class _DisjointSet:
    """Minimal union-find over ``range(size)``."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[max(root_a, root_b)] = min(root_a, root_b)


def group_similar(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.93,
    capture_times: Sequence[float | None] | None = None,
    max_seconds_apart: float | None = None,
    chunk_size: int = 2048,
) -> list[list[int]]:
    """Group images whose embeddings are near-identical.

    Args:
        embeddings: ``(n, d)`` array of L2-normalised vectors.
        threshold: Minimum cosine similarity for two images to be linked.
        capture_times: Optional capture timestamp per image.
        max_seconds_apart: When set, two images that *both* have a capture
            time are linked only if taken within this many seconds.
        chunk_size: Rows compared per step, bounding memory use.

    Returns:
        Groups of two or more image indices, each sorted ascending, ordered
        by their first member.
    """
    count = len(embeddings)
    if count < 2:
        return []

    vectors = np.asarray(embeddings, dtype=np.float32)
    times = (
        np.array(
            [np.nan if t is None else float(t) for t in capture_times], dtype=np.float64
        )
        if capture_times is not None and max_seconds_apart is not None
        else None
    )

    sets = _DisjointSet(count)
    for start in range(0, count, chunk_size):
        similarity = vectors[start : start + chunk_size] @ vectors.T
        rows, columns = np.nonzero(similarity >= threshold)
        for row, column in zip(rows + start, columns):
            if column <= row:
                continue
            if times is not None:
                gap = abs(times[row] - times[column])
                if not np.isnan(gap) and gap > max_seconds_apart:
                    continue
            sets.union(int(row), int(column))

    members: dict[int, list[int]] = {}
    for index in range(count):
        members.setdefault(sets.find(index), []).append(index)
    return sorted(
        (group for group in members.values() if len(group) > 1),
        key=lambda group: group[0],
    )
