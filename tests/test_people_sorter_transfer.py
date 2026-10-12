"""Tests for the people sorter's move / restore handling of files."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import people_sorter as sorter


def _moved_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Simulate a ``--transfer move`` run of one file in a sub-folder."""
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    original = input_dir / "trip" / "a.jpg"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"photo")

    sorter.write_state(output_dir, transfer_mode="move", input_dir=input_dir)
    moved = sorter.transfer(original, output_dir / "good" / "solo", "trip__a.jpg", "move")
    sorter.link_into(moved, output_dir / "similar_groups" / "group_0001", "best__trip__a.jpg")
    with (output_dir / sorter.REPORT_FILENAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "destination"])
        writer.writeheader()
        writer.writerow({"file": "trip/a.jpg",
                         "destination": moved.relative_to(output_dir).as_posix()})
    return original, moved, output_dir


def test_move_removes_the_original(tmp_path: Path) -> None:
    original, moved, _ = _moved_run(tmp_path)
    assert not original.exists()
    assert moved.read_bytes() == b"photo"


def test_group_entry_is_a_working_link(tmp_path: Path) -> None:
    _, _, output_dir = _moved_run(tmp_path)
    link = output_dir / "similar_groups" / "group_0001" / "best__trip__a.jpg"
    assert link.is_symlink()
    assert link.read_bytes() == b"photo"


def test_clean_refuses_to_delete_moved_originals(tmp_path: Path) -> None:
    _, moved, output_dir = _moved_run(tmp_path)
    with pytest.raises(FileExistsError, match="--restore"):
        sorter.prepare_output(output_dir, clean=True)
    assert moved.exists()


def test_restore_puts_files_back(tmp_path: Path) -> None:
    original, moved, output_dir = _moved_run(tmp_path)
    assert sorter.restore(output_dir) == 0
    assert original.read_bytes() == b"photo"
    assert not moved.exists()
    assert not (output_dir / "good").exists()
    assert not (output_dir / "similar_groups").exists()
    # The folder can be sorted into again afterwards.
    sorter.prepare_output(output_dir, clean=False)


def test_restore_keeps_folders_when_a_file_cannot_go_back(tmp_path: Path) -> None:
    original, moved, output_dir = _moved_run(tmp_path)
    original.write_bytes(b"something else")  # the original path is taken
    assert sorter.restore(output_dir) == 1
    assert moved.read_bytes() == b"photo"
    assert original.read_bytes() == b"something else"


def test_restore_without_a_move_run(tmp_path: Path) -> None:
    assert sorter.restore(tmp_path) == 1
