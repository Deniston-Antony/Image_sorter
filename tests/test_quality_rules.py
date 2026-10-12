"""Tests for the people sorter's decision rules."""

from __future__ import annotations

from typing import Any

import pytest

import quality_rules as rules

SETTINGS = rules.Settings()


def face(**overrides: Any) -> dict[str, Any]:
    """A large, frontal, eyes-open face; override fields to break it."""
    base = {
        "box": [100.0, 100.0, 200.0, 200.0],
        "confidence": 0.9,
        "size": 100.0,
        "mesh_found": True,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "blink_left": 0.05,
        "blink_right": 0.05,
        "eye_openness": 0.3,
    }
    return {**base, **overrides}


def person(
    face_data: dict[str, Any] | None = None,
    *,
    box: tuple[float, float, float, float] = (50.0, 50.0, 250.0, 650.0),
    confidence: float = 0.9,
    head: float = 0.0,
    shoulders: float = 0.9,
) -> dict[str, Any]:
    """A person record; ``head`` is the nose/eye keypoint confidence."""
    return {
        "box": list(box),
        "confidence": confidence,
        "head_keypoints": {
            "nose": head,
            "left_eye": head,
            "right_eye": head,
            "left_shoulder": shoulders,
            "right_shoulder": shoulders,
        },
        "head_size": 100.0,
        "face": face_data,
    }


def image(*persons: dict[str, Any], artwork: float = 0.0) -> dict[str, Any]:
    return {"file": "a.jpg", "persons": list(persons), "artwork_score": artwork}


CLOSED = {"eye_openness": 0.05, "blink_left": 0.6, "blink_right": 0.6}


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, rules.CATEGORY_NO_HUMAN),
        (1, rules.CATEGORY_SOLO),
        (2, rules.CATEGORY_COUPLE),
        (5, rules.CATEGORY_SMALL_GROUP),
        (6, rules.CATEGORY_LARGE_GROUP),
    ],
)
def test_category_for_count(count: int, expected: str) -> None:
    assert rules.category_for_count(count, SETTINGS) == expected


def test_no_persons_is_no_human() -> None:
    result = rules.assess_image(image(), SETTINGS)
    assert result.category == rules.CATEGORY_NO_HUMAN
    assert result.folder == "no_human"


def test_frontal_open_eyes_is_good() -> None:
    result = rules.assess_image(image(person(face())), SETTINGS)
    assert (result.category, result.verdict) == (rules.CATEGORY_SOLO, rules.VERDICT_GOOD)
    assert result.folder == "good/solo"
    assert result.style == rules.STYLE_POSED


def test_closed_eyes_is_bad() -> None:
    result = rules.assess_image(image(person(face(**CLOSED))), SETTINGS)
    assert (result.verdict, result.reason) == (rules.VERDICT_BAD, rules.REASON_EYES_CLOSED)


def test_wink_is_not_closed_eyes() -> None:
    wink = face(eye_openness=0.3, blink_left=0.7, blink_right=0.05)
    assert rules.assess_image(image(person(wink)), SETTINGS).verdict == rules.VERDICT_GOOD


def test_small_face_is_not_judged() -> None:
    tiny = face(size=20.0, **CLOSED)
    assert rules.assess_image(image(person(tiny)), SETTINGS).verdict == rules.VERDICT_GOOD


@pytest.mark.parametrize("angles", [{"yaw": 70.0}, {"yaw": -70.0}, {"pitch": 45.0}])
def test_turned_head_is_bad(angles: dict[str, float]) -> None:
    result = rules.assess_image(image(person(face(**angles))), SETTINGS)
    assert (result.verdict, result.reason) == (
        rules.VERDICT_BAD,
        rules.REASON_FACE_TURNED_AWAY,
    )


def test_slightly_turned_head_is_review() -> None:
    result = rules.assess_image(image(person(face(yaw=42.0))), SETTINGS)
    assert result.verdict == rules.VERDICT_REVIEW


def test_person_without_face_is_bad() -> None:
    result = rules.assess_image(image(person(None)), SETTINGS)
    assert (result.verdict, result.reason) == (
        rules.VERDICT_BAD,
        rules.REASON_FACE_NOT_VISIBLE,
    )


def test_undetected_frontal_face_is_only_review() -> None:
    # The pose model sees nose and eyes, so the face is there but unclear
    # (dark or blurred) - not hidden.
    result = rules.assess_image(image(person(None, head=0.9)), SETTINGS)
    assert (result.verdict, result.reason) == (
        rules.VERDICT_REVIEW,
        rules.REASON_FACE_UNCLEAR,
    )


def test_artwork_is_exempt_from_quality_rules() -> None:
    result = rules.assess_image(image(person(face(**CLOSED)), artwork=0.95), SETTINGS)
    assert result.category == rules.CATEGORY_ARTWORK
    assert result.verdict == ""
    assert result.folder == "artwork"


def test_background_bystander_is_ignored() -> None:
    bystander = person(None, box=(600.0, 100.0, 650.0, 250.0))
    result = rules.assess_image(image(person(face()), bystander), SETTINGS)
    assert (result.category, result.verdict) == (rules.CATEGORY_SOLO, rules.VERDICT_GOOD)
    assert result.people_total == 2


def test_person_with_much_smaller_face_is_a_bystander() -> None:
    # Similar body boxes (the near person is cropped), but a far smaller face.
    far = person(face(size=30.0, yaw=80.0), box=(300.0, 50.0, 500.0, 650.0))
    result = rules.assess_image(image(person(face()), far), SETTINGS)
    assert (result.category, result.verdict) == (rules.CATEGORY_SOLO, rules.VERDICT_GOOD)


def test_faceless_fragment_is_ignored() -> None:
    # A large faceless detection with no shoulders: an arm or a shadow.
    fragment = person(None, box=(300.0, 50.0, 500.0, 650.0), shoulders=0.1)
    result = rules.assess_image(image(person(face()), fragment), SETTINGS)
    assert (result.category, result.verdict) == (rules.CATEGORY_SOLO, rules.VERDICT_GOOD)


def test_lone_fragment_is_still_a_subject() -> None:
    result = rules.assess_image(image(person(None, shoulders=0.1)), SETTINGS)
    assert (result.category, result.verdict) == (rules.CATEGORY_SOLO, rules.VERDICT_BAD)


def test_couple_with_one_turned_back_is_bad() -> None:
    back = person(None, box=(300.0, 50.0, 500.0, 650.0))
    result = rules.assess_image(image(person(face()), back), SETTINGS)
    assert (result.category, result.verdict) == (rules.CATEGORY_COUPLE, rules.VERDICT_BAD)


def test_couple_moment_is_review() -> None:
    left = person(face(box=[100.0, 100.0, 200.0, 200.0], **CLOSED))
    right = person(
        face(box=[190.0, 100.0, 290.0, 200.0], **CLOSED), box=(150.0, 50.0, 350.0, 650.0)
    )
    result = rules.assess_image(image(left, right), SETTINGS)
    assert (result.verdict, result.reason) == (
        rules.VERDICT_REVIEW,
        rules.REASON_COUPLE_MOMENT,
    )


def _group(failing: int, total: int) -> dict[str, Any]:
    members = [person(face(**CLOSED)) for _ in range(failing)]
    members += [person(face()) for _ in range(total - failing)]
    return image(*members)


@pytest.mark.parametrize(
    ("failing", "total", "expected"),
    [
        (0, 5, rules.VERDICT_GOOD),
        (1, 5, rules.VERDICT_REVIEW),
        (2, 5, rules.VERDICT_BAD),
        (1, 3, rules.VERDICT_REVIEW),
        (2, 3, rules.VERDICT_BAD),
    ],
)
def test_group_tolerance(failing: int, total: int, expected: str) -> None:
    assert rules.assess_image(_group(failing, total), SETTINGS).verdict == expected


def test_one_borderline_subject_in_group_is_ignored() -> None:
    members = [person(face(yaw=42.0))] + [person(face()) for _ in range(4)]
    assert rules.assess_image(image(*members), SETTINGS).verdict == rules.VERDICT_GOOD


def test_candid_lenient_downgrades_bad_to_review() -> None:
    lenient = rules.Settings(candid_lenient=True)
    record = image(person(face(yaw=70.0)))
    assert rules.assess_image(record, SETTINGS).verdict == rules.VERDICT_BAD
    assert rules.assess_image(record, lenient).verdict == rules.VERDICT_REVIEW


def test_group_rank_prefers_good_then_open_eyes() -> None:
    good_open = image(person(face(eye_openness=0.35)))
    good_squint = image(person(face(eye_openness=0.2)))
    bad = image(person(face(**CLOSED)))
    ranked = sorted(
        (bad, good_squint, good_open),
        key=lambda record: rules.group_rank_key(
            rules.assess_image(record, SETTINGS), record
        ),
    )
    assert ranked == [good_open, good_squint, bad]
