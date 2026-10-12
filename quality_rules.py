"""Decision rules for the people sorter.

Everything here is a pure function of the measurements recorded during
analysis, with no model or image access. That keeps the rules easy to test and
lets a finished analysis be re-sorted with different thresholds in seconds
(``people_sorter.py --reuse-analysis``).

Three questions are answered per image:

1. **Category** - how many main subjects are there, or is it artwork / a photo
   with no human at all?
2. **Verdict** - ``good``, ``review`` or ``bad``, from the per-subject checks.
3. **Reason** - why, when the verdict is not ``good``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

CATEGORY_NO_HUMAN: Final[str] = "no_human"
CATEGORY_ARTWORK: Final[str] = "artwork"
CATEGORY_SOLO: Final[str] = "solo"
CATEGORY_COUPLE: Final[str] = "couple"
CATEGORY_SMALL_GROUP: Final[str] = "small_group"
CATEGORY_LARGE_GROUP: Final[str] = "large_group"
PEOPLE_CATEGORIES: Final[tuple[str, ...]] = (
    CATEGORY_SOLO,
    CATEGORY_COUPLE,
    CATEGORY_SMALL_GROUP,
    CATEGORY_LARGE_GROUP,
)

VERDICT_GOOD: Final[str] = "good"
VERDICT_REVIEW: Final[str] = "review"
VERDICT_BAD: Final[str] = "bad"
VERDICTS: Final[tuple[str, ...]] = (VERDICT_GOOD, VERDICT_REVIEW, VERDICT_BAD)

REASON_FACE_NOT_VISIBLE: Final[str] = "face_not_visible"
REASON_FACE_TURNED_AWAY: Final[str] = "face_turned_away"
REASON_EYES_CLOSED: Final[str] = "eyes_closed"
REASON_FACE_UNCLEAR: Final[str] = "face_unclear"
REASON_COUPLE_MOMENT: Final[str] = "couple_moment"

LEVEL_OK: Final[str] = "ok"
LEVEL_SOFT: Final[str] = "soft"
LEVEL_HARD: Final[str] = "hard"

STYLE_POSED: Final[str] = "posed"
STYLE_CANDID: Final[str] = "candid"


@dataclass(slots=True)
class Settings:
    """Thresholds for the sorting rules.

    Each check has a *hard* limit (a clear failure) and a *soft* limit (a
    borderline case that is sent to ``review`` instead of ``bad``).
    """

    #: A person is a main subject when their box area is at least this
    #: fraction of the largest person's.
    main_subject_area_ratio: float = 0.25

    #: A person whose face is smaller than this fraction of the largest face
    #: is further from the camera and treated as a bystander.
    main_subject_face_ratio: float = 0.4

    #: Largest head count still called a small group.
    small_group_max: int = 5

    #: Artwork score at or above which an image with people is ``artwork``.
    artwork_threshold: float = 0.75

    #: Faces smaller than this (pixels) are too small to judge eyes or pose.
    min_face_size: float = 28.0

    #: Head turn, in degrees, beyond which the face counts as turned away.
    yaw_hard: float = 50.0
    yaw_soft: float = 38.0
    #: Looking down, in degrees.
    pitch_down_hard: float = 35.0
    pitch_down_soft: float = 25.0
    #: Looking up, in degrees.
    pitch_up_hard: float = 40.0
    pitch_up_soft: float = 30.0

    #: Eye openness (lid gap over eye width, more open eye) at or below
    #: which the eyes count as closed. Open eyes are around 0.3.
    eye_closed_hard: float = 0.085
    eye_closed_soft: float = 0.115
    #: The landmarker's own eye-closed score must agree at least this much.
    blink_hard: float = 0.4
    blink_soft: float = 0.3
    #: Faces smaller than this (pixels) are too blurry to judge eyes.
    min_face_size_for_eyes: float = 40.0

    #: A person with no face only counts as a subject when the detector is
    #: this sure of them and sees both shoulders. This drops arms, shadows
    #: and people cut off by the frame edge.
    faceless_min_confidence: float = 0.5

    #: Head keypoint confidence above which a nose/eye is "seen".
    keypoint_visible: float = 0.5

    #: In groups of three or more, the photo is ``bad`` once this fraction of
    #: main subjects clearly fails. Below it, a clear failure means ``review``
    #: and a lone borderline subject is ignored.
    group_bad_fraction: float = 0.34

    #: Two faces closer than this many face-widths are treated as an
    #: intentional couple moment (kiss, looking at each other).
    couple_moment_distance: float = 2.2

    #: When ``True``, clear failures in candid (not camera-facing) photos are
    #: sent to ``review`` instead of ``bad``. Suited to wedding candids.
    candid_lenient: bool = False

    #: Yaw/pitch within which a face counts as facing the camera.
    posed_yaw: float = 25.0
    posed_pitch: float = 20.0


@dataclass(slots=True)
class SubjectResult:
    """Outcome of the quality checks for one main subject."""

    level: str
    reason: str = ""
    judged: bool = True


@dataclass(slots=True)
class ImageResult:
    """Outcome of the sorting rules for one image."""

    category: str
    verdict: str = ""
    reason: str = ""
    style: str = ""
    people_total: int = 0
    main_subjects: int = 0
    subjects_failing: int = 0
    subjects_borderline: int = 0
    subjects: list[SubjectResult] = field(default_factory=list)

    @property
    def folder(self) -> str:
        """Output sub-folder for this image, relative to the output root."""
        if self.category in (CATEGORY_NO_HUMAN, CATEGORY_ARTWORK):
            return self.category
        return f"{self.verdict}/{self.category}"


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _is_whole_person(person: Mapping[str, Any], settings: Settings) -> bool:
    """Whether a faceless detection is a real, reasonably complete person."""
    if person.get("face") is not None:
        return True
    keypoints = person.get("head_keypoints") or {}
    return person.get("confidence", 0.0) >= settings.faceless_min_confidence and all(
        keypoints.get(name, 0.0) >= settings.keypoint_visible
        for name in ("left_shoulder", "right_shoulder")
    )


def main_subjects(
    persons: Sequence[Mapping[str, Any]], settings: Settings
) -> list[Mapping[str, Any]]:
    """Return the people a photo is about.

    Background bystanders (a much smaller body or face than the largest) are
    dropped, and so are faceless fragments such as an arm at the frame edge - unless
    that would leave nobody, in which case the largest person is kept.
    """
    if not persons:
        return []
    largest = max(persons, key=lambda person: _area(person["box"]))
    largest_area = _area(largest["box"])
    if largest_area <= 0:
        return []
    largest_face = max(
        (person["face"]["size"] for person in persons if person.get("face")), default=0.0
    )
    subjects = [
        person
        for person in persons
        if _area(person["box"]) >= settings.main_subject_area_ratio * largest_area
        and _is_whole_person(person, settings)
        and (
            person.get("face") is None
            or person["face"]["size"] >= settings.main_subject_face_ratio * largest_face
        )
    ]
    return subjects or [largest]


def category_for_count(count: int, settings: Settings) -> str:
    """Map a number of main subjects to a people category."""
    if count <= 0:
        return CATEGORY_NO_HUMAN
    if count == 1:
        return CATEGORY_SOLO
    if count == 2:
        return CATEGORY_COUPLE
    if count <= settings.small_group_max:
        return CATEGORY_SMALL_GROUP
    return CATEGORY_LARGE_GROUP


def assess_subject(person: Mapping[str, Any], settings: Settings) -> SubjectResult:
    """Run the quality checks on one main subject."""
    face = person.get("face")
    keypoints = person.get("head_keypoints") or {}
    nose_seen = keypoints.get("nose", 0.0) >= settings.keypoint_visible
    eyes_seen = sum(
        keypoints.get(name, 0.0) >= settings.keypoint_visible
        for name in ("left_eye", "right_eye")
    )

    if face is None:
        too_small = person.get("head_size", 0.0) < settings.min_face_size
        if nose_seen and eyes_seen == 2:
            # The pose model sees a frontal head the face detector missed
            # (dark, tiny, blurred). That is not evidence of a hidden face.
            if too_small:
                return SubjectResult(LEVEL_OK, judged=False)
            return SubjectResult(LEVEL_SOFT, REASON_FACE_UNCLEAR)
        return SubjectResult(LEVEL_HARD, REASON_FACE_NOT_VISIBLE)

    if face["size"] < settings.min_face_size:
        return SubjectResult(LEVEL_OK, judged=False)

    if not face["mesh_found"]:
        if face["size"] < settings.min_face_size_for_eyes:
            return SubjectResult(LEVEL_OK, judged=False)
        # A clearly visible face that no mesh fits is, in practice, a strong
        # profile, a head bent right down or a face mostly covered.
        if not nose_seen and eyes_seen == 0:
            return SubjectResult(LEVEL_HARD, REASON_FACE_NOT_VISIBLE)
        if eyes_seen == 2:
            return SubjectResult(LEVEL_SOFT, REASON_FACE_UNCLEAR)
        return SubjectResult(LEVEL_HARD, REASON_FACE_TURNED_AWAY)

    yaw, pitch = abs(face["yaw"]), face["pitch"]
    if (
        yaw >= settings.yaw_hard
        or pitch >= settings.pitch_down_hard
        or -pitch >= settings.pitch_up_hard
    ):
        return SubjectResult(LEVEL_HARD, REASON_FACE_TURNED_AWAY)

    # Both eyes must be closed (a wink is not a blink), and the lid gap and
    # the landmarker's own score must agree.
    blink = min(face["blink_left"], face["blink_right"])
    openness = face["eye_openness"]
    eyes_judged = face["size"] >= settings.min_face_size_for_eyes
    if (
        eyes_judged
        and openness <= settings.eye_closed_hard
        and blink >= settings.blink_hard
    ):
        return SubjectResult(LEVEL_HARD, REASON_EYES_CLOSED)

    if (
        yaw >= settings.yaw_soft
        or pitch >= settings.pitch_down_soft
        or -pitch >= settings.pitch_up_soft
    ):
        return SubjectResult(LEVEL_SOFT, REASON_FACE_TURNED_AWAY)
    if (
        eyes_judged
        and openness <= settings.eye_closed_soft
        and blink >= settings.blink_soft
    ):
        return SubjectResult(LEVEL_SOFT, REASON_EYES_CLOSED)
    return SubjectResult(LEVEL_OK)


def _is_facing_camera(person: Mapping[str, Any], settings: Settings) -> bool:
    face = person.get("face")
    return bool(
        face
        and face["mesh_found"]
        and abs(face["yaw"]) <= settings.posed_yaw
        and abs(face["pitch"]) <= settings.posed_pitch
    )


def _is_couple_moment(
    subjects: Sequence[Mapping[str, Any]],
    results: Sequence[SubjectResult],
    settings: Settings,
) -> bool:
    """Whether two close faces fail only by closed eyes or a turned head."""
    faces = [subject.get("face") for subject in subjects]
    if len(faces) != 2 or any(face is None for face in faces):
        return False
    if any(result.reason == REASON_FACE_NOT_VISIBLE for result in results):
        return False
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = faces[0]["box"], faces[1]["box"]
    distance = math.hypot(
        (ax1 + ax2) / 2.0 - (bx1 + bx2) / 2.0, (ay1 + ay2) / 2.0 - (by1 + by2) / 2.0
    )
    mean_size = max((faces[0]["size"] + faces[1]["size"]) / 2.0, 1.0)
    return distance <= settings.couple_moment_distance * mean_size


def assess_image(analysis: Mapping[str, Any], settings: Settings) -> ImageResult:
    """Apply the sorting rules to one image's recorded measurements."""
    persons = analysis.get("persons") or []
    subjects = main_subjects(persons, settings)
    if not subjects:
        return ImageResult(category=CATEGORY_NO_HUMAN, people_total=len(persons))

    if analysis.get("artwork_score", 0.0) >= settings.artwork_threshold:
        return ImageResult(
            category=CATEGORY_ARTWORK,
            people_total=len(persons),
            main_subjects=len(subjects),
        )

    results = [assess_subject(subject, settings) for subject in subjects]
    hard = [result for result in results if result.level == LEVEL_HARD]
    soft = [result for result in results if result.level == LEVEL_SOFT]
    facing = sum(_is_facing_camera(subject, settings) for subject in subjects)
    style = STYLE_POSED if facing >= 0.6 * len(subjects) else STYLE_CANDID

    outcome = ImageResult(
        category=category_for_count(len(subjects), settings),
        style=style,
        people_total=len(persons),
        main_subjects=len(subjects),
        subjects_failing=len(hard),
        subjects_borderline=len(soft),
        subjects=results,
    )

    outcome.verdict = _verdict(subjects, results, len(hard), len(soft), settings)
    if outcome.verdict == VERDICT_GOOD:
        return outcome

    if outcome.verdict == VERDICT_REVIEW and len(subjects) == 2 and hard:
        outcome.reason = REASON_COUPLE_MOMENT
    else:
        flagged = hard or soft
        outcome.reason = Counter(r.reason for r in flagged).most_common(1)[0][0]
    if (
        outcome.verdict == VERDICT_BAD
        and settings.candid_lenient
        and style == STYLE_CANDID
    ):
        outcome.verdict = VERDICT_REVIEW
    return outcome


def _verdict(
    subjects: Sequence[Mapping[str, Any]],
    results: Sequence[SubjectResult],
    hard: int,
    soft: int,
    settings: Settings,
) -> str:
    """Combine the per-subject outcomes into one verdict for the photo."""
    count = len(subjects)
    if count >= 3:
        # One person blinking in a group does not ruin the photo.
        if hard / count >= settings.group_bad_fraction:
            return VERDICT_BAD
        if hard or (hard + soft) / count >= settings.group_bad_fraction:
            return VERDICT_REVIEW
        return VERDICT_GOOD
    if hard:
        if count == 2 and _is_couple_moment(subjects, results, settings):
            return VERDICT_REVIEW
        return VERDICT_BAD
    return VERDICT_REVIEW if soft else VERDICT_GOOD


def group_rank_key(result: ImageResult, analysis: Mapping[str, Any]) -> tuple:
    """Sort key that puts the best photo of a similar group first."""
    openness = [
        person["face"]["eye_openness"]
        for person in analysis.get("persons") or []
        if person.get("face") and person["face"]["mesh_found"]
    ]
    verdict_order = VERDICTS.index(result.verdict) if result.verdict in VERDICTS else 0
    return (
        verdict_order,
        result.subjects_failing,
        result.subjects_borderline,
        -round(sum(openness) / len(openness), 3) if openness else 0.0,
        analysis.get("file", ""),
    )
