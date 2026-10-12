"""Whole-image understanding for the people sorter, using CLIP.

One forward pass per image provides two things:

* an **artwork score**: how strongly the image looks like a painting, drawing,
  statue or cartoon rather than a photograph of real people. Artwork is exempt
  from the good/bad quality rules.
* an **embedding**: a visual fingerprint that :mod:`similar_groups` compares
  to find near-identical photos.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from logger import get_logger
from person_detector import resolve_torch_device

_log = get_logger()

CLIP_MODEL_NAME: Final[str] = "ViT-B-32-quickgelu"
CLIP_PRETRAINED: Final[str] = "openai"

_PHOTO_PROMPTS: Final[tuple[str, ...]] = (
    "a photograph of real people",
    "a photo of a real person",
    "a candid snapshot of people at an event",
    "a black and white photograph of a person",
    "a photo of a real baby",
    "a photo of a child",
)
_ARTWORK_PROMPTS: Final[tuple[str, ...]] = (
    "a painting of a person",
    "a drawing or pencil sketch",
    "a stone or bronze statue",
    "a cartoon or comic illustration",
    "an anime illustration",
    "graffiti or a mural on a wall",
    "an engraving or old print",
    "a stained glass window",
    "a plastic toy doll or figurine",
)


class SceneClassifierError(Exception):
    """Raised when the CLIP model cannot be created."""


class SceneClassifier:
    """Scores images as photo vs artwork and produces similarity embeddings."""

    def __init__(self, *, device: str = "auto") -> None:
        """Load CLIP, downloading its weights on first use.

        Raises:
            SceneClassifierError: If ``open_clip`` is missing or the model
                fails to load.
        """
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as error:
            raise SceneClassifierError(
                "The 'open_clip_torch' package is required. "
                "Install it with: pip install -r requirements.txt"
            ) from error

        self._torch = torch
        self._pil = Image
        self.device = resolve_torch_device(device)
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
            )
            tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        except Exception as error:  # noqa: BLE001 - surfaced with context
            raise SceneClassifierError(f"Could not load CLIP: {error}") from error

        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._labels = (*_PHOTO_PROMPTS, *_ARTWORK_PROMPTS)
        with torch.no_grad():
            text = self._model.encode_text(tokenizer(list(self._labels)).to(self.device))
            self._text = text / text.norm(dim=-1, keepdim=True)
        _log.info("Scene classifier ready: model=%s device=%s", CLIP_MODEL_NAME, self.device)

    @property
    def embedding_size(self) -> int:
        """Length of the vectors returned by :meth:`analyze`."""
        return int(self._text.shape[1])

    def analyze(
        self,
        image: np.ndarray,
        subject_box: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, float, float, str]:
        """Embed a BGR image and score it as artwork.

        Args:
            image: The whole image.
            subject_box: Box of the main person, if any. The artwork score is
                then also computed on that crop, which answers "is *this
                person* real?" rather than "is there art somewhere in frame?".

        Returns:
            ``(embedding, image_score, subject_score, label)``. The embedding
            is the whole image's, L2-normalised. Scores are in ``[0, 1]``,
            higher meaning more artwork-like; ``subject_score`` equals
            ``image_score`` when no box is given. ``label`` is the best
            matching artwork prompt.
        """
        torch = self._torch
        views = [image]
        if subject_box is not None:
            height, width = image.shape[:2]
            x1, y1, x2, y2 = subject_box
            margin = 0.08 * max(x2 - x1, y2 - y1)
            crop = image[
                max(0, int(y1 - margin)) : min(height, int(y2 + margin)),
                max(0, int(x1 - margin)) : min(width, int(x2 + margin)),
            ]
            if crop.size:
                views.append(crop)

        batch = torch.stack(
            [
                self._preprocess(self._pil.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
                for view in views
            ]
        ).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = (100.0 * features @ self._text.T).float().cpu().numpy()

        photo, artwork = logits[:, : len(_PHOTO_PROMPTS)], logits[:, len(_PHOTO_PROMPTS) :]
        # Best photo prompt against best artwork prompt, so the number of
        # prompts on each side does not bias the score.
        scores = 1.0 / (1.0 + np.exp(photo.max(axis=1) - artwork.max(axis=1)))
        label = _ARTWORK_PROMPTS[int(np.argmax(artwork[-1]))]
        embedding = features[0].float().cpu().numpy().astype(np.float32)
        return embedding, round(float(scores[0]), 3), round(float(scores[-1]), 3), label
