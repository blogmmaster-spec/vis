"""Optional OCR, used only to score lettering on text-intent images.

Why this exists
---------------
`technical_score` measures edge energy, contrast and saturation. It is blind to
*what* the pixels say, so best-of-N on a text-intent prompt happily keeps a
candidate that invented a garbled headline, because invented lettering is just as
crisp and contrasty as correct lettering.

This module reads the lettering back out and scores it against the strings the
caller actually asked for. Two things fall out:

  * `coverage` — did the requested strings survive?
  * `extra`    — how much *unrequested* text is on the image?

The second is what the 4-step klein model keeps failing: given a banner prompt it
invents a headline ("Sbw Vactearit Flacs ecosropide on hres") that was never asked
for. Penalising extra text is therefore the point, not a nice-to-have.

Dependency posture
------------------
`rapidocr-onnxruntime` is CPU-only, ~15 MB of models, and does not pull torch, so it
can run while the GPU is busy with the diffusion pipeline. It is installed on a
best-effort basis exactly like torchao (`|| true` in the Dockerfile): if it is
missing the worker still boots and `available()` returns False, and the caller
falls back to the technical score with no fake result.

That fallback is the dangerous part, because it is *silent from the outside*: the
images simply come back worse and nothing says why. `reason()` exists so the first
failure is carriable into the response instead of being one line in a boot log
nobody reads. The failure it was written for is worth knowing about — the base
image has no libGL, rapidocr pulls the GL-linked `opencv-python` wheel, and the
import dies on `libGL.so.1` long after pip reported success.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from PIL import Image

_engine = None
_load_failed = False
_reason: str | None = None

# A detected line shorter than this is treated as noise (icon noise, a stray glyph)
# and left out of the "invented text" tax rather than punished at full weight.
_MIN_LINE_CHARS = 3

_NOISE = re.compile(r"[^a-z0-9 ]+")


def _normalise(text: str) -> str:
    return " ".join(_NOISE.sub(" ", text.lower()).split())


def available() -> bool:
    """True when the OCR engine can be constructed. Cached after the first attempt."""
    return _get_engine() is not None


def reason() -> str | None:
    """The exception that stopped the engine loading, or None when it loaded.

    Only the first failure is kept. The engine is retried on every request, so later
    attempts would just repeat the same line and bury the one that explains it.
    """
    _get_engine()
    return _reason


def _get_engine():
    global _engine, _load_failed, _reason
    if _engine is not None or _load_failed:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        _engine = RapidOCR()
    except Exception as exc:
        _load_failed = True
        _engine = None
        _reason = f"{type(exc).__name__}: {exc}".strip()[:300]
    return _engine


def read_lines(image: Image.Image) -> list[str]:
    """Detected text lines, in reading order. Empty list when OCR is unavailable."""
    engine = _get_engine()
    if engine is None:
        return []
    try:
        import numpy as np

        result, _ = engine(np.asarray(image.convert("RGB")))
    except Exception:
        return []
    if not result:
        return []
    return [str(entry[1]).strip() for entry in result if len(entry) > 1 and entry[1]]


def _best_ratio(candidate: str, expected: str) -> float:
    """Similarity in 0..1, tolerant of a line being a fragment of the expected string.

    `SequenceMatcher.ratio()` alone punishes a correct-looking long string that OCR
    split across two lines, so partial matching is scored by the shorter/longer
    length ratio as well and the better of the two is kept.
    """
    if not candidate or not expected:
        return 0.0
    ratio = SequenceMatcher(None, candidate, expected).ratio()
    shorter, longer = sorted((len(candidate), len(expected)))
    if longer and candidate in expected:
        ratio = max(ratio, shorter / longer)
    return ratio


def score_lines(
    lines: list[str],
    expected: list[str],
) -> dict[str, float]:
    """Score OCR output against the strings the caller asked for.

    Returns `{"coverage", "extra", "score"}`, all in 0..1:

      coverage — mean best-match ratio across the expected strings
      extra    — share of detected lines that match nothing expected
      score    — 0.75 * coverage + 0.25 * (1 - extra)
    """
    got = [_normalise(line) for line in lines]
    got = [line for line in got if len(line) >= _MIN_LINE_CHARS]
    want = [_normalise(text) for text in expected]
    want = [text for text in want if text]

    if not want:
        # Nothing to verify against: don't pretend, report a neutral reading.
        return {"coverage": 0.0, "extra": 0.0, "score": 0.0}

    ratios: list[float] = []
    matched: set[int] = set()
    for target in want:
        best, best_index = 0.0, -1
        for index, line in enumerate(got):
            value = _best_ratio(line, target)
            if value > best:
                best, best_index = value, index
        ratios.append(best)
        if best >= 0.72 and best_index >= 0:
            matched.add(best_index)

    coverage = sum(ratios) / len(ratios)
    extra = (len(got) - len(matched)) / len(got) if got else 0.0
    extra = max(0.0, min(1.0, extra))

    return {
        "coverage": round(coverage, 4),
        "extra": round(extra, 4),
        "score": round(0.75 * coverage + 0.25 * (1.0 - extra), 4),
    }


def score(image: Image.Image, expected: list[str]) -> dict[str, float] | None:
    """OCR the image and score it. None when the engine is unavailable."""
    if _get_engine() is None:
        return None
    lines = read_lines(image)
    return score_lines(lines, expected)
