"""Optional OCR, used only to score lettering on text-intent images.

Why this exists
---------------
`technical_score` measures edge energy, contrast and saturation. It is blind to
*what* the pixels say, so best-of-N on a text-intent prompt happily keeps a
candidate that invented a garbled headline, because invented lettering is just as
crisp and contrasty as correct lettering.

This module reads the lettering back out and scores it against the strings the
caller actually asked for. Two things fall out:

  * `coverage` — did the requested strings survive, spelt correctly?
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


def _agreement(candidate: str, expected: str) -> float:
    """How much of `expected` is present in `candidate` *correctly*, in 0..1.

    Character similarity alone cannot judge short strings. On a 19-character CTA,
    dropping one letter and swapping another is two edits — `SequenceMatcher` calls
    that 0.947, so a button reading "Get Sturted for Free" scored as a near pass and
    was shipped. The caller got an ad with a typo in it and a receipt saying 95%.

    A word is either spelt right or it is not, so that is the unit for short strings:
    agreement is capped by the share of expected words that appear verbatim. The
    character ratio is still taken, because it is what tolerates OCR noise on long
    copy, but it can only lower the result.
    """
    if not candidate or not expected:
        return 0.0
    want = expected.split()
    if not want:
        return 0.0
    have = set(candidate.split())
    spelt = sum(1 for word in want if word in have) / len(want)
    return min(_best_ratio(candidate, expected), spelt)


def score_lines(
    lines: list[str],
    expected: list[str],
) -> dict[str, float]:
    """Score OCR output against the strings the caller asked for.

    Returns `{"coverage", "extra", "score"}`, all in 0..1:

      coverage — mean best-match agreement across the expected strings
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

    # Cover two ways the same text can arrive: one line, or a phrase the detector
    # split across two. Without the join, a correctly spelt string broken in half
    # scores like a string with a missing word.
    haystack = got + ([" ".join(got)] if len(got) > 1 else [])

    ratios = ([max(_agreement(text, target) for text in haystack) for target in want]
              if haystack else [0.0] * len(want))
    coverage = sum(ratios) / len(ratios)

    # Which lines are doing requested work, and which are text nobody asked for.
    #
    # The requested words are a *budget*, not a pattern to match: "Get Started" plus
    # "for Free" on two lines is one requested string, so both lines are accounted for
    # and a correct wrap is not billed as invented. A second, correctly spelt copy of
    # those same words then finds the budget spent and does count as extra — which is
    # what catches klein repeating the CTA as a headline over its own button.
    #
    # A line carrying words outside the budget is a misspelt attempt rather than an
    # invention, so similarity may still claim it; `coverage` is what punishes the
    # spelling, and charging it twice would double-count one defect.
    budget: dict[str, int] = {}
    for target in want:
        for word in target.split():
            budget[word] = budget.get(word, 0) + 1
    vocabulary = set(budget)

    matched: set[int] = set()
    for index, line in enumerate(got):
        words = line.split()
        if all(budget.get(word, 0) >= words.count(word) for word in set(words)):
            for word in words:
                budget[word] -= 1
            matched.add(index)
        elif (
            any(word not in vocabulary for word in words)
            and max(_best_ratio(line, target) for target in want) >= 0.72
        ):
            # Words the caller never asked for, shaped closely enough to be an attempt
            # at theirs. The `any(... not in vocabulary)` guard is what keeps a second,
            # perfectly spelt copy out of here: it has no unfamiliar word, so it is a
            # duplicate and the spent budget should stand.
            matched.add(index)
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
