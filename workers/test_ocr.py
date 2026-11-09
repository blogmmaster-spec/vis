"""Regression tests for the OCR scorer. No OCR engine, no GPU, no model needed.

`score_lines` decides which candidate ships once lettering is detected, and its
only client is a GPU worker that CI never boots. So the interesting cases live
here, as real detector output from the runs that went wrong, not as invented
strings: "Get Sturted for Free" is an actual button the model drew, and the
"image 2 as detected" row is the raw reading off one of those frames -- giant
headline plus button, both misspelt.

Run with:  python3 workers/test_ocr.py
"""

from __future__ import annotations

import sys

from ocr import score_lines

CTA = ["Get Started for Free"]

# name, detected lines, expected strings, coverage, extra
CASES = [
    # The happy path, both shapes the detector returns it in.
    ("perfect, one line", ["Get Started for Free"], CTA, 1.000, 0.000),
    ("perfect, split", ["Get Started", "for Free"], CTA, 1.000, 0.000),

    # The bug this file exists for. Two edits over nineteen characters is 0.947 by
    # character ratio, so this shipped as a near pass. It is a typo, and 0.75 says so.
    ("the shipped typo", ["Get Sturted for Free"], CTA, 0.750, 0.000),
    ("typo, letter transposed", ["Get Srarted for Free"], CTA, 0.750, 0.000),
    ("typo, one word", ["Get Started for Fere"], CTA, 0.750, 0.000),

    # Raw detector output off a shipped frame: the model drew the CTA twice, a giant
    # headline over its own button, and misspelt both. Three lines, one is plausibly
    # an attempt at the CTA, the other two are text nobody asked for.
    ("image 2 as detected",
     ["Get Stated", "for Freve", "Get Sturtere for Free"], CTA, 0.750, 0.667),

    # Same frame with signage in the background; the extra reading must not dilute the
    # coverage, only raise the invention tax.
    ("image 2 + background sign",
     ["Get Stated", "for Freve", "Get Sturtere for Free", "SHOP NOW"], CTA, 0.750, 0.750),

    # Correct, but drawn twice. Both copies are right, so coverage is 1.0; the duplicate
    # is still unrequested text. This is the case the similarity fallback used to excuse.
    ("correct, but twice", ["Get Started for Free", "Get Started for Free"], CTA, 1.000, 0.500),

    # The failure mode that made this module necessary: a fully invented headline.
    ("old garble", ["Sbw Vactearit Flacs ecosropide on hres"], CTA, 0.000, 1.000),

    # Nothing found. Neutral, not a pass and not a punish.
    ("no text at all", [], CTA, 0.000, 0.000),

    # Detector noise below the minimum line length is dropped before it can be taxed.
    ("noise line", ["AB"], CTA, 0.000, 0.000),

    # Nothing was asked for, so there is nothing to verify. Report a neutral reading
    # rather than inventing a score -- and note the explicit 0.0: the usual formula
    # would pay 0.25 for having found no unwanted text, which is not a result.
    ("nothing expected", ["Get Started for Free"], [], 0.000, 0.000, 0.000),
]

TOLERANCE = 0.005


def main() -> int:
    failures = 0
    width = max(len(case[0]) for case in CASES)
    for name, lines, expected, want_coverage, want_extra, *rest in CASES:
        got = score_lines(lines, expected)
        # score is defined by the other two, so derive it unless the case says otherwise.
        want_score = rest[0] if rest else round(0.75 * want_coverage + 0.25 * (1.0 - want_extra), 4)
        ok = (abs(got["coverage"] - want_coverage) <= TOLERANCE
              and abs(got["extra"] - want_extra) <= TOLERANCE
              and abs(got["score"] - want_score) <= TOLERANCE)
        print("{0} {1:<{2}}  coverage {3:.3f} (want {4:.3f})  extra {5:.3f} (want {6:.3f})  "
              "score {7:.3f} (want {8:.3f})".format(
                  "ok  " if ok else "FAIL", name, width,
                  got["coverage"], want_coverage, got["extra"], want_extra,
                  got["score"], want_score))
        if not ok:
            failures += 1
    print()
    if failures:
        print(f"{failures} of {len(CASES)} cases failed")
        return 1
    print(f"{len(CASES)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
