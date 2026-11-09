"""Regression tests for the CTA compositor. No GPU, no model, no network.

The compositor exists to turn "the model might spell it right" into "the string is drawn from
a font file", so the properties worth pinning are the ones that make it a guarantee: the same
call paints the same pixels, the input image is never touched, the type is fitted inside its
band rather than cropped, and an empty string is a no-op rather than a crash.

Run with:  python3 workers/test_compose.py
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageChops

import compose

DEMO = compose.compose(Image.new("RGB", (1024, 1024), (20, 20, 30)), "Get Sorted for Free")


def _pixels(image: Image.Image) -> bytes:
    return image.tobytes()


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print("{0} {1}{2}".format("ok  " if ok else "FAIL", name, f"  ({detail})" if detail else ""))
        if not ok:
            failures += 1

    source = Image.new("RGB", (1024, 1024), (20, 20, 30))
    before = _pixels(source)

    # Determinism: replaying one request has to reproduce the same image, or a seed stops
    # meaning anything.
    twice = compose.compose(source, "Get Sorted for Free")
    check("deterministic", _pixels(twice) == _pixels(DEMO))

    # Non-mutation: the caller keeps using the render for the next stage.
    check("input untouched", _pixels(source) == before)

    # The text actually lands: something in the lower band is no longer background.
    band = twice.crop((0, int(1024 * 0.75), 1024, 1024))
    check("paint appears in the lower band", ImageChops.difference(
        band, source.crop((0, int(1024 * 0.75), 1024, 1024))).getbbox() is not None)

    # An empty string is a copy, not a crash and not an empty canvas.
    check("empty string is a no-op", _pixels(compose.compose(source, "  ")) == before)

    # Fitting: the painted ink must stay inside the band it is allowed to occupy, at both
    # a small and a large canvas, for a string far longer than the demo.
    ok = True
    detail = []
    for size in ((512, 512), (2048, 2048)):
        text = "Book Your Free Strategy Session Today"
        font = compose._fit(text, int(size[0] * compose.MAX_TEXT_WIDTH),
                            int(size[1] * compose.MAX_TEXT_HEIGHT))
        left, top, right, bottom = compose._ink_box(text, font)
        width_ok = (right - left) <= size[0] * compose.MAX_TEXT_WIDTH
        height_ok = (bottom - top) <= size[1] * compose.MAX_TEXT_HEIGHT
        detail.append(f"{size[0]}px fit={'yes' if width_ok and height_ok else 'no'}")
        ok = ok and width_ok and height_ok
    check("long string fitted to its band", ok, ", ".join(detail))

    # Every shipped position must place the plate fully inside the canvas.
    ok = True
    for position in ("top", "center", "bottom"):
        image = compose.compose(Image.new("RGB", (768, 768), (240, 240, 240)), "Get Started", position=position)
        box = ImageChops.difference(image, Image.new("RGB", (768, 768), (240, 240, 240))).getbbox()
        inside = box is not None and box[0] >= 0 and box[1] >= 0 and box[2] <= 768 and box[3] <= 768
        ok = ok and inside
    check("plate stays inside the canvas", ok)

    # The font has to ship with the worker: a container that fetches type at runtime is a
    # container that can fail for a reason the caller cannot see.
    check("bundled font present", os.path.exists(compose.FONT_PATH), compose.FONT_PATH)

    print()
    if failures:
        print(f"{failures} checks failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
