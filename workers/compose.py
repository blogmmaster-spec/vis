"""Draw the caller's string onto a render, instead of asking the model to spell it.

Why this exists
---------------
Four-step klein draws lettering as an image, not as type. A measured run makes the
consequence concrete: a prompt quoting "Get Sorted for Free" came back with coverage 0.75
on three of four candidates -- one word of four substituted, reproducibly -- and an earlier
prompt quoting "Get Started for Free" came back reading "Get Sturted". Changing the quoted
word did not change the render, because the quote is a hint the text encoder averages into a
CTA-shaped idea, not a contract it copies.

No amount of best-of-N fixes a model that substitutes. So for `intent: "text"` the render
becomes the *artwork* and the CTA is drawn here, with a real font, once. Spelling is then
structural: the glyphs come from the font file, so the string cannot come back wrong.

Design notes
------------
* A pill behind the text is the default, not a flourish. It makes contrast independent of
  whatever the model put there, so composition never depends on the model obeying "leave a
  clean area" — a promise 4-step klein does not keep.
* Text size is fitted to a fraction of the canvas, so the same string works at 1024 and at
  2048 without the caller passing point sizes.
* Placement and fit use the **painted ink box**, not the font's metric box. A metric box
  carries ascent and descent the glyphs never touch, so centring with it leaves the type
  sitting visibly low in its plate — the difference between type that looks set and type
  that looks pasted on.
* The font is bundled (Anton, SIL OFL 1.1), never fetched at runtime: a boot that depends on
  a CDN is a boot that can fail for a reason the caller cannot see.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Anton-Regular.ttf")

# Band the text may occupy, as fractions of the canvas. Conservative on purpose: a CTA that
# touches the edge reads as a mistake, and a plate needs room to breathe.
MAX_TEXT_WIDTH = 0.70
MAX_TEXT_HEIGHT = 0.13
MARGIN = 0.075
PADDING_X = 0.42  # of the font size
PADDING_Y = 0.32

_ACCENT = (232, 32, 41)
_INK = (255, 255, 255)


def _ink_box(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Bounding box of the pixels `text` actually paints, in draw coordinates.

    Measured by rendering into a mask and asking for its bbox, because `textbbox` reports the
    font's ascent/descent: for a string with no descenders that box is taller than the type,
    and the extra height lands below the glyphs, pulling them up in the plate.
    """
    pad = font.size + 8
    width = pad * 2 + int(font.size * max(1, len(text)) * 0.8)
    mask = Image.new("L", (max(1, width), pad * 3))
    ImageDraw.Draw(mask).text((pad, pad), text, font=font, fill=255)
    box = mask.getbbox()
    if box is None:  # whitespace only
        return (0, 0, 0, 0)
    left, top, right, bottom = box
    return (left - pad, top - pad, right - pad, bottom - pad)


def _fits(text: str, font: ImageFont.FreeTypeFont, max_width: int, max_height: int) -> bool:
    left, top, right, bottom = _ink_box(text, font)
    return (right - left) <= max_width and (bottom - top) <= max_height


def _fit(text: str, max_width: int, max_height: int) -> ImageFont.FreeTypeFont:
    """Largest size at which the painted ink fits the box.

    Starts from a proportional guess off one measurement at 200px, then walks down in small
    steps: measuring is cheap, but a linear scan from 400 would measure fifty times per image.
    """
    probe = 200
    font = ImageFont.truetype(FONT_PATH, probe)
    left, top, right, bottom = _ink_box(text, font)
    width = max(1, right - left)
    height = max(1, bottom - top)
    size = max(12, int(probe * min(max_width / width, max_height / height)))
    while size > 12:
        font = ImageFont.truetype(FONT_PATH, size)
        if _fits(text, font, max_width, max_height):
            return font
        size -= 1
    return ImageFont.truetype(FONT_PATH, 12)


def _ink_for(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """White on a dark plate, near-black on a light one."""
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    return _INK if luminance < 150 else (17, 17, 17)


def compose(
    image: Image.Image,
    text: str,
    *,
    position: str = "bottom",
    accent: tuple[int, int, int] = _ACCENT,
    ink: tuple[int, int, int] | None = None,
    pill: bool = True,
) -> Image.Image:
    """Return a copy of `image` with `text` drawn on it. Never mutates the input.

    `position` is `bottom`, `center` or `top`. `pill=False` draws bare type with an offset
    shadow instead of a plate — legible over a quiet background, and the caller's choice.
    """
    text = " ".join(text.split())
    if not text:
        return image.copy()

    canvas = image.convert("RGB")
    width, height = canvas.size
    font = _fit(text, int(width * MAX_TEXT_WIDTH), int(height * MAX_TEXT_HEIGHT))
    left, top, right, bottom = _ink_box(text, font)
    text_width, text_height = right - left, bottom - top

    pad_x = int(font.size * PADDING_X)
    pad_y = int(font.size * PADDING_Y)
    plate_width = text_width + 2 * pad_x
    plate_height = text_height + 2 * pad_y

    margin = int(height * MARGIN)
    x = (width - plate_width) // 2
    if position == "top":
        y = margin
    elif position == "center":
        y = (height - plate_height) // 2
    else:
        y = height - margin - plate_height

    # Centre the painted ink inside the plate: the text origin has to be pulled back by the
    # ink box's own offset, or the glyphs land wherever the font's baseline happens to put them.
    text_x = x + (plate_width - text_width) // 2 - left
    text_y = y + (plate_height - text_height) // 2 - top

    draw = ImageDraw.Draw(canvas)
    if pill:
        draw.rounded_rectangle(
            (x, y, x + plate_width, y + plate_height),
            radius=plate_height // 2,
            fill=accent,
        )
        colour = ink or _ink_for(accent)
    else:
        # Bare type: a single offset dark copy, which survives most backgrounds without the
        # blur dependency a soft shadow would need.
        draw.text((text_x + 2, text_y + 3), text, font=font, fill=(0, 0, 0))
        colour = ink or _INK

    draw.text((text_x, text_y), text, font=font, fill=colour)
    return canvas
