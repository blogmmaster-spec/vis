"""Imotan Vision 1 — provisioning pipeline.

Owns everything the worker does to a request that is *not* the raw diffusion call:
prompt shaping, aspect-ratio maths, candidate selection, and encoding.

The diffusion model is loaded in `handler.py`; this module stays model-agnostic so the
same pipeline keeps working when the checkpoint moves from klein 4B to the trained 5.2B.
"""

from __future__ import annotations

import hashlib
import io
import math
import re

import numpy as np
from PIL import Image

# ─────────────────────────────── aspect ratios ───────────────────────────────

# Values mirror what the market actually exposes (OpenRouter image surface):
# five common ratios plus `auto`. Adding ratios later is safe; removing them is not.
ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
}

# 1MP target. The model is a ~4MP-class architecture, but launch is 1K by design:
# it keeps latency inside the cheap tier and matches the competitor baseline.
TARGET_PIXELS = 1024 * 1024
DIM_ALIGNMENT = 16


def _align(value: float) -> int:
    return max(DIM_ALIGNMENT, int(round(value / DIM_ALIGNMENT)) * DIM_ALIGNMENT)


def dimensions_for(aspect_ratio: str | None, size: str | None) -> tuple[int, int]:
    """Resolve explicit `size` ("1344x768") or a named ratio to (width, height)."""
    if size:
        match = re.fullmatch(r"\s*(\d{2,5})\s*[x×]\s*(\d{2,5})\s*", size)
        if match:
            return _align(int(match.group(1))), _align(int(match.group(2)))

    ratio = ASPECT_RATIOS.get((aspect_ratio or "").strip())
    if ratio is None:
        return _align(TARGET_PIXELS**0.5), _align(TARGET_PIXELS**0.5)

    width = math.sqrt(TARGET_PIXELS * ratio)
    height = math.sqrt(TARGET_PIXELS / ratio)
    return _align(width), _align(height)


# ──────────────────────────────── prompt shaping ─────────────────────────────

# Deliberately conservative. A template that always appends "cinematic, 35mm" ruins
# logos and diagrams; an unmatched prompt gets NO suffix rather than a wrong one.
STYLE_SUFFIX: dict[str, str] = {
    "photo": "photorealistic, natural light, accurate colour",
    "portrait": "portrait photograph, natural skin texture, soft directional light",
    "product": "product photograph, clean background, soft even lighting",
    "illustration": "illustration, flat colour, clean line work",
    "graphic": "graphic design, bold shapes, high contrast",
    # Fallback only — used when the prompt quotes no literal string to render.
    # See `_text_suffix`: when the caller *does* quote its lettering, the suffix is
    # derived from those quotes and becomes a constraint rather than a compliment.
    #
    # Why: "legible text, crisp lettering" tells the model that text should be good,
    # not that text should be limited. At 4 steps klein reads that as licence to add
    # a headline of its own invention, and invented strings are exactly what it
    # misspells. Naming the permitted strings is what actually stops it.
    "text": "clear legible lettering, correct spelling, no caption, no watermark",
}

# Straight and curly quotes, guillemets, and backticks. A string the caller put in
# quotes is a string it wants rendered verbatim — that is the whole contract.
_QUOTED = re.compile(r"[\"'“”„«»‘’`]([^\"'“”„«»‘’`\n]{1,60})[\"'“”„«»‘’`]")

_INTENT_PATTERNS: list[tuple[str, str]] = [
    ("text", r"\b(logo|poster|flyer|banner|sign|signage|label|text|typograph|wordmark)\b"),
    ("product", r"\b(product|packshot|bottle|shoe|phone|laptop|packaging|e-?commerce)\b"),
    ("portrait", r"\b(portrait|headshot|face|selfie|model|person|man|woman|child)\b"),
    ("illustration", r"\b(illustration|drawing|cartoon|anime|comic|sketch|watercolou?r)\b"),
    ("graphic", r"\b(graphic|icon|badge|pattern|minimal|flat design)\b"),
    ("photo", r"\b(photo|photograph|realistic|cinematic|street|landscape|shot)\b"),
]


def detect_intent(prompt: str) -> str | None:
    lowered = prompt.lower()
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern, lowered):
            return intent
    return None


def expected_strings(prompt: str) -> list[str]:
    """Literal strings the caller asked to be rendered, in prompt order.

    Only quoted spans count. This is deliberate: it gives callers a lever they can
    actually pull (quote it, and it is checked and protected) and it keeps the worker
    from guessing at capitalised nouns.
    """
    seen: list[str] = []
    for match in _QUOTED.finditer(prompt):
        value = " ".join(match.group(1).split())
        if len(value) < 2 or value.lower() in {item.lower() for item in seen}:
            continue
        seen.append(value)
    return seen[:4]


def _text_suffix(prompt: str) -> str:
    """Suffix for text-intent prompts, narrowed to the quoted strings when present."""
    wanted = expected_strings(prompt)
    if not wanted:
        return STYLE_SUFFIX["text"]
    listing = ", ".join(f'"{text}"' for text in wanted)
    return (
        f"only the text {listing}, no other text, no additional words, "
        "no headline, sharp legible spelling, clean plain background behind the text"
    )


def build_prompt(prompt: str) -> tuple[str, str | None]:
    """Return (shaped prompt, detected intent).

    The user's text is never rewritten — only suffixed — so intent is preserved.
    """
    prompt = " ".join(prompt.split())
    if not prompt:
        raise ValueError("prompt is empty")

    intent = detect_intent(prompt)
    suffix = _text_suffix(prompt) if intent == "text" else STYLE_SUFFIX.get(intent or "")
    if not suffix:
        return prompt, intent
    # Avoid doubling a suffix the caller already wrote.
    if suffix.split(",")[0].strip().lower() in prompt.lower():
        return prompt, intent
    return f"{prompt}, {suffix}", intent


# ─────────────────────────────── candidate scoring ───────────────────────────

def technical_score(image: Image.Image) -> float:
    """Cheap, honest quality proxy: edge energy, contrast, colourfulness.

    This is a *technical* score, not an aesthetic one. It reliably rejects blurred,
    flat and washed-out candidates — which is most of what best-of-N needs to catch
    at 4 steps. It does not judge composition, and it should not be described as if
    it does.

    The Laplacian is computed with numpy rather than `ImageFilter.FIND_EDGES` so the
    result is exact and free of the border response that filter produces.
    """
    grey = image.convert("L")
    width, height = grey.size
    border = max(2, min(width, height) // 64)
    if width > border * 4 and height > border * 4:
        grey = grey.crop((border, border, width - border, height - border))

    pixels = np.asarray(grey, dtype=np.float32)
    if pixels.size == 0:
        return 0.0

    # 5-point Laplacian over the interior.
    interior = pixels[1:-1, 1:-1]
    laplacian = (
        -4.0 * interior
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    edge_energy = float(laplacian.var()) / 300.0
    contrast = float(pixels.std()) / 30.0

    rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
    saturation = float((rgb.max(axis=2) - rgb.min(axis=2)).mean()) / 255.0

    return 0.6 * min(edge_energy, 1.0) + 0.3 * min(contrast, 1.0) + 0.1 * saturation


# ────────────────────────────────── encoding ────────────────────────────────

MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


def encode(image: Image.Image, fmt: str, quality: int = 90) -> tuple[bytes, str]:
    fmt = fmt.lower()
    mime = MIME_TYPES.get(fmt)
    if mime is None:
        raise ValueError(f"unsupported format: {fmt}")

    if fmt in ("jpeg", "jpg"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    if mime == "image/png":
        image.save(buffer, format="PNG", optimize=True)
    else:
        image.save(buffer, format="WEBP" if mime == "image/webp" else "JPEG", quality=quality)
    return buffer.getvalue(), mime


def fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


# ────────────────────────────── safety (pre-launch) ─────────────────────────

# A prompt blocklist is NOT content moderation. It exists so internal testing cannot
# generate obvious prohibited content. Before the endpoint is public it must be
# replaced with a real classifier on both prompt and output — see IMAGE-MODEL.md §7
# and the FLUX.2 model card's guidance on filters.
BLOCKED_TERMS = (
    "csam", "child porn", "loli", "underage nude", "minor nude",
    "nonconsensual", "revenge porn", "deepfake nude",
)


def check_prompt(prompt: str) -> str | None:
    lowered = prompt.lower()
    for term in BLOCKED_TERMS:
        if term in lowered:
            return "prompt_blocked"
    return None
