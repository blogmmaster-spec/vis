"""Imotan Vision 1 — post-generation stages: super-resolution and vectorisation.

Both stages are optional and both are honest about failure: if a stage cannot run it
raises, and the caller falls back to the unprocessed image **and says so in the response
metadata**. Nothing here silently substitutes an approximation (e.g. Lanczos) for real
super-resolution, because "looks like 4K" is precisely what we are not selling.
"""

from __future__ import annotations

import io
import os
import tempfile

from PIL import Image

UPSCALE_MODEL = os.environ.get("IMOTAN_UPSCALE_MODEL", "/models/realesrgan/RealESRGAN_x4plus.pth")
UPSCALE_MAX_OUTPUT_PIXELS = int(os.environ.get("IMOTAN_UPSCALE_MAX_PIXELS", str(16_777_216)))  # 4K
# Whole-image inference, not tiled. A 1MP source is ~13GB of activations at 4× on top of the
# diffusion model, so guard the input rather than risk an OOM mid-request.
UPSCALE_MAX_INPUT_PIXELS = int(os.environ.get("IMOTAN_UPSCALE_MAX_INPUT", str(2_100_000)))


class StageUnavailable(RuntimeError):
    """Raised when an optional stage is not installed or its weights are missing."""


# ──────────────────────────────── super-resolution ──────────────────────────

_upscaler = None


def _load_upscaler():
    global _upscaler
    if _upscaler is not None:
        return _upscaler

    if not os.path.exists(UPSCALE_MODEL):
        raise StageUnavailable(f"upscaler weights not found at {UPSCALE_MODEL}")

    try:
        import torch
        from spandrel import ModelLoader
    except Exception as exc:  # pragma: no cover - depends on the image
        raise StageUnavailable(f"upscaler dependencies missing: {exc}") from exc

    descriptor = ModelLoader().load_from_file(UPSCALE_MODEL)
    descriptor.eval()
    descriptor.to("cuda" if torch.cuda.is_available() else "cpu")
    _upscaler = descriptor
    return descriptor


def _run_upscaler(descriptor, image: Image.Image) -> Image.Image:
    import torch

    tensor = torch.from_numpy(_to_float_array(image)).permute(2, 0, 1).unsqueeze(0)
    device = next(descriptor.model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        output = descriptor.model(tensor)

    output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return _from_float_array(output)


def _to_float_array(image: Image.Image):
    import numpy as np

    return (np.asarray(image.convert("RGB"), dtype="float32") / 255.0).copy()


def _from_float_array(array):
    import numpy as np

    return Image.fromarray((np.clip(array, 0, 1) * 255.0).round().astype("uint8"), "RGB")


def upscale(image: Image.Image, factor: int = 2) -> tuple[Image.Image, dict]:
    """Super-resolve by `factor` (2 or 4), via a single whole-image pass of a 4× model.

    Returns (image, metadata). Raises StageUnavailable if the stage cannot run, and
    ValueError if the request would exceed a configured pixel budget.
    """
    if factor not in (2, 4):
        raise ValueError("upscale factor must be 2 or 4")

    input_pixels = image.width * image.height
    if input_pixels > UPSCALE_MAX_INPUT_PIXELS:
        raise ValueError(
            f"source is {input_pixels} px, above the {UPSCALE_MAX_INPUT_PIXELS} px upscale limit"
        )

    target_w = image.width * factor
    target_h = image.height * factor
    if target_w * target_h > UPSCALE_MAX_OUTPUT_PIXELS:
        raise ValueError(f"upscaled output would exceed {UPSCALE_MAX_OUTPUT_PIXELS} pixels")

    descriptor = _load_upscaler()
    scale = int(getattr(descriptor, "scale", 4) or 4)

    out = _run_upscaler(descriptor, image)

    if out.size != (target_w, target_h):
        out = out.resize((target_w, target_h), Image.LANCZOS)

    return out, {
        "stage": "upscale",
        "requested_factor": factor,
        "model_scale": scale,
        "output": f"{out.width}x{out.height}",
    }


# ───────────────────────────────── vectorisation ────────────────────────────

# Presets tuned per output class. Tracing is genuinely good for flat art and genuinely
# bad for photographs — the caller must choose, and the response says which was used.
VECTOR_PRESETS: dict[str, dict] = {
    "graphic": {
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "polygon",
        "filter_speckle": 4,
        "color_precision": 6,
        "corner_threshold": 60,
        "path_precision": 3,
    },
    "photo": {
        # High precision keeps more colour, but a traced photo is still a traced photo.
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 2,
        "color_precision": 8,
        "corner_threshold": 30,
        "path_precision": 5,
    },
}


def vectorize(image: Image.Image, mode: str = "graphic") -> tuple[bytes, dict]:
    """Trace a raster to SVG. Raises StageUnavailable if vtracer is missing."""
    try:
        import vtracer
    except Exception as exc:  # pragma: no cover - depends on the image
        raise StageUnavailable(f"vectoriser not installed: {exc}") from exc

    preset = VECTOR_PRESETS.get(mode)
    if preset is None:
        raise ValueError(f"unknown vector mode: {mode}")

    with tempfile.TemporaryDirectory() as tmp:
        raster_path = os.path.join(tmp, "in.png")
        svg_path = os.path.join(tmp, "out.svg")
        image.convert("RGBA").save(raster_path, format="PNG")
        vtracer.convert_image_to_svg_py(raster_path, svg_path, **preset)
        with open(svg_path, "rb") as handle:
            svg = handle.read()

    return svg, {
        "stage": "vectorize",
        "mode": mode,
        "bytes": len(svg),
        # Be explicit: traced SVG is an approximation of the raster, not designed vector art.
        "fidelity": "traced",
    }


# ──────────────────────────────── availability ──────────────────────────────

def upscaler_available() -> bool:
    """True when weights and dependencies are present — not a guess, a check."""
    if not os.path.exists(UPSCALE_MODEL):
        return False
    try:
        import spandrel  # noqa: F401
    except Exception:
        return False
    return True


def vectorizer_available() -> bool:
    try:
        import vtracer  # noqa: F401
    except Exception:
        return False
    return True
