"""Imotan Vision 1 — RunPod Serverless worker.

Serves FLUX.2 [klein] 4B (distilled, 4 steps) behind an OpenAI-compatible image contract.
The endpooint is internal-only until content moderation is real: see pipeline.check_prompt.

Deploy notes are in README.md. The important operational detail is that the model is
loaded and warmed **at worker boot**, before the handler starts serving — that is what
makes RunPod's FlashBoot snapshot worth restoring (IMAGE-MODEL.md §2.2 and §6.2).
"""

from __future__ import annotations

import base64
import os
import time

import runpod

import stages
from pipeline import (
    build_prompt,
    check_prompt,
    dimensions_for,
    encode,
    technical_score,
)

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
DEFAULT_STEPS = int(os.environ.get("IMOTAN_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.environ.get("IMOTAN_GUIDANCE", "1.0"))
DEFAULT_FORMAT = os.environ.get("IMOTAN_FORMAT", "jpeg")
DEFAULT_QUALITY = int(os.environ.get("IMOTAN_QUALITY", "90"))
BEST_OF = max(1, min(int(os.environ.get("IMOTAN_BEST_OF", "2")), 3))
MAX_CANDIDATES = 6
WARMUP_DIMS = (256, 256)

_pipe = None
_active_precision = "unknown"
_active_text_precision = "unknown"


def _weights_ready(path: str) -> bool:
    """True when `path` holds at least one checkpoint shard."""
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if any(name.endswith(".safetensors") for name in files):
            return True
    return False


def _ensure_weights() -> str:
    """Resolve MODEL_PATH, downloading once if it is empty.

    * Baked image — the path already holds the weights, so this is a no-op.
    * Network volume — the first worker in a data center downloads into the mount and
      every later worker mounts that cache instead, which is the whole point of the
      volume. `IMOTAN_ALLOW_DOWNLOAD=0` disables this and fails fast instead.

    HF_HUB_OFFLINE is set only *after* the weights resolve, so a baked worker never
    touches the network while a volume-backed one still can.
    """
    model_path = os.environ.get("MODEL_PATH", "/models/flux2-klein-4b")
    if _weights_ready(model_path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return model_path

    if os.environ.get("IMOTAN_ALLOW_DOWNLOAD", "1") != "1":
        raise RuntimeError(f"no weights at {model_path} and IMOTAN_ALLOW_DOWNLOAD=0")

    print(f"[imotan] no weights at {model_path} — downloading once", flush=True)
    os.environ.pop("HF_HUB_OFFLINE", None)

    from huggingface_hub import snapshot_download

    from download_weights import ALLOW_PATTERNS, REPO_ID

    os.makedirs(model_path, exist_ok=True)
    snapshot_download(
        repo_id=os.environ.get("HF_REPO", REPO_ID),
        local_dir=model_path,
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=ALLOW_PATTERNS,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    return model_path


def _quantization_config():
    """int8 weight-only quantisation for the transformer.

    Chosen over fp8 deliberately: fp8 compute needs Ada/Hopper, so on a 3090 or A5000 it is
    emulated and slower. int8 weight-only cuts the weight footprint and the bytes moved per
    step, which is what actually binds diffusion on Ampere.

    Returns None for bf16 (IMOTAN_DTYPE=bf16).
    """
    global _active_precision, _active_text_precision

    mode = os.environ.get("IMOTAN_DTYPE", "int8").strip().lower()
    if mode in ("bf16", "bfloat16", "fp16", "none"):
        _active_precision = "bf16"
        _active_text_precision = "bf16"
        return None
    if mode not in ("int8", "8bit", "8-bit"):
        raise RuntimeError(f"IMOTAN_DTYPE must be int8 or bf16, got {mode!r}")

    # torchao is optional: version alignment is brittle, and a quantisation nicety must never
    # be able to fail a deploy. Without it the worker serves bf16 — ~16GB on a 24GB card.
    try:
        import torchao  # noqa: F401
    except Exception:
        print("[imotan] torchao not installed — falling back to bf16", flush=True)
        _active_precision = "bf16"
        _active_text_precision = "bf16"
        return None

    # Preferred: the current diffusers API. A config *object* wrapped in a quant mapping.
    #
    # Only the transformer is quantised here. Adding `text_encoder` to this mapping makes
    # `from_pretrained` die inside huggingface_hub argument validation (the component is a
    # transformers model and diffusers re-enters the hub loader for it), which crash-loops the
    # worker. Quantising the encoder needs a separate path — load it with
    # transformers.TorchAoConfig and assign it, rather than listing it in this mapping.
    try:
        from diffusers import PipelineQuantizationConfig, TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig
        from torchao.quantization.granularity import PerGroup

        config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig(Int8WeightOnlyConfig(granularity=PerGroup(128)))
            }
        )
        _active_precision = "int8-wo"
        _active_text_precision = "bf16"
        return config
    except Exception as exc:
        print(f"[imotan] modern torchao config unavailable: {exc}", flush=True)

    # Fallback for older diffusers/torchao pairs — transformer only, encoder stays bf16.
    try:
        from diffusers import TorchAoConfig

        legacy = TorchAoConfig("int8_weight_only")
        _active_precision = "int8-wo"
        _active_text_precision = "bf16"
        return legacy
    except Exception as exc:
        print(f"[imotan] legacy torchao config unavailable: {exc}", flush=True)

    _active_precision = "bf16"
    _active_text_precision = "bf16"
    return None


def _load():
    """Load the pipeline and run one throwaway inference.

    The warmup is not optional. Without it the first customer request pays for CUDA
    kernel autotuning, and FlashBoot has no warm process state to snapshot.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import Flux2KleinPipeline

    started = time.time()
    resolved = _ensure_weights()
    quant = _quantization_config()

    kwargs = {"torch_dtype": torch.bfloat16}
    if quant is not None:
        kwargs["quantization_config"] = quant

    pipe = Flux2KleinPipeline.from_pretrained(resolved, **kwargs).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Warm the kernels so the first real request is not the slow one.
    pipe(
        prompt="warmup",
        height=WARMUP_DIMS[0],
        width=WARMUP_DIMS[1],
        num_inference_steps=1,
        guidance_scale=1.0,
    )
    torch.cuda.synchronize()

    _pipe = pipe
    _report_components(pipe)
    _report_vram("after load")
    print(
        f"[imotan] loaded {MODEL_ID} ({'int8 weight-only' if quant else 'bf16'}) "
        f"and warmed up in {time.time() - started:.1f}s",
        flush=True,
    )
    return _pipe


def _component_bytes(module) -> tuple[float, str]:
    """Actual resident weight bytes and effective precision of a pipeline component.

    Must read `untyped_storage().nbytes()`, not `numel() * element_size()`: torchao wraps
    int8 weights in an AffineQuantizedTensor, whose logical `dtype`/`element_size` still
    describe bf16, so the naive product over-reports by ~2x and makes a quantised component
    look unquantised. Bytes-per-parameter from real storage is the honest signal.
    """
    seen: set[int] = set()
    total = 0
    params = 0

    for param in module.parameters():
        params += param.numel()
        try:
            storage = param.untyped_storage()
        except Exception:
            continue
        key = id(storage)
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()

    if params and total:
        bytes_per_param = total / params
        if bytes_per_param <= 1.2:
            precision = "int8"
        elif bytes_per_param <= 2.4:
            precision = "bf16"
        else:
            precision = f"{bytes_per_param:.1f}B/param"
    else:
        precision = "?"

    return total / (1024**3), precision


def _report_components(pipe) -> None:
    """Log measured size/dtype per component and correct the precision labels."""
    global _active_precision, _active_text_precision

    for name in ("transformer", "text_encoder", "vae"):
        component = getattr(pipe, name, None)
        if component is None:
            continue
        size_gb, dtype = _component_bytes(component)
        print(f"[imotan] {name}: {size_gb:.2f}GB {dtype}", flush=True)

        if name == "transformer":
            _active_precision = "int8-wo" if dtype in ("int8", "uint8") else dtype
        elif name == "text_encoder":
            _active_text_precision = "int8-wo" if dtype in ("int8", "uint8") else dtype


def _report_vram(stage: str) -> None:
    """Log real VRAM use, so capacity planning is measured rather than estimated.

    The pipeline carries a Qwen3-4B text encoder alongside the 4B transformer, so the
    whole thing is ~16GB in bf16 — noticeably more than the checkpoint's quoted figure.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(
            f"[imotan] vram {stage}: {allocated:.1f}GB allocated, "
            f"{reserved:.1f}GB reserved, {peak:.1f}GB peak, {total:.1f}GB total",
            flush=True,
        )
    except Exception as exc:  # never let diagnostics break a boot
        print(f"[imotan] vram report skipped: {exc}", flush=True)


def _generate_one(pipe, prompt: str, width: int, height: int, steps: int, guidance: float, seed: int):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    return result.images[0]


def _precision_label() -> str:
    """The precision actually loaded — not what the env asked for."""
    if _active_precision != "unknown":
        return _active_precision
    mode = os.environ.get("IMOTAN_DTYPE", "int8").strip().lower()
    return "bf16" if mode in ("bf16", "bfloat16", "fp16", "none") else "int8-wo (requested)"


class _BadRequest(ValueError):
    pass


def _parse_upscale(value) -> int:
    """Normalise the `upscale` field: false/0/"none" -> 0, otherwise 2 or 4."""
    if value in (None, False, 0, "0", "false", "none", "", "off"):
        return 0
    if isinstance(value, bool) or value is True:
        return 2
    text = str(value).strip().lower().rstrip("x")
    if text in ("2", "4"):
        return int(text)
    raise _BadRequest(f"upscale must be 2 or 4, got {value!r}")


def _vector_mode(intent: str | None) -> str:
    """Photographs trace badly; everything else uses the graphic preset."""
    return "photo" if intent in ("photo", "portrait", "product") else "graphic"


def generate_payload(body: dict) -> dict:
    """Generate images for an OpenAI-shaped request body.

    Shared by the RunPod handler (`/runsync`) and the HTTP shim (`/v1/images/generations`)
    so both entry points behave identically.
    """
    raw_prompt = (body.get("prompt") or "").strip()
    if not raw_prompt:
        return {"error": {"code": "invalid_request", "message": "prompt is required"}}

    blocked = check_prompt(raw_prompt)
    if blocked:
        return {"error": {"code": blocked, "message": "request rejected by content policy"}}

    count = max(1, min(int(body.get("n") or 1), 4))
    ratio = body.get("aspect_ratio")
    size = body.get("size")
    width, height = dimensions_for(ratio, size)

    steps = max(1, min(int(body.get("steps") or DEFAULT_STEPS), 16))
    guidance = float(body.get("guidance_scale") or DEFAULT_GUIDANCE)
    fmt = (body.get("format") or body.get("output_format") or DEFAULT_FORMAT).lower()
    if fmt not in ("png", "jpeg", "jpg", "webp", "svg"):
        return {"error": {"code": "invalid_request", "message": f"unsupported format: {fmt}"}}

    upscale_factor = 0
    if body.get("upscale"):
        try:
            upscale_factor = _parse_upscale(body.get("upscale"))
        except _BadRequest as exc:
            return {"error": {"code": "invalid_request", "message": str(exc)}}

    # `style` is accepted for compatibility with the older client shape and folded into
    # the prompt rather than forwarded as an upstream parameter (DALL-E-only field).
    prompt, intent = build_prompt(raw_prompt)
    if body.get("style"):
        prompt = f"{prompt}, {str(body['style']).strip()}"

    base_seed = int(body.get("seed") if body.get("seed") is not None else time.time_ns() % 2**31)

    budget = MAX_CANDIDATES
    per_image = max(1, min(BEST_OF, budget // count))

    pipe = _load()
    created = int(time.time())
    images = []
    diagnostics = []

    for index in range(count):
        candidates = []
        for attempt in range(per_image):
            seed = base_seed + index * 1000 + attempt
            image = _generate_one(pipe, prompt, width, height, steps, guidance, seed)
            candidates.append((technical_score(image), seed, image))

        best_score, best_seed, best_image = max(candidates, key=lambda entry: entry[0])

        stages_applied: list[dict] = []
        stage_notes: list[str] = []

        if upscale_factor:
            try:
                best_image, upscale_meta = stages.upscale(best_image, upscale_factor)
                stages_applied.append(upscale_meta)
            except stages.StageUnavailable as exc:
                stage_notes.append(f"upscale skipped: {exc}")
            except ValueError as exc:
                stage_notes.append(f"upscale skipped: {exc}")

        if fmt == "svg":
            try:
                payload, vector_meta = stages.vectorize(best_image, _vector_mode(intent))
                stages_applied.append(vector_meta)
                mime = "image/svg+xml"
            except stages.StageUnavailable as exc:
                stage_notes.append(f"svg skipped: {exc}")
                payload, mime = encode(best_image, DEFAULT_FORMAT, DEFAULT_QUALITY)
        else:
            payload, mime = encode(best_image, fmt, DEFAULT_QUALITY)

        images.append(
            {
                "b64_json": base64.b64encode(payload).decode("ascii"),
                "media_type": mime,
                "revised_prompt": prompt,
            }
        )
        diagnostics.append(
            {
                "index": index,
                "selected_seed": best_seed,
                "score": round(best_score, 4),
                "candidates": [{"seed": seed, "score": round(score, 4)} for score, seed, _ in candidates],
                "stages": stages_applied,
                "notes": stage_notes,
            }
        )

    _report_vram(f"after {count} image(s) x {per_image} candidate(s)")

    return {
        "created": created,
        "data": images,
        "imotan": {
            "model": MODEL_ID,
            "width": width,
            "height": height,
            "steps": steps,
            "format": fmt,
            "intent": intent,
            "best_of": per_image,
            "precision": _precision_label(),
            "text_precision": _active_text_precision if _active_text_precision != "unknown" else "bf16",
            "upscale": upscale_factor or None,
            "stages_available": {
                "upscale": stages.upscaler_available(),
                "vectorize": stages.vectorizer_available(),
            },
            "selection": diagnostics,
        },
    }


# Load before serving so the snapshot captured by FlashBoot includes a warm model.
_load()


def handler(job):
    """RunPod entry point: `POST /runsync` with {"input": {...}}."""
    return generate_payload(job.get("input") or {})


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
